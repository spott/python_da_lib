#!/usr/local/bin/python3

import numpy as np
import pandas as pd
import data_analysis as da
import sys
from os.path import join
import click
from collections import OrderedDict
from itertools import islice, count
import math
from fractions import Fraction
import logging
from mpi4py import MPI



import petsc4py
petsc4py.init(sys.argv)

from petsc4py import PETSc

rank = PETSc.COMM_WORLD.getRank()

if rank == 0:
    logging.basicConfig(
        format=f"[{rank}]:" + '%(levelname)s:%(message)s', level=logging.INFO)

prototype_index = None
prototype = None


def find_zeros(t, x):
    positive = np.where(x > 0)[0]
    negative = np.where(x < 0)[0]
    zeros = sorted(
        list(np.intersect1d(positive, negative - 1)) +
        list(np.intersect1d(positive, negative + 1)))
    return (zeros, t[zeros])

@profile
def perturb_order_petsc(psi, dipole, H, d_dot_an, ef, time):
    # exp(t) * dipole * E(t) * psi * exp1(t)
    exp1 = H.copy()
    exp1.scale(-1j * time)
    exp1.exp()
    exp1.pointwiseMult(exp1, psi)
    exp = H.copy()
    exp.scale(1j * time)
    exp.exp()
    d_dot_an = dipole.createVecLeft()
    dipole.mult(exp1, d_dot_an)
    coef = 1j * ef
    d_dot_an.pointwiseMult(exp, d_dot_an)
    d_dot_an.scale(coef)
    exp.destroy()
    exp1.destroy()
    return d_dot_an


def vector_to_array_on_zero(vs):
    ##logging.debug(f"entering 'vector_to_array_on_zero' with argument {vs}")
    from collections.abc import Iterable
    if not isinstance(vs, Iterable):
        vs = [vs]

    results = []
    for v in vs:
        scatter, U0 = PETSc.Scatter.toZero(v)
        scatter.scatter(v, U0, False, PETSc.Scatter.Mode.FORWARD)
        if rank == 0:
            ##logging.debug("appending results")
            results.append(U0.getArray().copy())
            ##logging.debug(f"results now are: {results}")

        scatter.destroy()
        U0.destroy()
    ##logging.debug("leaving 'vector_to_array_on_zero'")
    return results


def initialize_objects(hamiltonian_folder, absorber_size=[200, 200]):
    #logging.debug("entering 'initialize_objects'")
    v = PETSc.Viewer().createBinary(
        join(hamiltonian_folder, "dipole_matrix.dat.gz"), 'r')
    D = PETSc.Mat().load(v)
    v.destroy()
    v = PETSc.Viewer().createBinary(
        join(hamiltonian_folder, "energy_eigenvalues_vector.dat"), 'r')
    H = PETSc.Vec().load(v)
    psi_0 = D.createVecRight()
    psi_0.setValue(0, 1)
    psi_0.assemble()
    psi_1 = D.createVecRight()
    psi_1.set(0)
    psi_1.assemble()
    psi_2 = D.createVecRight()
    psi_2.set(0)
    psi_2.assemble()
    psi_3 = D.createVecRight()
    psi_3.set(0)
    psi_3.assemble()
    n_1_mask = D.createVecRight()
    n_1_mask.set(1)
    n_1_mask.setValue(0, 0)
    l_3_mask = D.createVecRight()
    l_3_mask.set(1)

    global prototype
    global prototype_index
    prototype = da.import_prototype_from_file(
        join(hamiltonian_folder, "vector_prototype.dat"))
    prototype_index = da.prototype_as_multiindex(prototype)

    # if rank == 0:
    #     #logging.debug("prototype_index")
    #     #logging.debug(prototype_index)

    n_max = max([x[0] for x in prototype])

    for i, (n, l, m, j, e) in enumerate(prototype):
        if l == 0:
            if n_max - n < absorber_size[0]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[0]))**(1)
                n_1_mask.setValue(i, val)
        if l == 2:
            if n_max - n < absorber_size[0]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[0]))**(1)
                n_1_mask.setValue(i, val)
        if l == 1:
            if n_max - n < absorber_size[1]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[1]))**(1)
                l_3_mask.setValue(i, val)
        if l == 3:
            l_3_mask.setValue(i, 0)
    n_1_mask.assemble()
    l_3_mask.assemble()

    #logging.debug(f"psi_0: size: {psi_0.getSize()}")
    #logging.debug(f"vec is: \n{ print_vec(psi_0, prototype_index) }")
    #logging.debug(f"psi_1: size: {psi_1.getSize()}")
    #logging.debug(f"vec is: \n{ print_vec(psi_1, prototype_index) }")
    #logging.debug(f"psi_2: size: {psi_2.getSize()}")
    #logging.debug(f"vec is: \n{ print_vec(psi_2, prototype_index) }")
    #logging.debug(f"psi_3: size: {psi_3.getSize()}")
    #logging.debug(f"vec is: \n{ print_vec(psi_3, prototype_index) }")

    ret_dict = {
        "D": D,
        "H": H,
        "psis": [psi_0, psi_1, psi_2, psi_3],
        "masks": [n_1_mask, l_3_mask],
        "prototype": (prototype, prototype_index)
    }
    ##logging.debug(f"{ret_dict}")
    #logging.debug("leaving 'initialize_object'")
    return ret_dict


def get_efield(folder):
    import scipy.interpolate
    efield = da.Abinitio(folder).laser.efield
    t = efield.time
    ef = efield.tdata
    ef_func = scipy.interpolate.CubicSpline(t, ef)
    zero_indices, zeros = find_zeros(t, ef)
    return {
        "zero_indices": zero_indices,
        "efield": ef,
        "time": t,
        "efield_fn": ef_func
    }


def dipole(psi, D, psi2=None):
    temp = psi.copy()
    if psi2:
        D.mult(psi2, temp)
    else:
        D.mult(psi, temp)
    return psi.dot(temp)


class Interval:
    def __init__(self, a, b=None):
        c = a
        d = b
        if isinstance(a, tuple) and len(a) == 2:
            c = a[0]
            d = a[1]
        if c > d:
            raise ValueError("not ordered")
        self.a = Fraction(c).limit_denominator()
        self.b = Fraction(d).limit_denominator()

    def __contains__(self, item):
        return self.a <= item.a and self.b >= item.b

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b

    def __bool__(self):
        return not self.a == self.b

    def __repr__(self):
        return f"<Interval: ({self.a}, {self.b})>"

    def __lt__(self, other):
        return self.a < other.a

    def __gt__(self, other):
        return self.a > other.a

    def intersection(self, other):
        #logging.debug(f"Interval::intersection: {self}, {other}")
        if self.b < other.a or self.a > other.b:
            return None

        if self.a <= other.a and self.b <= other.b:
            #logging.debug(f"Interval::intersection: return {other.a},{self.b}")
            return Interval(other.a, self.b)

        if self.a <= other.a and self.b >= other.b:
            #logging.debug(
                #f"Interval::intersection: return {other.a},{other.b}")
            return Interval(other.a, other.b)

        if self.a >= other.a and self.b >= other.b:
            #logging.debug(f"Interval::intersection: return {self.a},{other.b}")
            return Interval(self.a, other.b)

        if self.a >= other.a and self.b <= other.b:
            #logging.debug(f"Interval::intersection: return {self.a},{self.b}")
            return Interval(self.a, self.b)

    def union(self, other):
        #logging.debug(f"Interval::union: {self}, {other}")
        if self.b < other.a or self.a > other.b:
            return None

        return Interval(
            min(self.a, self.b, other.a, other.b),
            max(self.a, self.b, other.a, other.b))


class IntegrationMethod:
    pass


class Box(IntegrationMethod):
    order = 1

    @staticmethod
    def __call__(vs, dt, step):
        return vs[0] * dt


class Trapezoid(IntegrationMethod):
    order = 2

    @staticmethod
    def __call__(vs, dt, step):
        return (vs[0] + vs[1]) * dt / 2


class Simpsons(IntegrationMethod):
    order = 3

    @staticmethod
    def __call__(vs, dt, step):
        if step % 2 == 0:
            return (vs[0] + vs[1] * 4 + vs[2]) * dt / 3
        else:
            return 0


class Simpsons_3_8(IntegrationMethod):
    order = 4

    @staticmethod
    def __call__(vs, dt, step):
        if step % 3 == 0:
            return (vs[0] + vs[1] * 3 + vs[2] * 3 + vs[3]) * dt * 3. / 8.
        else:
            return 0


class DummyDict:
    def __init__(self, psi):
        ##logging.debug(f"entering 'DummyDict::__init__' with argument {vector}")
        self.vector = psi.duplicate()
        self.vector.set(0)
        self.psi = psi.copy()
        # #logging.debug(print_vec(self.vector))
        # #logging.debug(f"leaving 'DummyDict::__init__'")

    def __getitem__(self, item):
        return self.vector

    def find_or_make_child(self, i, f):
        # #logging.debug("entering 'DummyDict::find_or_make_child': ")
        # #logging.debug(print_vec(self.vector))
        return self.vector


class Integration_Tree:
    def __init__(self,
                 limits,
                 convergence_criteria,
                 lower_order_tree,
                 psi,
                 parent=None,
                 I=None,
                 order=0,
                 max_depth=6,
                 dipole=None):
        self.parent = parent
        self.I = I
        self.dipole = dipole
        self.limits = limits if isinstance(limits,
                                           Interval) else Interval(limits)
        self.converged = False
        self.convergence_criteria = convergence_criteria
        self.children = None
        self.lower_order_tree = lower_order_tree
        self.order = order
        self.max_depth = max_depth
        self.psi = psi.copy()
        self.my_depth = None

    @profile
    def generate_children(self, f):
        if self.children:
            return self.converged
        a, b = self.limits.a, self.limits.b
        Im1 = self.lower_order_tree

        # order = 2
        # sets = 2
        # steps = order * sets
        # h = Fraction((b - a), steps)
        # overall = h / 12
        # coefficients = [1, 4, 1]
        # total_coefficients = [
        #     overall * coefficient for coefficient in coefficients
        # ]
        # locs = [i * h for i in range(order + 1)]
        # loc_sets = [a + i * order * h for i in range(sets + 1)]

        # intervals = [
        #     Interval(a + (i - 1) * h, a + i * h) for i in range(1, steps + 1)
        # ]

        order = 3
        sets = 3
        steps = order * sets
        h = Fraction((b - a), steps)
        overall = h * 3 / 4
        coefficients = [1, 3, 3, 1]
        total_coefficients = [
            overall * coefficient for coefficient in coefficients
        ]
        locs = [i * h for i in range(order + 1)]
        loc_sets = [a + i * order * h for i in range(sets + 1)]

        intervals = [
            Interval(a + (i - 1) * h, a + i * h) for i in range(1, steps + 1)
        ]

        logging.info(f"{self.order}" + " " * self.depth(
        ) + f"Finding children for interval {self.limits}, with I = {self.I}, with h: {h}, for Node {self}"
                     )
        logging.info(f"{self.order}" + " " * self.depth(
        ) + f" need intervals: {intervals} for order: {order}, sets: {sets}, h: {h}, total_coefficients: {total_coefficients}, locs: {locs}, loc_sets: {loc_sets}"
                     )

        #logging.debug(f"psi is: {self.psi}")

        Is = []
        dipole_moment = []

        # for x in range(100, 1100, 100):
        #     mask = Im1.psi.duplicate()
        #     mask.set(1)

        #     global prototype
        #     for i, (n,l,m,j,e) in enumerate(prototype):
        #         if n > x:
        #             mask.setValue(i,0)

        def generate_psi():
            diffs = [Im1.find_or_make_child(i, f) for i in intervals]
            #logging.debug(f"diffs: { diffs }, { len(diffs) }")
            last = Im1.psi
            for x in range(steps):
                #last.pointwiseMult(last, mask)
                yield last
                last += diffs[x]
                if x % order == 0:
                    #last.pointwiseMult(last, mask)
                    yield last

        psis = list(generate_psi())

        #simpsons 3/8 rule:
        Is = [
            sum(
                f(y + x0, h, psi) * coef
                for y, psi, coef in zip(locs, psis, total_coefficients))
            for x0 in loc_sets[:-1]
        ]

        #get the order-1 psi:

        #get current psi:
        # root = self.root()
        # root_value = root.breadth_first_sum(Interval(root.limits.a,self.limits.a))
        # psi_new = [self.psi + root_value + sum(Is[:n]) for n in range(1,len(Is)+1)]

        # dipole_moment = [dipole(psi_old, f.D, psi) for psi_old, psi in zip(psis[order::order], psi_new)]

        # logging.info(f"{self.order}" + " " * self.depth() +  f"{ len(psis) }, { len(psi_new) }, {len(Is)}, dipole = {dipole_moment}, last_dipole = {self.dipole}")

        # #logging.debug(f"converging simpsons_vs_simpsons_38:")
        # self.is_converged(sum(Is_1), sum(Is))
        # #logging.debug(f"is converged {self.converged}")

        # the integral of the whole range for this order:

        self.children = tuple(
            Integration_Tree(
                (x0, x1),
                self.convergence_criteria,
                self.lower_order_tree,
                self.psi,
                self,
                I,
                order=self.order)
            for x0, x1, I in zip(loc_sets, loc_sets[1:], Is))
        #logging.debug("made children")

        self.is_converged(self.I, sum(Is))

        logging.info(f"{self.order}" + " " * self.depth() + f"is converged {self.converged} for interval {self.limits}")
        self.I = sum(Is)

        #logging.debug("free parents:")
        self.free_parents()


        # if self.converged and all(child.converged for child in self.parent.children):
        #     self.parent.converged = True
        #     self.parent.I = sum(child.I for child in self.parent.children)
        #logging.debug("leaving 'generate_children'")
        return self.converged

    def generate_children_to_convergence(self, f):
        #logging.debug(
            #f"entered 'generate_children_to_convergence' for {self}:")
        if not self.generate_children(f) and self.depth() <= self.max_depth:
            for child in self.children:
                child.generate_children_to_convergence(f)
        # elif not self.generate_children(f):
        #     return self.children[0].generate_children_to_convergence(f) and self.children[1].generate_children_to_convergence(f)
        # elif not self.children:
        #     return self.children[0].generate_children_to_convergence(f) and self.children[1].generate_children_to_convergence(f)
        # if not self.generate_children(f):
        #     return self.children[0].generate_children_to_convergence(f) and self.children[1].generate_children_to_convergence(f)
        #logging.debug("exiting 'generate_children_to_convergence'")

    def is_converged(self, I1, I2):
        #logging.debug(f"entering 'is_converged' with {I2} for {self}")
        if self.I is None:
            return False
        self.converged = self.convergence_criteria(I2, I1, self.limits)
        logging.info(f"{self.order}" + " " * self.depth(
        ) + f"converged? {self.converged!r}, {self.limits}, order: {self.order}"
                     )
        return self.converged

    def free(self):
        if self.children and (
                all(map(lambda x: x.children is not None, self.children))):
            self.I.destroy()

    def free_parents(self):
        if not self.parent or not self.parent.parent:
            return
        parent = self.parent

        # if parent is to the left, and the right has children,
        #
        if all(
                map(lambda x: x.children is not None,
                    self.parent.parent.children)):
            self.parent.parent.I.destroy()
        # if parent is self.parent.parent.children[0] and self.parent.parent.children[1].children is not None:
        #     self.parent.parent.I.destroy()
        # elif parent is self.parent.parent.children[1] and self.parent.parent.children[0].children is not None:
        #     self.parent.parent.I.destroy()
        else:
            return

    def depth(self):
        if self.my_depth:
            return self.my_depth

        if self.parent is None:
            self.my_depth = 1
            return self.my_depth
        else:
            self.my_depth = 1 + self.parent.depth()
            return self.my_depth

    def root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.root()

    def __getitem__(self, interval):
        if not isinstance(interval, Interval):
            interval = Interval(interval)
        if interval == self.limits:
            return self
        elif not self.children:
            raise IndexError(
                "interval: {interval} not below this node: {self!r}")
        for child in children:
            if interval in self.child:
                return self.children[0][interval]
        # else:
        #     raise IndexError(
        #         "interval: {interval} not below this node: {self!r}")

    def __contains__(self, interval):
        if self.limits == interval:
            return True
        if not self.children:
            return False
        else:
            return any(map(lambda x: interval in x, self.children))

    def find_or_make_child(self, interval, f):
        #logging.debug(f"trying to find or make {interval} in {self}")
        if not isinstance(interval, Interval):
            interval = Interval(interval)
        if interval not in self.limits:
            raise IndexError(f"interval: {interval} not below this node")
        if interval == self.limits:
            #logging.debug(f"found as self returning breadth_first_sum")
            return self.breadth_first_sum()
        elif not self.children:
            self.generate_children(f)

        #logging.debug(
        #     f"trying to find or make {interval} in {list(map(lambda x: x.limits, self.children))}"
        # )
        for child in self.children:
            if interval in child.limits:
                return child.find_or_make_child(interval, f)
        # if interval in self.children[0].limits:
        #     return self.children[0].find_or_make_child(interval, f)
        # if interval in self.children[1].limits:
        #     return self.children[1].find_or_make_child(interval, f)

        # it isn't in the limits of the children, but it is in the limits of self.
        # so, we need to ask
        return sum(
            child.find_or_make_child(child.limits.intersection(interval), f)
            for child in self.children)

        # raise IndexError(
        #     f"interval: {interval} failed, not in {self.children[0].limits} or {self.children[1].limits}, for node: {self!r}"
        # )

    @profile
    def breadth_first_sum(self, interval=None):
        ##logging.debug(f"entering 'breadth_first_sum': with {interval}")
        if self.children is None and interval is None:
            ##logging.debug(f"no children, returning I.  {self!r}")
            return self.I
        elif self.children is None and interval == self.limits:
            return self.I
        elif interval is None:
            return sum(child.breadth_first_sum() for child in self.children)
        elif interval is not None and self.children is not None:
            return sum(child.breadth_first_sum(child.limits.intersection(interval)) for child in self.children if child.limits.intersection(interval))
        else:
            return 0

    def repr_helper(self, level=0):
        rep = f"<Node: {self.I}, limits: {self.limits}, order: {self.order}"
        repend = "\t" * level if self.children else ""
        repend += ">" if not self.children else ""
        repc = ""
        if self.children:
            for i, child in enumerate(self.children):
                repc = "\n" + '\t' * (
                    level + 1) + f"{i}: {child.repr_helper(level+1)},"
            repc += ">"
        return rep + repc + repend

    def __repr__(self):
        return self.repr_helper()


def convergence_criteria_gen(err_goal_rel, err_goal_abs, mask, order):
    mask_ = None
    if mask is not None:
        mask_ = mask
    order_ = order

    @profile
    def convergence_criteria(Ia, Ib, interval):
        #logging.debug(
            #f"entering 'convergence_cirteria' with Ia: {Ia}, Ib: {Ib}")
        #logging.debug(f"Ia: ")
        #logging.debug(print_vec(Ia))
        #logging.debug(f"Ib: ")
        #logging.debug(print_vec(Ib))
        error = Ia.copy()
        error.abs()
        _, ma = error.max()
        error = Ib.copy()
        error.abs()
        _, mb = error.max()
        m = max(ma, mb)
        error = Ia.copy()
        error -= Ib
        #logging.debug(f"error: ")
        #logging.debug(print_vec(error))
        error.abs()
        i_abs, m_abs = error.max()
        avg_abs = error.sum()
        norm_abs = error.norm()
        #logging.debug(f"abs error: ")
        #logging.debug(print_vec(error))
        error /= abs(Ia)
        array = error.getArray()
        array = np.nan_to_num(array)
        nonzero_ = np.array(np.count_nonzero(array), 'i')
        nonzero = np.array(0.0,'i')
        MPI.COMM_WORLD.Allreduce([nonzero_, MPI.INT],
        [nonzero, MPI.INT],
        op=MPI.SUM)

        error.setArray(array)
        #logging.debug(f"rel_error: ")
        #logging.debug(print_vec(error))
        i_rel, m_rel = error.max()
        avg_rel = error.sum() / nonzero
        avg_abs = np.real(avg_abs / nonzero)

        norm_rel = error.norm()
        if rank == 0:
            loc_rel = prototype_index.get_level_values("n")[
                i_rel], prototype_index.get_level_values("l")[i_rel]
            loc_abs = prototype_index.get_level_values("n")[
                i_abs], prototype_index.get_level_values("l")[i_abs]
            logging.info(
                f"max relative error for {interval} is {m_rel} at {loc_rel}, average is {avg_rel} and norm is {norm_rel} goal is {err_goal_rel}")
            logging.info(f"max absolute error for {interval} is {m_abs} compared to max {m} is {m_abs/m} at {loc_abs}, average is {avg_abs} and norm is {norm_abs} goal is {err_goal_abs}")
        if mask_ is not None:
            error.pointwiseMult(error, mask_)
            i, m_rel = error.max()
            avg_rel = error.sum() / error.size
            norm_rel = error.norm()
            if rank == 0:
                loc = prototype_index.get_level_values("n")[
                    i], prototype_index.get_level_values("l")[i]
                logging.info(
                    f"after mask, max error for {interval} is {m_rel} at {loc}, average error is: {avg_rel} and norm is {norm_rel}"
                    f"  err_goal is {err_goal_rel}")
        if math.isnan(m_rel):
            raise Exception("nan!")
        if avg_abs / m < err_goal_rel or avg_abs < err_goal_abs:
            return True
        else:
            return False

    return convergence_criteria


def recursive_integrate(fs, psis, limits, err_goal_rel, err_goal_abs):
    #logging.debug(
        #f"entering 'recursive_integrate' with arguments: f: {fs}, psis: {psis}, limits: {limits}, err_goal_rel: {err_goal_rel}, err_goal_abs: {err_goal_abs}"
    #)

    I_of_n = OrderedDict()

    # this allows us to generalize the algorithm without special cases.
    if 0 not in I_of_n:
        I_of_n[0] = DummyDict(psis[0])

    # for o, ( psi, f ) in zip(count(1), psis[1:], fs)):
    #     if o + 1 not in I_of_n:
    #         convergence_func = convergence_criteria_gen(err_goal)
    #         I_of_n[o + 1] = Integration_Tree(
    #             limits=limits,
    #             convergence_criteria=convergence_func,
    #             lower_order_tree=I_of_n[o],
    #             psi=psi,
    #             order=o + 1)

    for o, psi, f in zip(count(1), psis[1:], fs):
        if o not in I_of_n:
            if f.mask is not None:
                convergence_func = convergence_criteria_gen(
                    err_goal_rel, err_goal_abs, None, o)
            else:
                convergence_func = convergence_criteria_gen(err_goal_rel, err_goal_abs, None, o)
            I_of_n[o] = Integration_Tree(
                limits=limits,
                convergence_criteria=convergence_func,
                lower_order_tree=I_of_n[o - 1],
                psi=psi,
                order=o)
        I_of_n[o].generate_children_to_convergence(f)
        logging.info(f"I_of_n[o]: {I_of_n[o]}")
        logging.info(f"psi: {print_vec(psi)}")
        logging.info(
            f"\n\n\n\n\n\n\n===============================================\n\n\n\n\n\n\n"
        )
    for o, psi in zip(count(1), psis[1:]):
        logging.info(f"I_of_n[o]: {I_of_n[o]}")
        logging.info(f"psi: {print_vec(psi)}")
        psi += I_of_n[o].breadth_first_sum()


    # cleanup:
    #logging.debug(f"leaving 'recursive_integrate' with arguments:")
    return psis


# def recursive_integrate_single(f, psi, order, na, nb, err_goal, higher_I=None):
#     h = (nb - na) / 4.

#     o = order

#     I_of_n = recursive_integrate.I_of_n

#     I_of_n[o][na + 2 * h] = (
#         f(na, psi) + f(na + h, psi + I_of_n[o - 1][na + h]) * 4 + f(
#             na + 2 * h, psi + I_of_n[o - 1][na + 2 * h])) * (nb - na) / 3.
#     # the integral of the second half of the range, which calls the
#     I_of_n[o][nb] = (f(na + 2 * h, psi + I_of_n[o - 1][na + 2 * h]) +
#                      f(na + 3 * h, psi + I_of_n[o - 1][na + 3 * h]) * 4 + f(
#                          nb, psi + I_of_n[o - 1][nb])) * (nb - na) / 3.
#     # the integral of the whole range for this order:
#     Iab = I_of_n[o][na + 2 * h] + I_of_n[o][nb]

#     # for this order, the error:
#     if higher_I:
#         error = higher_I - Iab
#         error.abs()
#         error /= abs(Iab)
#         array = error.getArray()
#         array = np.nan_to_num(array)
#         error.setArray(array)
#         _, m = error.max()
#         if rank == 0:
#             print(f"max error for integral between {na} and {nb} is {m}."
#                   f"  err_goal is {err_goal}")
#         if math.isnan(m):
#             raise Exception("nan!")
#         if m < err_goal:
#             return Iab

#     return recursive_integrate(f, psi, order, na, (na + nb) / 2, err_goal,
#                                I_of_n[o][na + 2 * h]) + recursive_integrate(
#                                    f, psi + I_of_n[o - 1][na + 2 * h], order,
#                                    (na + nb) / 2, nb, err_goal, I_of_n[o][nb])

# def recursive_integrate(fs, psis, na, nb, err_goal, higher_I=None):
#     import math

#     if higher_I is None:
#         higher_I = [None for _ in psis]

#     h = (nb - na) / 4.

#     # zeroth order psi doesn't depend on time, so psi[1] doesn't
#     # care about integrand time, but higher order psis do.

#     # dict of OrderedDicts to hold the temporary values of the integration.
#     # a static variable, will need to be cleaned up when we completely finish:
#     I_of_n = recursive_integrate.I_of_n

#     # this allows us to generalize the algorithm without special cases.
#     if 0 not in I_of_n:
#         I_of_n[0] = DummyDict(psis[0])

#     # we skip the zeroth
#     for o, (f, psi) in islice(enumerate(zip(fs, psis)), 1, None):
#         # we build a orderedDict to hold the integrated points we find for
#         # the perturbative order we are on:
#         if o not in I_of_n:
#             I_of_n[o] = OrderedDict()

#         # the integral of the first half of the range
#         I_of_n[o][na + 2 * h] = (
#             f(na, psi) + f(na + h, psi + I_of_n[o - 1][na + h]) * 4 + f(
#                 na + 2 * h, psi + I_of_n[o - 1][na + 2 * h])) * (nb - na) / 3.
#         # the integral of the second half of the range, which calls the
#         I_of_n[o][nb] = (f(na + 2 * h, psi + I_of_n[o - 1][na + 2 * h]) +
#                          f(na + 3 * h, psi + I_of_n[o - 1][na + 3 * h]) * 4 +
#                          f(nb, psi + I_of_n[o - 1][nb])) * (nb - na) / 3.
#         # the integral of the whole range for this order:
#         Iab = I_of_n[o][na + 2 * h] + I_of_n[o][nb]

#         # for this order, the error:
#         if higher_I:
#             error = higher_I - Iab
#             error.abs()
#             error /= abs(Iab)
#             array = error.getArray()
#             array = np.nan_to_num(array)
#             error.setArray(array)
#             _, m = error.max()
#             if rank == 0:
#                 print(f"max error for integral between {na} and {nb} is {m}."
#                       f"  err_goal is {err_goal}")
#             if math.isnan(m):
#                 raise Exception("nan!")
#             if m < err_goal:
#                 return Iab

#         return recursive_integrate(f, psi, na, (na + nb) / 2, err_goal,
#                                    I_of_n[na + 2 * h]) + recursive_integrate(
#                                        f, psi + I_of_n[o - 1][na + 2 * h],
#                                        (na + nb) / 2, nb, err_goal, I_of_n[nb])


def copy_old_vectors(v):
    if len(v) == 1:
        return
    else:
        vv = reversed(list(zip(v[:-1], v[1:])))
        for v1, v2 in vv:
            v1.copy(v2)
    return


def basis_vec_to_df_on_zero(v, prototype):
    #logging.debug(f"entering 'basis_vec_to_df_on_zero' with argument {v}")
    vs = vector_to_array_on_zero(v)
    if rank == 0:
        ##logging.debug(f"vs: {vs.shape}")
        #logging.debug(f"{vs}")
        df = pd.DataFrame(vs).T
        df.index = prototype
        #logging.debug(f"leaving 'basis_vec_to_df_on_zero' {df!r}")
        return df
    else:
        #logging.debug(f"leaving 'basis_vec_to_df_on_zero'")
        return None


def print_vec(v, index=None):
    ##logging.debug(f"entering 'print_vec' with argument {v}")
    vs = vector_to_array_on_zero(v)
    #if rank == 0:
    ##logging.debug(f"vs: {vs}, vs[0]: {vs[0]}, size: {len(vs)}, shape: {vs[0].shape}")
    if rank == 0:
        if len(vs) == 1:
            vs = vs[0]
        df = pd.Series(vs).T
        ##logging.debug(f"df shape: {df.shape}")
        if index is not None:
            ##logging.debug(f"index shape: {index.shape}")
            df.index = index
        else:
            ##logging.debug(f"index shape: {prototype_index.shape}")
            df.index = prototype_index
        ##logging.debug("leaving 'print_vec'")
        return df.loc[df != 0]
    ##logging.debug("leaving 'print_vec'")
    return "Not on this rank"


def check_phase_change(v1, v2, prototype):
    v_new = v1.copy()
    v_new_2 = v2.copy()
    with v_new as array1, v_new_2 as array2:
        array2 += array1
        array2 = np.angle(array2)
        array1 = np.angle(array1)
        array2 -= array1
    i, m = v_new_2.max()
    i2, m2 = v_new_2.min()
    v_new.destroy()
    v_new_2.destroy()
    if abs(m) > abs(m2):
        return prototype[i], m
    else:
        return prototype[i2], m2


def format_phase_change(pc):
    return f"( { pc[0][0] }, { pc[0][1] }, {float(np.real(pc[0][4])):.2g} ): {pc[1]: 1.2e}"


class gen_integrand:
    def __init__(self, p, D, H, efield, time, mask=None, **kwargs):
        self.v = p.duplicate()
        self.dt = time[1] - time[0]
        self.mask = None
        if mask is not None:
            self.mask = mask
        self.ef_fn = kwargs["efield_fn"]
        self.D = D
        self.H = H

    @profile
    def __call__(self, n, dn, psi):
        #logging.debug(
            #f"entering 'integrand' with arguments n: {n}, dn: {dn} and psi: {psi}"
        #)
        # #logging.debug(f"psi: ")
        # #logging.debug(print_vec(psi))
        t = float(n * self.dt)
        ef = self.ef_fn(t)
        #if rank == 0:
            #logging.debug(
                #f"integrand: n = {n}, time = {t}, dt = {self.dt}, ef = {ef}")
        self.v = perturb_order_petsc(psi, self.D, self.H, self.v, ef,
                                     t) * self.dt
        if self.mask is not None:
            self.v.pointwiseMult(self.v, self.mask)
        # #logging.debug(f"integrand: ")
        # #logging.debug(print_vec(vv))
        #logging.debug(f"leaving 'integrand'")
        return self.v


def run_perturbation_calculation_recurse(D,
                                         H,
                                         psis,
                                         masks,
                                         efield,
                                         time,
                                         zero_indices,
                                         steps=3*3,
                                         save_steps=1,
                                         integration_method=Trapezoid(),
                                         **kwargs):
    #logging.debug(f"entering 'run_perturbation_calculation_recurse'")
    from itertools import islice
    pop_at_zero = {}

    #logging.debug("psis:")
    #for i, psi in enumerate(psis):
        #logging.debug(f"psi[{i}]")
        #logging.debug(print_vec(psi))

    integrands = [gen_integrand(psis[0], D, H, efield, time, **kwargs)] + [
        gen_integrand(psis[0], D, H, efield, time, mask, **kwargs)
        for mask in masks
    ]

    for i, (efi, ti) in islice(
            enumerate(zip(efield, time)), steps, None, steps):

        psis = recursive_integrate(integrands, psis,
                                   Interval(i - steps, i), 1e-3, 1e-16)

        # temp = integration_method(v, dt, i)
        # copy_old_vectors(v)
        # v2[0] = recursive_integrate(integrand_1, i - steps, i, 1e-2)
        # perturb_order_petsc(psis[1], D, H, v2[0], efi, ti)
        # # remove the ground state...
        # v2[0].pointwiseMult(v2[0], masks[0])
        # temp = integration_method(v2, dt, i)
        # phase[2] = check_phase_change(psis[2], temp, kwargs["prototype"][0])
        # psis[2] += temp
        # copy_old_vectors(v2)
        # v3[0] = perturb_order_petsc(psis[2], D, H, v3[0], efi, ti)
        # # remove l = 3 terms because they will be too large,
        # v3[0].pointwiseMult(v3[0], masks[1])
        # copy_old_vectors(v3)
        # #psis[3] += (v3[0] + v3[1]) * dt * steps / 2
        # # if i % 2 == 0:
        # #     psis[3] += (v3[0] + v3[1] * 4 + v3[2]) * dt / 3
        # temp = integration_method(v3, dt, i)
        # phase[3] = check_phase_change(psis[3], temp, kwargs["prototype"][0])
        # psis[3] += temp
        norms = [p.norm() for p in psis]
        if (i % 1 == 0 or i in zero_indices) and rank == 0:
            print(f"step: {i}, ef: {efi:1.2e}, t: {ti:4.1f}, "
                  f" psi_1_norm: {norms[1]:1.2e}, "
                  f"psi_2_norm: {norms[2]:1.2e}, psi_3_norm: {norms[3]:1.2e} ")
            #f"phases: {[format_phase_change(p) for p in phase[1:]]}")
        # if i in zero_indices:
        #     z = psis[1].copy()
        #     pop_at_zero[(1, indexOf(i,zero_indices))] = vector_to_array_on_zero(z)
        #     zz = psis[2].copy()
        #     pop_at_zero[(2, indexOf(i,zero_indices))] = vector_to_array_on_zero(zz)
        #     zzz = psis[3].copy()
        #     pop_at_zero[(3, indexOf(i,zero_indices))] = vector_to_array_on_zero(zzz)
        if i % save_steps == 0:
            z = psis[1].copy()
            pop_at_zero[(1, i)] = vector_to_array_on_zero(z)
            zz = psis[2].copy()
            pop_at_zero[(2, i)] = vector_to_array_on_zero(zz)
            zzz = psis[3].copy()
            pop_at_zero[(3, i)] = vector_to_array_on_zero(zzz)
        break
    return pop_at_zero


# def run_perturbation_calculation(D,
#                                  H,
#                                  psis,
#                                  masks,
#                                  efield,
#                                  time,
#                                  zero_indices,
#                                  steps=1,
#                                  save_steps=10,
#                                  integration_method=Trapezoid(),
#                                  **kwargs):
#     from itertools import islice
#     dt = time[steps] - time[0]
#     pop_at_zero = {}
#     num_terms = integration_method.order
#     v = [psis[0].duplicate() for x in range(0, num_terms)]
#     [vv.set(0) for vv in v]
#     v2 = [psis[2].duplicate() for x in range(0, num_terms)]
#     [vv.set(0) for vv in v2]
#     v3 = [psis[3].duplicate() for x in range(0, num_terms)]
#     [vv.set(0) for vv in v3]

#     temp = psis[0].duplicate()

#     phase = [0 for _ in psis]
#     integrand_0 = gen_integrand(psis[0], D, H, efield, time)
#     for i, (efi, ti) in islice(
#             enumerate(zip(efield, time)), None, None, steps):

#         #v[0] = perturb_order_petsc(psis[0], D, H, v[0], efi, ti)

#         v[0] = recursive_integrate(
#             integrand_0,
#             i - steps,
#             i, )

#         temp = integration_method(v, dt, i)
#         phase[1] = check_phase_change(psis[1], temp, kwargs["prototype"][0])
#         psis[1] += temp
#         copy_old_vectors(v)
#         v2[0] = perturb_order_petsc(psis[1], D, H, v2[0], efi, ti)
#         # remove the ground state...
#         v2[0].pointwiseMult(v2[0], masks[0])
#         temp = integration_method(v2, dt, i)
#         phase[2] = check_phase_change(psis[2], temp, kwargs["prototype"][0])
#         psis[2] += temp
#         copy_old_vectors(v2)
#         v3[0] = perturb_order_petsc(psis[2], D, H, v3[0], efi, ti)
#         # remove l = 3 terms because they will be too large,
#         v3[0].pointwiseMult(v3[0], masks[1])
#         copy_old_vectors(v3)
#         #psis[3] += (v3[0] + v3[1]) * dt * steps / 2
#         # if i % 2 == 0:
#         #     psis[3] += (v3[0] + v3[1] * 4 + v3[2]) * dt / 3
#         temp = integration_method(v3, dt, i)
#         phase[3] = check_phase_change(psis[3], temp, kwargs["prototype"][0])
#         psis[3] += temp
#         norms = [v[0].norm(), v2[0].norm(), v3[0].norm()
#                  ] + [p.norm() for p in psis]
#         # diffs = [(v2[0] - psis[1]).norm(), (v3[0] - psis[2]).norm()]
#         if (i % 1 == 0 or i in zero_indices) and rank == 0:
#             print(f"rank: {rank}\t"
#                   f"step: {i}, ef: {efi:1.2e}, t: {ti:4.1f}, "
#                   f"norm: {norms[0]:1.2e}, norm_2: {norms[1]:1.2e}, "
#                   f"norm_3: {norms[2]:1.2e}, psi_1_norm: {norms[4]:1.2e}, "
#                   f"psi_2_norm: {norms[5]:1.2e}, psi_3_norm: {norms[6]:1.2e} "
#                   f"phases: {[format_phase_change(p) for p in phase[1:]]}")
#         # if i in zero_indices:
#         #     z = psis[1].copy()
#         #     pop_at_zero[(1, indexOf(i,zero_indices))] = vector_to_array_on_zero(z)
#         #     zz = psis[2].copy()
#         #     pop_at_zero[(2, indexOf(i,zero_indices))] = vector_to_array_on_zero(zz)
#         #     zzz = psis[3].copy()
#         #     pop_at_zero[(3, indexOf(i,zero_indices))] = vector_to_array_on_zero(zzz)
#         if i % save_steps == 0:
#             z = psis[1].copy()
#             pop_at_zero[(1, i)] = vector_to_array_on_zero(z)
#             zz = psis[2].copy()
#             pop_at_zero[(2, i)] = vector_to_array_on_zero(zz)
#             zzz = psis[3].copy()
#             pop_at_zero[(3, i)] = vector_to_array_on_zero(zzz)

#     return pop_at_zero


@click.command()
@click.option("--hamiltonian_folder", type=str)
@click.option("--efield_folder", type=str)
@click.option(
    "--output_folder",
    type=str,
    default="~/Documents/Data/local_tests/perturbative/")
@click.option("--key", type=str, default=None)
def setup_and_run(hamiltonian_folder, efield_folder, output_folder, key):
    #logging.debug(f"rank: {rank} started")
    options = get_efield(efield_folder)
    #logging.debug(f"rank: {rank} got efield")
    options.update(initialize_objects(hamiltonian_folder))
    #logging.debug(f"rank: {rank} initialized objects")
    if rank == 0:
        print(f"zeros: {options['zero_indices']}")
    global prototype_index
    prototype_index = options["prototype"][1]

    mask_df = basis_vec_to_df_on_zero(options["masks"],
                                      options["prototype"][1])
    if 0 == PETSc.COMM_WORLD.getRank():
        intensity = da.Abinitio(efield_folder).laser.intensity
        cycles = da.Abinitio(efield_folder).laser.cycles
        if not key:
            key = f"p_{intensity:1.1e}_{cycles}".replace("+", "")
        key_run = key + "_absorbers"
        mask_df.to_hdf(join(output_folder, "perturbative.hdf"), key=key_run)
    pop_at_zero = run_perturbation_calculation_recurse(**options)
    if 0 == PETSc.COMM_WORLD.getRank():
        intensity = da.Abinitio(efield_folder).laser.intensity
        cycles = da.Abinitio(efield_folder).laser.cycles
        if not key:
            key = f"p_{intensity:1.1e}_{cycles}".replace("+", "")
        key_run = key + "_run"
        pop_df = pd.DataFrame(pop_at_zero, index=options["prototype"][1])
        pop_df.to_hdf(join(output_folder, "perturbative.hdf"), key=key_run)


if __name__ == "__main__":
    setup_and_run()
