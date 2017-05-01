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

def perturb_order_petsc(psi, dipole, Hl, Hr, d_dot_an, ef, time):
    # exp(t) * dipole * E(t) * psi * exp1(t)
    exp1 = Hr.copy()
    exp1.scale(-1j * time)
    exp1.exp()
    exp1.pointwiseMult(exp1, psi)
    exp = Hl.copy()
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
    global prototype
    global prototype_index
    prototype = da.import_prototype_from_file(
        join(hamiltonian_folder, "vector_prototype.dat"))
    prototype_index = da.prototype_as_multiindex(prototype)

    v = PETSc.Viewer().createBinary(
        join(hamiltonian_folder, "dipole_matrix.dat.gz"), 'r')
    D = PETSc.Mat().load(v)
    v.destroy()

    l_0_list, l_1_list, l_2_list = [],[],[]
    for i, (n,l,m,j,e) in enumerate(prototype):
        if l == 0:
            l_0_list += [i]
        if l == 1:
            l_1_list += [i]
        if l == 2:
            l_2_list += [i]
    is_0 = PETSc.IS().createGeneral(l_0_list)
    is_1 = PETSc.IS().createGeneral(l_1_list)
    is_2 = PETSc.IS().createGeneral(l_2_list + l_0_list)

    D_01 = D.createSubMatrix(is_1, is_0)
    D_12 = D.createSubMatrix(is_2, is_1)
    D_21 = D.createSubMatrix(is_1, is_2)

    v = PETSc.Viewer().createBinary(
        join(hamiltonian_folder, "energy_eigenvalues_vector.dat"), 'r')
    H = PETSc.Vec().load(v)

    H_0 = H.getSubVector(is_0)
    H_1 = H.getSubVector(is_1)
    H_2 = H.getSubVector(is_2)
    H_3 = H.getSubVector(is_1)

    psi_0_whole = D.createVecRight()
    psi_0_whole.setValue(0, 1)
    psi_0_whole.assemble()
    psi_1_whole = D.createVecRight()
    psi_1_whole.set(0)
    psi_1_whole.assemble()
    psi_2_whole = D.createVecRight()
    psi_2_whole.set(0)
    psi_2_whole.assemble()
    psi_3_whole = D.createVecLeft()
    psi_3_whole.set(0)
    psi_3_whole.assemble()
    n_1_mask_ = D.createVecLeft()
    n_1_mask_.set(1)
    n_1_mask_.setValue(0, 0)
    l_3_mask_ = D.createVecLeft()
    l_3_mask_.set(1)


    n_max = max([x[0] for x in prototype])

    for i, (n, l, m, j, e) in enumerate(prototype):
        if l == 0:
            if n_max - n < absorber_size[0]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[0]))**(1)
                n_1_mask_.setValue(i, val)
        if l == 2:
            if n_max - n < absorber_size[0]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[0]))**(1)
                n_1_mask_.setValue(i, val)
        if l == 1:
            if n_max - n < absorber_size[1]:
                val = np.sin(
                    (n_max - n) * np.pi / (2 * absorber_size[1]))**(1)
                l_3_mask_.setValue(i, val)
        if l == 3:
            l_3_mask_.setValue(i, 0)
    n_1_mask_.assemble()
    l_3_mask_.assemble()

    n_1_mask = n_1_mask_.getSubVector(is_2)
    l_3_mask = l_3_mask_.getSubVector(is_1)

    # logging.info(f"psi_0: size: {psi_0.getSize()}")
    # logging.info(f"vec is: \n{ print_vec(psi_0, prototype_index) }")
    # logging.info(f"psi_1: size: {psi_1.getSize()}")
    # logging.info(f"vec is: \n{ print_vec(psi_1, prototype_index) }")
    # logging.info(f"psi_2: size: {psi_2.getSize()}")
    # logging.info(f"vec is: \n{ print_vec(psi_2, prototype_index) }")
    # logging.info(f"psi_3: size: {psi_3.getSize()}")
    # logging.info(f"vec is: \n{ print_vec(psi_3, prototype_index) }")

    psi_0 = Wavefunction(psi_0_whole, prototype_index, H_0, "psi_0", l_0_list)
    psi_1 = Wavefunction(psi_1_whole, prototype_index, H_1, "psi_1", l_1_list)
    psi_2 = Wavefunction(psi_2_whole, prototype_index, H_2, "psi_2", l_0_list + l_2_list, n_1_mask)
    psi_3 = Wavefunction(psi_3_whole, prototype_index, H_3, "psi_3", l_1_list, l_3_mask)


    ret_dict = {
        "D": [D_01, D_12, D_21],
        "psis": [psi_0, psi_1, psi_2, psi_3],
    }
    # logging.debug(f"{ret_dict}")
    # logging.debug("leaving 'initialize_object'")
    return ret_dict

class Wavefunction:

    def __init__(self, psi, prototype, H, name, is_list=None, mask=None):
        self.prototype_whole = prototype
        self.prototype = None
        self.psi_whole = psi.copy()
        self.H = H
        self.psi = None
        self.name = name
        if is_list:
            self.is_list = is_list
            self.is_ = PETSc.IS().createGeneral(is_list)
            self.psi = self.psi_whole.getSubVector(self.is_)
            self.prototype = self.prototype_whole.take(is_list)
        self.mask = mask
        if mask:
            assert(self.mask.size == self.psi.size)
            assert(self.mask.size == self.H.size)

    def __repr__(self):
        return f"{self.name}: \nprint_vec(self.psi, self.prototype)"

    def mask(self):
        if self.mask:
            self.psi.pointwiseMult(self.psi, self.mask)

    def copy(self):
        return Wavefunction(self.psi_whole, self.prototype_whole, self.H, self.name, self.is_list, self.mask)




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
        if self.b < other.a or self.a > other.b:
            return None

        if self.a <= other.a and self.b <= other.b:
            return Interval(other.a, self.b)

        if self.a <= other.a and self.b >= other.b:
            return Interval(other.a, other.b)

        if self.a >= other.a and self.b >= other.b:
            return Interval(self.a, other.b)

        if self.a >= other.a and self.b <= other.b:
            return Interval(self.a, self.b)

    def union(self, other):
        if self.b < other.a or self.a > other.b:
            return None

        return Interval(
            min(self.a, self.b, other.a, other.b),
            max(self.a, self.b, other.a, other.b))


class DummyDict:
    def __init__(self, psi):
        self.vector = psi.psi.duplicate()
        self.vector.set(0)
        self.psi = psi.copy()

    def __getitem__(self, item):
        return self.vector

    def find_or_make_child(self, i):
        return self.vector


class Integration_Tree:
    def __init__(self,
                 limits,
                 convergence_criteria,
                 lower_order_tree,
                 psi,
                 integrand,
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
        self.integrand = integrand
        self.psi = psi.copy()
        self.my_depth = None

    def generate_children(self):
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
        f = self.integrand
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
        ) + f"Finding children for interval {self.limits}, with I = {self.I}, with h: {h}, for Node {self}, for wavefunction: {self.psi.name}"
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
            diffs = [Im1.find_or_make_child(i) for i in intervals]
            #logging.debug(f"diffs: { diffs }, { len(diffs) }")
            last = Im1.psi.psi
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
                self.integrand,
                self,
                I,
                order=self.order)
            for x0, x1, I in zip(loc_sets, loc_sets[1:], Is))
        #logging.debug("made children")

        self.is_converged(sum(Is))

        logging.info(f"{self.order}" + " " * self.depth() + f"is converged {self.converged} for interval {self.limits}")
        self.I = sum(Is)

        #logging.debug("free parents:")
        self.free_parents()


        return self.converged

    def generate_children_to_convergence(self):
        #logging.debug(
            #f"entered 'generate_children_to_convergence' for {self}:")
        if not self.generate_children() and self.depth() <= self.max_depth:
            for child in self.children:
                child.generate_children_to_convergence()

    def is_converged(self, I2):
        #logging.debug(f"entering 'is_converged' with {I2} for {self}")
        if self.I is None:
            return False
        self.converged = self.convergence_criteria(I2, self.I, self.limits, self.psi.prototype)
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

    def find_or_make_child(self, interval):
        #logging.debug(f"trying to find or make {interval} in {self}")
        if not isinstance(interval, Interval):
            interval = Interval(interval)
        if interval not in self.limits:
            raise IndexError(f"interval: {interval} not below this node")
        if interval == self.limits:
            #logging.debug(f"found as self returning breadth_first_sum")
            return self.breadth_first_sum()
        elif not self.children:
            self.generate_children()

        #logging.debug(
        #     f"trying to find or make {interval} in {list(map(lambda x: x.limits, self.children))}"
        # )
        for child in self.children:
            if interval in child.limits:
                return child.find_or_make_child(interval)


        # it isn't in the limits of the children, but it is in the limits of self.
        # so, we need to ask
        return sum(
            child.find_or_make_child(child.limits.intersection(interval))
            for child in self.children)

        # raise IndexError(
        #     f"interval: {interval} failed, not in {self.children[0].limits} or {self.children[1].limits}, for node: {self!r}"
        # )

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

    def convergence_criteria(Ia, Ib, interval, prototype_index = None):
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
            # if prototype_index:
            #     loc_rel = prototype_index.get_level_values("n")[
            #         i_rel], prototype_index.get_level_values("l")[i_rel]
            #     loc_abs = prototype_index.get_level_values("n")[
            #         i_abs], prototype_index.get_level_values("l")[i_abs]
            #     logging.info(
            #         f"max relative error for {interval} is {m_rel} at {loc_rel}, average is {avg_rel} and norm is {norm_rel} goal is {err_goal_rel}")
            #     logging.info(f"max absolute error for {interval} is {m_abs} compared to max {m} is {m_abs/m} at {loc_abs}, average is {avg_abs} and norm is {norm_abs} goal is {err_goal_abs}")
            #else:
            logging.info(
                f"max relative error for {interval} is {m_rel}, average is {avg_rel} and norm is {norm_rel} goal is {err_goal_rel}")
            logging.info(f"max absolute error for {interval} is {m_abs} compared to max {m} is {m_abs/m}, average is {avg_abs} and norm is {norm_abs} goal is {err_goal_abs}")
        # if mask_ is not None:
        #     error.pointwiseMult(error, mask_)
        #     i, m_rel = error.max()
        #     avg_rel = error.sum() / error.size
        #     norm_rel = error.norm()
        #     if rank == 0:
        #         if prototy
        #         loc = prototype_index.get_level_values("n")[
        #             i], prototype_index.get_level_values("l")[i]
        #         logging.info(
        #             f"after mask, max error for {interval} is {m_rel} at {loc}, average error is: {avg_rel} and norm is {norm_rel}"
        #             f"  err_goal is {err_goal_rel}")
        if math.isnan(m_rel):
            raise Exception("nan!")
        if avg_abs / m < err_goal_rel or avg_abs < err_goal_abs:
            return True
        else:
            return False

    return convergence_criteria


def recursive_integrate(fs, psis, limits, err_goal_rel, err_goal_abs):

    I_of_n = OrderedDict()

    # this allows us to generalize the algorithm without special cases.
    if 0 not in I_of_n:
        I_of_n[0] = DummyDict(psis[0])

    for o, psi, f in zip(count(1), psis[1:], fs):
        if o not in I_of_n:
            # if f.mask is not None:
            #     convergence_func = convergence_criteria_gen(
            #         err_goal_rel, err_goal_abs, None, o)
            # else:
            convergence_func = convergence_criteria_gen(err_goal_rel, err_goal_abs, None, o)
            I_of_n[o] = Integration_Tree(
                limits=limits,
                convergence_criteria=convergence_func,
                lower_order_tree=I_of_n[o - 1],
                psi=psi,
                integrand=f,
                order=o)
        I_of_n[o].generate_children_to_convergence()
        logging.info(f"I_of_n[o]: {I_of_n[o]}")
        #logging.info(f"psi: {print_vec(psi)}")
        logging.info(
            f"\n\n\n\n\n\n\n===============================================\n\n\n\n\n\n\n"
        )
    for o, psi in zip(count(1), psis[1:]):
        logging.info(f"I_of_n[o]: {I_of_n[o]}")
        logging.info(f"psi: {psi}")
        psi.psi += I_of_n[o].breadth_first_sum()


    return psis


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
    def __init__(self, psil, psir, D, efield, time, **kwargs):
        self.D = D
        self.v = self.D.createVecLeft()
        self.dt = time[1] - time[0]
        self.psil = psil
        self.psir = psir
        # if mask is not None:
        #     self.mask = mask
        self.ef_fn = kwargs["efield_fn"]

    def __call__(self, n, dn, psi):
        #logging.debug(
            #f"entering 'integrand' with arguments n: {n}, dn: {dn} and psi: {psi}"
        #)
        # #logging.debug(f"psi: ")
        # #logging.debug(print_vec(psi))
        t = float(n * self.dt)
        ef = self.ef_fn(t)
        # if rank == 0:
        #     logging.info(
        #         f"integrand: n = {n}, time = {t}, dt = {self.dt}, ef = {ef}, in_vec = {psi.size}, Hl= {self.psil.H.size}, Hr = {self.psir.H.size}")
        self.v = perturb_order_petsc(psi, self.D, self.psil.H, self.psir.H, self.v, ef,
                                     t) * self.dt
        if self.psil.mask is not None:
            self.v.pointwiseMult(self.v, self.psil.mask)
        # #logging.debug(f"integrand: ")
        # #logging.debug(print_vec(vv))
        #logging.debug(f"leaving 'integrand'")
        return self.v


def run_perturbation_calculation_recurse(D,
                                         psis,
                                         efield,
                                         time,
                                         zero_indices,
                                         steps=3*3,
                                         save_steps=1,
                                         **kwargs):
    #logging.debug(f"entering 'run_perturbation_calculation_recurse'")
    from itertools import islice
    pop_at_zero = {}

    #logging.debug("psis:")
    #for i, psi in enumerate(psis):
        #logging.debug(f"psi[{i}]")
        #logging.debug(print_vec(psi))

    integrands = [
        gen_integrand(psil, psir, D, efield, time, **kwargs)
        for D, psil, psir in zip(D, psis[1:], psis)
    ]

    for i, (efi, ti) in islice(
            enumerate(zip(efield, time)), steps, None, steps):

        psis = recursive_integrate(integrands, psis,
                                   Interval(i - steps, i), 1e-3, 1e-16)

        norms = [p.psi_whole.norm() for p in psis]
        if (i % 1 == 0 or i in zero_indices) and rank == 0:
            print(f"step: {i}, ef: {efi:1.2e}, t: {ti:4.1f}, "
                  f" psi_1_norm: {norms[1]:1.2e}, "
                  f"psi_2_norm: {norms[2]:1.2e}, psi_3_norm: {norms[3]:1.2e} ")
        if i % save_steps == 0:
            z = psis[1].psi_whole.copy()
            pop_at_zero[(1, i)] = vector_to_array_on_zero(z)
            zz = psis[2].psi_whole.copy()
            pop_at_zero[(2, i)] = vector_to_array_on_zero(zz)
            zzz = psis[3].psi_whole.copy()
            pop_at_zero[(3, i)] = vector_to_array_on_zero(zzz)
        break
    return pop_at_zero



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
    # prototype_index = options["prototype"][1]

    # mask_df = basis_vec_to_df_on_zero(options["masks"],
    #                                   prototype_index)
    # if 0 == PETSc.COMM_WORLD.getRank():
    #     intensity = da.Abinitio(efield_folder).laser.intensity
    #     cycles = da.Abinitio(efield_folder).laser.cycles
    #     if not key:
    #         key = f"p_{intensity:1.1e}_{cycles}".replace("+", "")
    #     key_run = key + "_absorbers"
    #     mask_df.to_hdf(join(output_folder, "perturbative.hdf"), key=key_run)
    pop_at_zero = run_perturbation_calculation_recurse(**options)
    if 0 == PETSc.COMM_WORLD.getRank():
        intensity = da.Abinitio(efield_folder).laser.intensity
        cycles = da.Abinitio(efield_folder).laser.cycles
        if not key:
            key = f"p_{intensity:1.1e}_{cycles}".replace("+", "")
        key_run = key + "_run"
        pop_df = pd.DataFrame(pop_at_zero, index=prototype_index)
        pop_df.to_hdf(join(output_folder, "perturbative.hdf"), key=key_run)


if __name__ == "__main__":
    setup_and_run()
