#!/curc/tools/x86_64/rh6/software/python/3.5.2/gcc/5.1.0/bin/python3

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
mpi_size = PETSc.COMM_WORLD.getSize()



def dipole_moment(psil, dipole, psir):
    temp = psil.duplicate()
    dipole.mult(psir, temp)
    d = temp.dot(psil)
    temp.destroy()
    return d


def initialize_objects(hamiltonian_folder):
    global prototype
    global prototype_index
    prototype = da.import_prototype_from_file(
        join(hamiltonian_folder, "vector_prototype.dat"))
    prototype_index = da.prototype_as_multiindex(prototype)

    v = PETSc.Viewer().createBinary(
        join(hamiltonian_folder, "dipole_matrix.dat.gz"), 'r')
    D = PETSc.Mat().load(v)
    logging.info("D size: " + str(D.getSizes()) + " info: " + str(D.getInfo()))
    v.destroy()

    return D


def hdf_store_to_dataframe(store_filename, store_key):
    df = pd.read_hdf(store_filename, store_key)
    df.columns.set_names("order", inplace=True)
    df.set_index(["step", "time", "efield"], append=True, inplace=True)
    df = df.unstack(["step", "time", "efield"])
    df.index = prototype_index
    df = df.reorder_levels(["step", "time", "efield", "order"], axis=1)
    df.sort_index(level="step", axis=1, inplace=True)
    return df


def series_to_vec(df, vec_everyone):
    #vector for everyone:
    scatter, V0 = PETSc.Scatter.toZero(vec_everyone)

    if rank == 0:
        V0.setArray(df.values.copy())

    scatter.scatter(V0, vec_everyone, False, PETSc.Scatter.Mode.REVERSE)

    scatter.destroy()
    V0.destroy()

    return vec_everyone


def decomposition_energy_pair_to_IS(a, b):
    is_list = []
    for i, (n, l, j, m, e) in enumerate(prototype):
        if a <= e < b and i % mpi_size == rank:
            is_list += [i]

    return PETSc.IS().createGeneral(is_list)


def decomposition_n_pair_to_IS(a, b):
    is_list = []
    for i, (n, l, j, m, e) in enumerate(prototype):
        if a <= n < b and i % mpi_size == rank:
            is_list += [i]

    return PETSc.IS().createGeneral(is_list)


def find_dipole_moments(D, energy_splits, dataframe, out_store, out_store_key):
    from itertools import combinations_with_replacement

    IS_set = []
    for split in energy_splits:
        IS_set += [(split, decomposition_energy_pair_to_IS(*split))]

    combinations = list(combinations_with_replacement(IS_set, 2))

    store = pd.HDFStore(out_store)

    matrices = {}
    for (splita, is_a), (splitb, is_b) in combinations:
        matrices[(splita, splitb)] = D.createSubMatrix(is_a, is_b)

    psi0 = D.createVecLeft()
    psi0.setValue(0,1)
    psi1 = psi0.duplicate()
    psi2 = psi0.duplicate()
    psi3 = psi0.duplicate()

    logging.info(dataframe.head())

    dipoles = {}#((splita, splitb),"1+2"):
            #for (splita, _), (splitb, _) in combinations}
    time = []
    for step in dataframe.stack("order"):
        time += [step]
        logging.info("finding dipoles for time : {}".format(step))
        #logging.info(dataframe[step].head())
        # 2 + 1 and 3 + 0
        df = dataframe[step]

        psi1 = series_to_vec(df[1],psi1)
        psi2 = series_to_vec(df[2],psi2)
        psi3 = series_to_vec(df[3],psi3)

        for (splita, is1), (splitb, is2) in combinations:
            logging.info("finding dipoles for splits {} and {}".format(splita, splitb))
            D_part = matrices[(splita, splitb)]

            psi1_part = psi1.getSubVector(is1)
            psi2_part = psi2.getSubVector(is2)

            psi3_part = psi3.getSubVector(is1)
            psi0_part = psi0.getSubVector(is2)
            dipoles[((splita, splitb),"1+2")]= dipole_moment(
                psi1_part, D_part, psi2_part)
                   #dipoles[str((splita, splitb)) + "1+2"] = dipole_moment(
            #dipoles[str((splita, splitb)) + "3+0"] = dipole_moment(
            dipoles[((splita, splitb),"3+0")]= dipole_moment(
                psi3_part, D_part, psi0_part)

            psi0.restoreSubVector(is2, psi0_part)
            psi1.restoreSubVector(is1, psi1_part)
            psi2.restoreSubVector(is2, psi2_part)
            psi3.restoreSubVector(is1, psi3_part)

        # dipole_df = pd.Series(dipoles.values(), index=dipoles.keys())
        # dipole_df.index.set_names(["split"])

        # dipole_df
        # logging.info(dipole_df)

    dipole_df = pd.DataFrame(dipoles, index=time)
    logging.info(dipole_df)

    if rank == 0:
        dipole_df.to_hdf(store, out_store_key)
    #store.append(out_store_key, dipole_df)
    store.close()


@click.command(context_settings=dict(ignore_unknown_options=True,))
@click.option("--hamiltonian_folder", type=str)
@click.option("--out_file", type=str, default=None)
@click.option("--out_key", type=str, default=None)
@click.option("--in_file", type=str, default=None)
@click.option("--in_key", type=str, default=None)
@click.option("--splits_folder", type=str, default=None)
@click.argument("other_args", nargs=-1, type=click.UNPROCESSED)
def setup_and_run(hamiltonian_folder, out_file, out_key, in_file, in_key, splits_folder, other_args):
    logging.basicConfig(
            filename = "log_dipole_" + str(rank) + in_key + ".log",
        format="[{}]:".format(rank) + '%(levelname)s:%(message)s',
        level=logging.INFO)
    D = initialize_objects(hamiltonian_folder)
    dataframe = hdf_store_to_dataframe(in_file, in_key)
    splits = [(-1000,0),(0,1000)]
    if splits_folder is not None:
        decomps = da.Abinitio(splits_folder).dipole.decompositions
        #logging.info(decomps)
        #logging.info([ x for x in decomps.values()])
        splits = [ x for x in decomps.values()]
    find_dipole_moments(D, splits, dataframe, out_file, out_key)


if __name__=="__main__":
    setup_and_run()
