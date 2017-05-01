"""
stft_analysis.py
An object for analyzing the time dependent susceptibility of a run, and some
functions for generating a series of runs.
"""

from __future__ import print_function, division

from os import walk, fstat
from os.path import join
import nonperturbative
import numpy as np
from units import atomic
import pandas as pd


def get_valid_subfolders(folder: str):
    """
    A function that returns subfolders in the folder that contain valid
    calculations
    """
    folders = []
    for dirp, dirs, files in walk(folder):
        # check to see if "wf_final.dat" is in folder,
        # along with dipole.dat.
        if "wf_final.dat" in files \
           and fstat("wf_final.dat").st_size != 0 \
           and "dipole.dat" in files \
           and fstat("dipole.dat").st_size != 0:
            folders += [dirp]
    return folders


class ArraysNotEqual(Exception):
    def __init__(self, message, lengths):
        self.message = message
        self.lengths = lengths

    def __repr__(self):
        return "< ArraysNotEqual(Exception) message: {}, length 1: {}, length 2: {} >".format(
            self.message, self.lengths[0], self.lengths[1])

    def __str__(self):
        return self.__repr__()


class TDSusceptibility:
    """
    A time dependent susceptibility class

    takes a folder and finds all the dipole splits, where they are
    (energy or principle quantum number), and then lazily finds them.
    """

    def __init__(self, folder: str, **kwargs):
        self.folder = folder
        # find all the dipole files in the folder:
        dirp, _, files = walk(folder)
        self.terms = []
        for f in files:
            f = f.lower()
            if f.startswith("dipole_"):
                self.terms += [f.split('_')[1].split('.')[0]]

        # look at the dipole file:
        self.decomps = []
        with open(join(folder, "Dipole.config")) as dpconf:
            for line in dpconf.readlines():
                if line.startswidth('-dipole_decomposition'):
                    self.decomps = line.split(" ")[1].split(',')

        if len(self.decomps) != self.terms:
            raise ArraysNotEqual(
                "Dipole.config and the file structure don't agree",
                len(self.decomps), len(self.terms))

        self.nonperturb = nonperturbative.Nonperturbative(self.folder,
                                                          **kwargs)

    @staticmethod
    def terms_to_files(term):
        return "dipole_" + term + ".dat"

    def calculate(self, **kwargs):

        dfs = []
        names = []
        efield = None
        for name, t in map(lambda x: (x, self.terms_to_file(x)), self.terms):
            this_frame = self.get_stft_data(
                self.nonperturb, t=t, name=name, **kwargs)
            this_frame.columns = this_frame.columns.get_level_values('value')
            if not efield:
                efield = this_frame['efield']
            self.dfs.append(this_frame[['susceptibility', 'dipole']])
            self.names.append(name)
        self.data = pd.DataFrame(efield)
        for df, name in zip(dfs, names):
            self.data[name] = df['susceptibility']
            self.data[name + " dipole"] = df['dipole']

        return self.data

    def save(self, filename=None):
        if filename is None:
            filename = join(self.folder, "stft_data.csv")
        self.data.to_csv(filename)
