"""
contains the class "Basis", representing a basis set.
"""
from __future__ import print_function, division
import numpy as np
import os
import pandas as pd
from common import get_file



class Basis(object):

    """
    Basis class represents a basis.

    Attributes:
        folder: the folder of the basis
        grid: the grid that the basis functions are run on
        points: number of points that the basis functions have

    member functions:
        wf: a wavefunction
        get_prototype: creates a multiindex.  uses the binary data format.
        prototype: helper function for get_prototype
    """

    def __init__(self, folder):
        self.folder = folder
        self.grid = get_file(os.path.join(folder, "grid.dat"))
        self.points = len(self.grid)

    def wf(self, n, l):
        """ takes the principle quantum number and the angular quantum number
        and returns the wavefunction.
        """
        filename = os.path.join(self.folder, "l_" + str(l) + ".dat")
        with open(filename, 'rb') as wf_file:
            npy = np.fromfile(wf_file, 'd', (n - (l)) * self.points)[-self.points:]
        return npy

    def prototype(self):
        """ internal ish function that grabs the binary file prototype.dat, and
        puts it into the right data_structure"""
        filename = os.path.join(self.folder, "prototype.dat")
        dt = np.dtype([('n', np.int32), ('l', np.int32), ('j', np.int32),
                       ('m', np.int32), ('e', np.complex128)])
        with open(filename, 'rb') as prototype_file:
            npy = np.fromfile(prototype_file, dt)
        return npy

    def get_prototype(self):
        """ returns a multiindex for the basis."""
        n = []
        l = []
        j = []
        m = []
        e = []
        prototype_f = self.prototype()
        for a in prototype_f:
            n.append(a['n'])
            l.append(a['l'])
            j.append(float(a['j'] / 2.))
            m.append(a['m'])
            e.append(a['e'])
        return pd.MultiIndex.from_arrays([n, l, j, m, e], names=["n", "l", "j", "m", "e"])
