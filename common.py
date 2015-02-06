"""
common.py

A bunch of common utilities.  currently a frange, and two files for
importing binary files or petsc files
"""

from __future__ import print_function,division
import os
import math
import numpy as np
import struct


def get_file(filename, dtype='d'):
    '''Imports a binary file.

    simple wrapper around numpys "fromfile" for importing binary files.

    Args:
        filename: the name of the file.
        dtype: the binary type of the elements.
    Returns:
        a numpy.array containing the data.
    '''
    with open(os.path.join(filename), 'rb') as bfile:
        npy = np.fromfile(bfile, dtype)
    return npy



def import_petsc_vec(filename):
    ''' import a PETSc vec. '''
    with open(filename, "rb") as pfile:
        byte = pfile.read(8)
        size = struct.unpack('>ii', byte)[1]
        npy = np.fromfile(pfile, '>D', size)
        return npy


def frange(limit1, limit2=None, increment=1.):
    """
    Range function that accepts floats (and integers).

    Usage:
    frange(-2, 2, 0.1)
    frange(10)
    frange(10, increment = 0.5)

    The returned value is an iterator.  Use list(frange) for a list.
    """

    if limit2 is None:
        limit2, limit1 = limit1, 0.
    else:
        limit1 = float(limit1)

    count = int(math.ceil(limit2 - limit1) / increment)
    return (limit1 + n * increment for n in range(count))
