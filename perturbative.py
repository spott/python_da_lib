"""
perturbative.py:
    Mostly just contains the nonperturbative_set and nonperturbative classes.

"""
from __future__ import print_function, division

import os as os
import pandas as pd

from common import get_file

from units import atomic

FIRST_HARMONIC_LABELS = ["1", "-1,1,1", "-1,-1,1,1,1", "-1,-1,-1,1,1,1,1",
                         "-1,-1,-1,-1,1,1,1,1,1", "-1,-1,-1,-1,-1,1,1,1,1,1,1"]
THIRD_HARMONIC_LABELS = ["1,1,1", "-1,1,1,1,1", "-1,-1,1,1,1,1,1",
                         "-1,-1,-1,1,1,1,1,1,1", "-1,-1,-1,-1,1,1,1,1,1,1,1"]
FIFTH_HARMONIC_LABELS = ["1,1,1,1,1", "-1,1,1,1,1,1,1",
                         "-1,-1,1,1,1,1,1,1,1", "-1,-1,-1,1,1,1,1,1,1,1,1"]


class PerturbativeSet(object):

    ''' holds the dataframe that has all the perturbative chi calculations. '''

    def __init__(self, folder=None, df=None):
        ''' parent folder is the input, containing at least a
        "rmax/npoints/nmax_nonlinear_DNWM" structure '''
        if folder is None and df is None:
            raise Exception("no arguments passed!")
        elif folder is not None:
            self.data = None

            self.rmax = []
            self.nmax = []
            self.npoints = []
            basedir = folder
            is_folder = lambda x: os.path.isdir(os.path.join(folder, x))
            for rmax in filter(is_folder, os.listdir(folder)):
                folder = os.path.join(basedir, rmax)
                for npoints in filter(is_folder, os.listdir(folder)):
                    folder = os.path.join(basedir, rmax, npoints)
                    for nmax_folder in filter(
                            lambda x: x.count("nonlinear") == 1
                            and is_folder(x),
                            os.listdir(folder)):
                        folder = os.path.join(
                            basedir, rmax, npoints, nmax_folder)
                        if "frequencies.dat" in os.listdir(folder):
                            try:
                                self.rmax += [int(rmax)]
                                self.nmax += [int(nmax_folder.split("_")[0])]
                                self.npoints += [int(npoints)]
                                # try:
                                d = PerturbativeSet.perturbative(folder,
                                                                 rmax,
                                                                 npoints,
                                                                 nmax_folder.split("_")[0])
                                if self.data is None:
                                    self.data = d
                                else:
                                    # try:
                                    self.data = self.data.join(d, how="outer")
                                # except Exception as inst:
                                    # print(d)
                                    # print("\n")
                                    # print(inst)
                                    # print(os.path.join(folder,rmax,npoints,nmax_folder))
                                    #print("==== join failed ====")
                            except Exception as inst:
                                print(folder)
                                print(type(inst))
                                print(inst)
                                print("==== find failed ====")
        else:
            self.data = df

    @staticmethod
    def perturbative(folder, rmax, npoints, nmax):
        ''' create the dataframe for a single perturbative chi calculation
        , taking the rmax, npoints, nmax as strings.
        populate all the information on it. '''
        # import the frequencies:
        freqs = [str(i)[:5]
                 for i in get_file(os.path.join(folder, "frequencies.dat"))]
        rmax = int(rmax)
        npoints = int(npoints)
        nmax = int(nmax)

        # import the imaginary component:
        imgs = []
        with open(os.path.join(folder, "que"), 'r') as que_file:
            terms = []
            for line in que_file:
                if line.strip().startswith("mpiexec"):
                    terms = line.split(" ")
                    break

            for i in range(len(terms)):
                if terms[i] == "-nonlinear_img":
                    imgs = [float(i) for i in terms[i + 1].split(',')]
                    break

        chis = {}
        for i in filter(lambda x: x.count("chi") == 1, os.listdir(folder)):
            splits = i.split("_")
            if len(splits) == 2:
                splits = splits[1].split(".")[0]
            else:
                splits = splits[1:-1]
            chis[",".join(splits)] = get_file(os.path.join(folder, i), 'D')
        if not imgs:
            raise Exception("imgs is empty", imgs, folder)
        if not freqs:
            raise Exception("freqs is empty", freqs, folder)
        data = pd.DataFrame(chis)
        # print data
        # print imgs
        # print freqs
        data.index = pd.MultiIndex.from_product(
            [imgs, freqs], names=["epsilon", "frequency"])
        data.columns = pd.MultiIndex.from_product(
            [[rmax], [npoints], [nmax], data.columns], names=["rmax", "npoints", "nmax", "chi"])
        return data
        # self.data.columns.set_levels =

    def chis(self, freq="0.056"):
        """return all the chis"""
        chis_df = self.data.xs(freq, level="frequency")
        chis_df = chis_df.xs(max(self.nmax), level="nmax", axis=1)
        chis_df = chis_df.xs(max(self.npoints), level="npoints", axis=1)
        return chis_df[max(self.rmax)].iloc[-1].T

    def dnwm(self, freq="0.056"):
        """return the degenerate n wave mixing terms of the chis"""
        dnwm_df = self.data.xs(freq, level="frequency")
        dnwm_df = dnwm_df.xs(max(self.nmax), level="nmax", axis=1)
        dnwm_df = dnwm_df.xs(max(self.npoints), level="npoints", axis=1)
        return dnwm_df[500][FIRST_HARMONIC_LABELS].iloc[-1]

    def nlchi_vs_intensity(self, intensities, freq="0.056"):
        """ takes a list of intensities, and a frequency (in string form),
        and returns a DataFrame with the nonlinear part (chi3 and above)
        of the perturbative susceptibility.
        """
        nlchis = {}
        dnwm_data = self.dnwm(freq)
        for i in range(1, len(FIRST_HARMONIC_LABELS)):
            # print(int(m/2))
            chi = lambda intensity: sum(
                [
                    dnwm_data[FIRST_HARMONIC_LABELS[j]]
                    * atomic.averaged_intensity(intensity, j)
                    for j in range(1, i + 1)
                ])
            key = "chi" + str(i * 2 + 1)
            nlchis[key] = [chi(intensity) for intensity in intensities]

        return pd.DataFrame(nlchis, index=intensities)

    def chi_vs_intensity(self, intensities, freq="0.056", harmonic=1, averaged=True):
        """ takes a list of intensities, a frequency (as a x.xxx string), the
        harmonic order that is desired, and if it will be averaged intensity or
        not
        returns a dataframe with index as the intensities, and rows as the chi
        for that intensity as sums over chi_n for n = harmonic to n = max"""
        chis = {}
        data = self.chis(freq)
        if harmonic == 1:
            labels = FIRST_HARMONIC_LABELS
        else:
            labels = THIRD_HARMONIC_LABELS
        for i in range(0, len(labels)):
            # print(int(m/2))
            if averaged:
                chi_fn = lambda intensity: sum(
                    [
                        data[labels[j]]
                        * atomic.averaged_intensity(intensity, j)
                        for j in range(0, i + 1)
                    ])
            else:
                chi_fn = lambda intensity: sum(
                    [
                        data[labels[j]]
                        * (intensity / atomic.intensity) ** j
                        for j in range(0, i + 1)
                    ])
            key = "chi" + str(i * 2 + 1)
            chis[key] = [chi_fn(intensity) for intensity in intensities]

        return pd.DataFrame(chis, index=intensities)

    #def fractional_chis(self, intensities, compared_to=3, freq="0.056", tot=False, averaged=True):
        #""" """
        #l = {}
        #dnwm_data = self.dnwm(freq)
        #if averaged:
            #intensity_fun = atomic.averaged_intensity
        #else:
            #intensity_fun = lambda inten, c: (inten / atomic.intensity)**c
        #comp_to = lambda i: i
        #if tot:
            #comp_to = lambda i: sum(
                    #[
                        #dnwm_data[int(c / 2)]
                        #* intensity_fun(i, int(c / 2))
                        #for c in range(1, compared_to + 2, 2)
                    #])
        #else:
            #comp_to = lambda i: \
                #dnwm_data[int(compared_to / 2)] \
                #* intensity_fun(i, int(compared_to / 2))

        #for m in range(int(compared_to / 2) + 1, len(FIRST_HARMONIC_LABELS)):
            ## print(int(m/2))
            #l["chi" + str(m * 2 + 1)] = [(dnwm_data[FIRST_HARMONIC_LABELS[m]]
                                          #* intensity_fun(i, m)) / comp_to(i) for i in intensities]

        #return pd.DataFrame(l, index=intensities)


    def fractional_chis(self, intensities, compared_to=3, freq="0.056", averaged=True):
        """ if compared_to is negative, then do \chi^{N} I / \chi^{N-1} with the 
        minimum \chi == chi^{-compared_to}.
        Otherwise, just do the normal compared to thing."""
        l = {}
        dnwm_data = self.dnwm(freq)
        if averaged:
            intensity_fun = atomic.averaged_intensity
        else:
            intensity_fun = lambda inten, c: (inten / atomic.intensity)**c
        comp_to = lambda i: i
        if compared_to < 0:
            for m in range(int(-compared_to / 2) + 1, len(FIRST_HARMONIC_LABELS)):
                l["chi" + str(m * 2 + 1)] = \
                    [
                        (dnwm_data[FIRST_HARMONIC_LABELS[m]]
                            * intensity_fun(i, m)) /
                        (dnwm_data[FIRST_HARMONIC_LABELS[m-1]]
                            * intensity_fun(i, m-1)) for i in intensities
                    ]
            return pd.DataFrame(l, index=intensities)
        else:
            comp_to = lambda i: \
                dnwm_data[int(compared_to / 2)] \
                * intensity_fun(i, int(compared_to / 2))

            for m in range(int(compared_to / 2) + 1, len(FIRST_HARMONIC_LABELS)):
                # print(int(m/2))
                l["chi" + str(m * 2 + 1)] = [(dnwm_data[FIRST_HARMONIC_LABELS[m]]
                                              * intensity_fun(i, m)) / comp_to(i) for i in intensities]

            return pd.DataFrame(l, index=intensities)

    def rmax_convergence(self, nmax=None, npoints=None, epsilon=None):
        """ for a specific nmax, npoints, and epsilon (where maximums
        of the first two, and minimums of the last are the defaults).
        returns a DataFrame for all chis as a function of rmax."""

        if nmax is None:
            nmax = max(self.nmax)
        if npoints is None:
            npoints = max(self.npoints)
        if epsilon is None:
            epsilon = min(self.data.index.get_level_values("epsilon"))
        rmax_df = self.data.xs(epsilon, level="epsilon")
        rmax_df = rmax_df.xs(nmax, level="nmax", axis=1)
        rmax_df = rmax_df.xs(npoints, level="npoints", axis=1)
        return rmax_df

    def nmax_convergence(self, rmax=None, npoints=None, epsilon=None):
        """ for a specific rmax, npoints, and epsilon (where maximums
        of the first two, and minimums of the last are the defaults).
        returns a DataFrame for all chis as a function of nmax"""
        if rmax is None:
            rmax = max(self.rmax)
        if npoints is None:
            npoints = max(self.npoints)
        if epsilon is None:
            epsilon = min(self.data.index.get_level_values("epsilon"))
        nmax_df = self.data.xs(epsilon, level="epsilon")
        nmax_df = nmax_df.xs(npoints, level="npoints", axis=1)
        return nmax_df[rmax]

    def npoints_convergence(self, rmax=None, nmax=None, epsilon=None):
        """ for a specific rmax, nmax, and epsilon (where maximums
        of the first two, and minimums of the last are the defaults).
        returns a DataFrame for all chis as a function of nmax"""
        if rmax is None:
            rmax = max(self.rmax)
        if nmax is None:
            nmax = max(self.nmax)
        if epsilon is None:
            epsilon = min(self.data.index.get_level_values("epsilon"))
        npoints_df = self.data.xs(epsilon, level="epsilon")
        return npoints_df.xs(nmax, level="nmax", axis=1)[rmax]
