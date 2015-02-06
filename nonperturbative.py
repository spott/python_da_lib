"""
nonperturbative.py:
    Mostly just contains the nonperturbative_set and nonperturbative classes.

"""

from __future__ import print_function, division
import numpy as np
import os
import pandas as pd
import scipy.signal
from common import import_petsc_vec
from units import atomic
import sys
from window import *


import fourier


class NonperturbativeSet(object):

    '''A set of nonperturbative (full TDSE) runs,
    specifically their chi values.  '''

    def __init__(self, folder, n=1, wanted="susceptibility", window=scipy.signal.boxcar, dipoles=None):
        self.data = None
        self.folders = []
        self.n = n
        self.window = window
        self.dipoles = dipoles

        if wanted not in ["susceptibility","peak_dipole","efield","integrated_harmonic_power","peak_harmonic_power"]:
            raise Exception("wanted value <" + str(wanted) + "> not known")

        self.wanted = wanted

        os.path.walk(folder, NonperturbativeSet.__visit, self)
        self.data = self.data.groupby(self.data.index).sum()

        self.data.sort_index(inplace=True)

    def __visit(self, dirname, names):
        """ a helper function for the directory walk """
        if not "wf_final.dat" in names:
            return
        # try:
        if self.data is None:
            self.data = pd.DataFrame(
                NonperturbativeSet.Nonperturbative(dirname, self.n, self.wanted, self.window, self.dipoles).data)
            self.folders.append([dirname])
        else:
            new_data = NonperturbativeSet.Nonperturbative(dirname, self.n, self.wanted, self.window, self.dipoles).data
            self.data = pd.concat(
                [self.data, new_data])
            self.folders.append([dirname])
        # except Exception as inst:
            #print("Failed",dirname, inst)
            # return

    def nl_chis(self):
        """finds the nonlinear part of the susceptibility for each
        individual column by subtracting out the y-intercept of a
        linear fit of the first two points in intensity.

        Returns:
            a new dataframe with just the nonlinear part of the susceptibility
        """
        nl_susc = self.data.copy()
        for column in nl_susc.columns:
            slope = (nl_susc[column].iloc[1] - nl_susc[column].iloc[0]) / \
                (nl_susc[column].index[1] - nl_susc[column].index[0])
            _y0 = - \
                (nl_susc[column].index[0] * slope - nl_susc[column].iloc[0])
            nl_susc[column] = nl_susc[column] - _y0
        return nl_susc

    def ionizations(self, zero=-1):
        """ Find the ionization for the set of runs.
        Calls Nonperturbative.ionization(n) for each run
        in the set, and puts them all in a DataFrame.

        Args:
            n: the zero of the field to find the ionization in.

        Returns:
            a pandas single column DataFrame with a index
            parameterized on intensity, wavelength and cycles.
        """
        ionization = {}
        for i in self.folders:
            run = self.Nonperturbative(i[0])
            ionization[(run.intensity, run.wavelength, run.cycles)] = float(
                run.ionization(zero))
        index = pd.MultiIndex.from_tuples(
            ionization.keys(), names=["intensity", "wavelength", "cycles"])
        return pd.DataFrame(ionization.values(), index=index).sort_index()

    class Nonperturbative(object):

        """ nonperturbative object representing a single nonperturbative run
        allows access to properties of the run, and to calculations done on
        the run
        """

        def __init__(self, folder, n=1, wanted="susceptibility" ,window=scipy.signal.boxcar, dipoles=None):
            self.window=window
            self.wanted=wanted
            if not os.path.exists(os.path.join(folder, "wf_final.dat")):
                #raise Exception("folder doesn't have wf_final.dat", folder)
                print("folder doesn't have wf_final.dat, attempting anyways")
            self.folder = folder
            with open(os.path.join(folder, "que"), 'r') as que_file:
                self.intensity = float()
                line_of_interest = str()
                for line in que_file:
                    if line.startswith("mpirun"):
                        line_of_interest = line
                line_of_interest = line_of_interest.replace('\t', '   ')
                for element in line_of_interest.split('    '):
                    if element.startswith("-laser_intensity"):
                        self.intensity = float(element.split(" ")[1])

            with open(os.path.join(folder, "Laser.config"), 'r') as laser_conf:
                self.cycles = int()
                self.dt = float()
                self.wavelength = float()
                self.shape = "sin_squared"
                self.height = float(0)
                for line in laser_conf:
                    if line.startswith("-laser_cycles"):
                        self.cycles = int(line.split(" ")[1])
                    if line.startswith("-laser_lambda"):
                        self.wavelength = float(line.split(" ")[1])
                    if line.startswith("-laser_dt "):
                        self.dt = float(line.split(" ")[1])
                    if line.startswith("-laser_envelope"):
                        self.shape = line.split(" ")[1].strip()
                    if line.startswith("-laser_height"):
                        self.height = float(line.split(" ")[1])
                    #if line.startswith("-laser_dt "):
                        #self.dt = float(line.split(" ")[1])
            #self.data_ = None
            #self.chi_ = None

        @property
        def data(self):
            if self.data_ is None:
                mindex = pd.MultiIndex.from_arrays(
                    [[self.cycles], [self.wavelength]], 
                    names=["cycles", "wavelength"])
                self.data_ = pd.DataFrame(
                    self.chi, columns=mindex, index=[self.intensity])
                self.data_.index.name = "intensity"
            return self.data_

        @property
        def chi(self):
            if self.chi_ is None:
                if self.wanted == "susceptibility":
                    self.chi_ = self.harmonic(n, self.window, dipoles=dipoles)
                elif self.wanted == "peak_dipole":
                    self.chi_ = self.dipole(self.window, t=dipoles)(n * atomic.from_wavelength(self.wavelength))
                elif self.wanted == "peak_harmonic_power":
                    self.chi_ = np.square(np.abs(self.dipole(self.window,t=dipoles)(n * atomic.from_wavelength(self.wavelength))))
                elif self.wanted == "efield":
                    self.chi_ = self.efield(self.window)(n * atomic.from_wavelength(self.wavelength))
                elif self.wanted == "integrated_harmonic_power":
                    self.chi_ = self.dipole(self.window, t=dipoles).integrated_freq(
                            lambda x: np.abs(x)**2, 
                            ((n-1) * atomic.from_wavelength(self.wavelength)),
                            ((n+1) * atomic.from_wavelength(self.wavelength)))

                else:
                    raise Exception("wanted value <" + str(wanted) + "> not known")
            return self.chi_


        # def dipole_t(self):
            # with open(os.path.join(self.folder, "dipole.dat"), 'rb') as f:
            #dp = np.fromfile(f, 'd', -1)
            # with open(os.path.join(self.folder, "time.dat"), 'rb') as f:
            #time = np.fromfile(f, 'd', -1)

            # return (time, dp)

        def dipole(self, window=None, t=None):
            """
            returns a Fourier object of the dipole moment selected of the run.
            t = None : the regular, total, dipole moment.
            t = "ab","ba","bb", etc.  the section of the total dipole moment.  if t[0] > t[1], flip and take the complex conjugate.
            t = [(x, "ab"), (y, "bc")], etc.  x(dipole moment of "ab") + y(dipole moment of "bc"), etc.
            """

            def ident(x):
                return x

            try:
                with open(os.path.join(self.folder, "time.dat"), 'rb') as time_f:
                    time_f.seek(0, os.SEEK_END)
                    timesize = time_f.tell() / 8
                    time_f.seek(0)
                    time = np.fromfile(time_f, 'd', -1)
            except IOError:
                with open(os.path.join(self.folder, "dipole.dat"), 'rb') as dipolef:
                    dipolef.seek(0, os.SEEK_END)
                    dipolesize = dipolef.tell() / 16
                    timesize = dipolesize
                time = np.linspace(0, dipolesize * self.dt, dipolesize, endpoint=False, dtype='d')

            files = []
            if t is None:
                files = [(ident, "dipole.dat")]
            if t is not None:
                if isinstance(t, str):
                    files = [(ident, "dipole_" + t + ".dat")]
                else:
                    for fun, f in t:
                        if (f is None):
                            files.append( (fun, "dipole.dat") )
                        else:
                            if (f[0] > f[1]):
                                files.append( (lambda x: fun(np.conj(x)), "dipole_" + f[1] + f[0] + ".dat") )
                            else:
                                files.append( (fun, "dipole_" + f + ".dat"))
            print(files)
            print(self.folder)
            dp = np.zeros(timesize, dtype='d')
            for func, f in files:
                with open(os.path.join(self.folder, f), 'rb') as dipolef:
                    dipolef.seek(0, os.SEEK_END)
                    dipolesize = dipolef.tell() / 8
                    dipolef.seek(0)
                    if dipolesize == 2 * timesize:
                        dp = np.subtract(dp,func(np.fromfile(dipolef, 'D', -1)))
                    elif dipolesize == timesize:
                        dp = np.subtract(dp,func(np.fromfile(dipolef, 'd', -1)))
                    else:
                        raise Exception("dipole: dipole file does not have a filesize equal to, or double that of the time file")
            #dp *= -1
            print( dp)
            if window is not None:
                return fourier.Fourier(time, dp, window)
            else:
                return fourier.Fourier(time, dp, self.window)

        def population(self):
            """
            returns a dataframe containing a list of the population in every "section" of the wavefunction that there is a dipole moment for.
            """

            splits = set(''.join(map(lambda x: x[7:9], filter( lambda x: x.startswith("dipole_"), os.listdir(self.folder)))))

            with open(os.path.join(self.folder, "population.dat"), 'rb') as pop:
                population = np.fromfile(pop,'d', -1)
                population = population.reshape((-1,len(splits)))

            with open(os.path.join(self.folder,"time.dat"), 'rb') as f:
                time = np.fromfile(f,'d',-1)

            return pd.DataFrame(population, index=time, columns=sorted(list(splits)))

        def efield(self, window=None):
            """
            returns a Fourier object of the efield of the run.
            """
            try:
                with open(os.path.join(self.folder, "time.dat"), 'rb') as time_f:
                    time = np.fromfile(time_f, 'd', -1)
            except IOError:
                with open(os.path.join(self.folder, "dipole.dat"), 'rb') as dipolef:
                    dipolef.seek(0, os.SEEK_END)
                    dipolesize = dipolef.tell() / 16
                time = np.linspace(0, dipolesize * self.dt, dipolesize, endpoint=False, dtype='d')
                #ef = np.append(ef, np.zeros(dipolesize - len(ef), dtype='d'))
                
            try:
                with open(os.path.join(self.folder, "efield.dat"), 'rb') as ef_file:
                    ef = np.fromfile(ef_file, 'd', -1)
            except IOError:
                freq = atomic.from_wavelength(self.wavelength)
                if self.shape == "gaussian":
                    fwhm_time = np.pi * 2 * self.cycles / freq
                    mean = fwhm_time * np.sqrt(np.log(1. / self.height))
                    mean /= (2. * np.sqrt(np.log(2.)))
                    std_dev = fwhm_time / np.sqrt(8. * np.log(2.))
                    ef = np.exp(- (time - mean)**2 / (2. * std_dev**2))
                    ef *= np.sin(freq * time)
                    ef = ef * np.sqrt(self.intensity / atomic.intensity)
                elif self.shape == "sin_squared":
                    ef = np.sin(freq * time / (self.cycles * 2)) ** 2
                    ef *= np.sin(freq * time)
                    ef = ef * np.sqrt(self.intensity / atomic.intensity)
            if np.shape(time) != np.shape(ef):
                ef = np.append(ef, np.zeros(len(time) - len(ef),dtype='d'))
            print("efield shape:", np.shape(ef))
            print("time shape:", np.shape(time))
            if window:
                return fourier.Fourier(time, ef, window)
            else:
                return fourier.Fourier(time, ef, self.window)

        # def efield_t(self):
            # with open(os.path.join(self.folder, "efield.dat"), 'rb') as f:
            #ef = np.fromfile(f, 'd', -1)
            # with open(os.path.join(self.folder, "time.dat"), 'rb') as f:
            #time = np.fromfile(f, 'd', -1)
            # return (time,ef)

        def harmonic(self, order=1, window=scipy.signal.boxcar, dipoles=None):
            """
            return the susceptibility for a specific harmonic `order`.
            """
            dipole = self.dipole(window, t=dipoles)
            dipole = dipole(order * atomic.from_wavelength(self.wavelength))
            efield = self.efield(window)
            # We want the susceptibility at order omega from omega
            efield = efield(atomic.from_wavelength(self.wavelength))
            return dipole / efield

        def get_prototype(self):
            """get the prototype from the csv file in the run's
            directory.  returns a pandas MultiIndex"""
            n = []
            l = []
            j = []
            m = []
            e = []
            with open(os.path.join(self.folder, "prototype.csv"), 'r') as prototype_f:
                for line in prototype_f:
                    i = line.split(',')
                    n.append(int(i[0]))
                    l.append(int(i[1]))
                    j.append(float(int(i[2])) / 2.)
                    m.append(int(i[3]))
                    e.append(float(i[4]))
            return pd.MultiIndex.from_arrays([n, l, j, m, e], names=["n", "l", "j", "m", "e"])

        def wf(self, zero=-1):
            """ The wavefunction at the zero of the field number `zero`.
            """
            total_zeros = len(list(filter(lambda x: x.find("wf_") != -1, os.listdir(self.folder))))
            if zero == -1:
                wavefn = import_petsc_vec(
                    os.path.join(self.folder, "wf_final.dat"))
                cols = pd.MultiIndex.from_tuples([(self.intensity, self.cycles*2)], names=["intensity", "zero"])
                return pd.DataFrame(wavefn, index=self.get_prototype(), columns=cols)
            elif zero < total_zeros - 1:
                wavefn = import_petsc_vec(
                    os.path.join(self.folder, "wf_" + str(zero) + ".dat"))
                cols = pd.MultiIndex.from_tuples([(self.intensity, zero)], names=["intensity", "zero"])
                return pd.DataFrame(wavefn, index=self.get_prototype(), columns=cols)
            else:
                raise Exception(zero, "n not less than ", total_zeros)

        def gs_population(self, zero=-1):
            """get the ground state population at the zero of the
            field represented by `zero`."""
            wavefn = self.wf(zero)
            return wavefn.apply(lambda x: abs(x) ** 2).query('n == 1').iloc[0]

        def bound_population(self, zero=-1):
            """get the bound state population (including the gs) at
            the zero of the field represented by `zero`."""
            wavefn = self.wf(zero)
            return wavefn.apply(lambda x: abs(x) ** 2).query('e < 0').sum()

        def ionization(self, zero=-1):
            """get the ionized population at the zero of the
            field represented by `zero`."""
            wavefn = self.wf(zero)
            absorbed = 1 - wavefn.apply(lambda x: abs(x) ** 2).sum()
            return absorbed + wavefn.apply(lambda x: abs(x) ** 2).query('e > 0').sum()

    @staticmethod
    def get_prototype(filename="prototype.csv"):
        """get the prototype from a csv file
        returns a pandas MultiIndex"""
        with open(filename, 'r') as prototype_f:
            n = []
            l = []
            j = []
            m = []
            e = []
            for line in prototype_f:
                i = line.split(',')
                n.append(int(i[0]))
                l.append(int(i[1]))
                j.append(float(int(i[2])) / 2.)
                m.append(int(i[3]))
                e.append(float(i[4]))
            return pd.MultiIndex.from_arrays([n, l, j, m, e], names=["n,l,j,m,e"])



def sum_dipoles(df, dipoles):
    name = '+'.join(dipoles)
    simple = []
    conj = []
    for dipole in dipoles:
        if dipole in df.columns:
            simple.append(dipole)
        elif (dipole[1] + dipole[0]) in df.columns:
            conj.append((dipole[1] + dipole[0]))
        else:
            raise(Exception("dipole element doesn't exist" + (dipole[1] + dipole[0])))
    data = df[simple].join(df[conj].apply(np.conjugate), rsuffix='conj')
    return pd.DataFrame(data.sum(axis=1), columns=[name])


def get_stft_data(folder, t=None, name="All",
                  window_fn=flattop,
                  cycles=5, freq=None, ra=[0, -1], filt=None, dt=5):
    if freq is None:
        freq = cycles
    if t is not None:
        print("calculating for " + folder + " " + str(t) + ", " + window_fn.name + " cycles: " + str(cycles) + "...")
    elif t is None:
        print("calculating for " + folder + " All, " + window_fn.name + " cycles: " + str(cycles) + "...")

    print("importing folder...", end="")
    sys.stdout.flush()
    run = NonperturbativeSet.Nonperturbative(folder)
    print("done.")

    # 1 period
    period = 2. * np.pi / atomic.from_wavelength(run.wavelength)

    print("finding dipole moment...", end="")
    sys.stdout.flush()
    dipole = run.dipole(t=t)
    efield = run.efield()
    print("done.")

    if (ra[1] == -1):
        ra[1] = dipole.time[-1]

    window = np.searchsorted(dipole.time, [period * cycles], "left")[0]
    jump = int(dt / (dipole.time[1] - dipole.time[0]))  # int(window_fn.rov * window)

    ra[0] = ra[0] - window/2.
    ra[1] = ra[1] + window/2.

    ra = np.searchsorted(dipole.time, ra)
    print("window: ",window," jump: ", jump)
    print("find stft for dipole...", end="")
    sys.stdout.flush()
    time, td_dipole = dipole.stft(window, jump,
                                  window_fn=window_fn,
                                  ra=ra, filt=filt)
    print("done")
    print("find stft for efield...", end="")
    sys.stdout.flush()
    _, td_ef = efield.stft(window, jump,
                           window_fn=window_fn,
                           ra=ra, filt=filt)
    print("done")

    #from scipy.optimize import curve_fit

    #def sinsquared(x, a):
        #return np.select([x <= (2*30890 * .05), x > (2*30890 * .05)],
                         #[a * np.sin(x * np.pi / (2*30890 * .05))**2, 0])

    #popt, pconv = curve_fit(sinsquared,
                            #time,
                            #np.abs(td_ef.T[cycles]), [1])
    run_df = pd.DataFrame.from_dict({
                "dipole": td_dipole.T[freq],
                "efield": td_ef.T[freq],
                "susceptibility": td_dipole.T[freq] / td_ef.T[freq]})
                #"susceptibility_fit": td_dipole.T[freq] / np.array(
                    #sinsquared(time, popt[0]) * np.exp(1j * atomic.from_wavelength(run.wavelength) * time - 1j * np.pi / 2))})
    run_df.index = pd.Index(time, name="time")
    window_fn = window_fn.name
    filt = "None" if filt is None else filt
    run_df.columns = pd.MultiIndex.from_tuples([(run.intensity, window_fn, filt, freq, cycles, name, "dipole"),
                                                (run.intensity, window_fn, filt, freq, cycles, name, "efield"),
                                                (run.intensity, window_fn, filt, freq, cycles, name, "susceptibility")],
                                                #(run.intensity, window_fn, filt, freq, cycles, name, "susceptibility_fit")], 
                                                names=["intensity", "windowing function", "filter", "freq", "cycle", "decomp" ,"value"])
    print("done")
    return run_df

