from os.path import exists, join, dirname
from collections import namedtuple
from units import atomic
import numpy as np
import pandas as pd
from common import file_size, get_file, import_petsc_vec
from fourier import Fourier


class Laser(object):

    Shape = namedtuple("Shape", ['front', 'back'])

    def __init__(self, filename, time):
        self.time = time
        self.cycles = int()
        self.cep = float()
        self.dt = float()
        self.wavelength = float()
        self.shape = "sin_squared"
        self.height = float(0)
        self.intensity = float()
        sp = []

        with open(filename, 'r') as laser_conf:
            for line in laser_conf:
                if line.startswith("-laser_intensity"):
                    self.intensity = float(line.split(" ")[1])
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
                if line.startswith("-laser_shape"):
                    self.shape = str(line.split(" ")[1])
                if line.startswith("-laser_front_shape"):
                    sp[0] = int(line.split(" ")[1])
                if line.startswith("-laser_back_shape"):
                    sp[1] = int(line.split(" ")[1])
                if line.startswith("-laser_cep"):
                    self.cep = float(line.split(" ")[1])
                if line.startswith("-laser_filename"):
                    self.ef_filename = join(
                        dirname(filename), line.split(" ")[1])

        self.shape_parameters = Laser.Shape(sp[0], sp[1])
        self.freq = atomic.from_wavelength(self.wavelength)
        self.zeros_ = []
        self.center_of_pulse_ = float()
        self.ef_ = None

    @property
    def zeros(self):
        if self.zeros_ is None:
            self.efield()
        return self.zeros_

    @property
    def center_of_pulse(self):
        if self.center_of_pulse_ is None:
            self.efield()
        return self.center_of_pulse_

    @property
    def efield(self):
        if self.ef_ is None:
            try:
                get_file(self.ef_filename)
            except IOError:
                if self.shape == "gaussian":
                    fwhm_time = np.pi * 2 * self.cycles / self.freq
                    mean = fwhm_time * np.sqrt(np.log(1. / self.height))
                    mean /= (2. * np.sqrt(np.log(2.)))
                    std_dev = fwhm_time / np.sqrt(8. * np.log(2.))
                    ef = np.exp(-(self.time - mean)**2 / (2. * std_dev**2))
                    ef *= np.sin(self.freq * self.time)
                    ef = ef * np.sqrt(self.intensity / atomic.intensity)
                elif self.shape == "sin_squared":
                    ef = np.sin(self.freq * self.time / (self.cycles * 2))**2
                    ef *= np.sin(self.freq * self.time)
                    ef = ef * np.sqrt(self.intensity / atomic.intensity)
            if np.shape(self.time) != np.shape(ef):
                ef = np.append(
                    ef, np.zeros(
                        len(self.time) - len(ef), dtype='d'))

            # find the peak and zeros of the efield:
            signs = np.sign(ef)
            signs[signs == 0.0] = -1
            self.zeros_ = np.where(np.diff(signs))[0]
            if self.shape == "gaussian":
                self.center_of_pulse_ = self.time[int(len(self.time) / 2)]
            elif self.shape == "sin_squared":
                self.center_of_pulse_ = np.pi * self.cycles / self.freq
            self.ef_ = ef
        return Fourier(self.time, self.ef_)


class Dipole(object):
    def __init__(self, filename, time):
        from string import ascii_lowercase
        from itertools import combinations_with_replacement

        self.time = time
        letters = iter(ascii_lowercase)
        self.decompositions = {next(letters): 0.0}
        self.folder = dirname(filename)

        self.pondermotive = False

        with open(filename, 'r') as dipole_conf:
            for line in dipole_conf:
                if line.startswith("-dipole_decomposition"):
                    self.decompositions.update(
                        dict(
                            map(lambda x: (next(letters), float(x)),
                                line.split(" ")[1].strip(',\n').split(","))))
                if line.startswith("-dipole_ponder"):
                    self.pondermotive = True

        self.decomp_units = None
        # check if float or integer decomposition boundaries make more sense
        for val in self.decompositions.values():
            if float(int(round(val))) != val:
                # there is at least one float, so this is in
                # terms of energy, not index:
                self.decomp_units = "energy"
        if self.decomp_units is None:
            self.decomp_units = "index"

        self.names = combinations_with_replacement(self.decompositions.keys(),
                                                   2).sort()

        for x in self.names:
            term_filename = self.term_to_filename(x)
            if not exists(term_filename):
                raise FileNotFoundError("%s not found!" % term_filename)

    def term_to_filename(self, term):
        if term.lower() == "all":
            return join(self.folder, "dipole.dat")

        return join(self.folder, "dipole_" + term + ".dat")

    def __getitem__(self, key):
        if type(key) is slice:
            if key.stop > len(self.names):
                raise IndexError("%s is a larger dipole than we have")
            elements = range(key.start, key.stop, key.step)
        else:
            if key not in self.names:
                raise IndexError("%s not a valid dipole term" % key)
            else:
                return self.dipole(self, key)

        dipoles = []
        for key in elements:
            dipoles.append(self.dipole(self, self.names[key]))

        return pd.DataFrame(
            map(lambda x: x.tdata, dipoles),
            index=self.time,
            columns=map(lambda x: self.names(x), elements))

    def dipole(self, term):
        fname = self.term_to_filename(term)

        size = int(file_size(fname) / 8)

        if size == len(self.time):
            filetype = 'd'
        elif size == len(self.time) * 2:
            filetype = 'D'
        return Fourier(self.time, get_file(fname, filetype))


class Wavefunctions(object):
    def __init__(self, path):
        from itertools import count
        self.path = path
        self.zeros = int()
        for z in count():
            self.zeros = z
            if not exists(join(path, "wf_" + str(z) + ".dat")):
                break

    def get_index(self):
        """get the prototype from the csv file in the run's
        directory.  returns a pandas MultiIndex"""
        n = []
        l = []
        j = []
        m = []
        e = []
        with open(join(self.path, "prototype.csv"), 'r') as prototype_f:
            for line in prototype_f:
                i = line.split(',')
                n.append(int(i[0]))
                l.append(int(i[1]))
                j.append(float(int(i[2])) / 2.)
                m.append(int(i[3]))
                e.append(float(i[4]))
        return pd.MultiIndex.from_arrays(
            [n, l, j, m, e], names=["n", "l", "j", "m", "e"])

    def __len__(self):
        return self.zeros

    def __getitem__(self, zslice):
        elements = []
        if type(zslice) is slice:

            if zslice.stop > len(self):
                raise IndexError("requesting too many zeros")
            elements = range(zslice.start, zslice.stop, zslice.step)
        else:
            elements = [zslice]

        vecs = []
        for zero in elements:
            if zero >= self.zeros or zero < 0:
                raise IndexError("Zero %s not a zero!" % zero)
            fname = join(self.path, "wf_" + str(zero) + ".dat")
            if zero == self.zeros:
                fname = join(self.path, "wf_final.dat")
            if not exists(fname):
                raise FileNotFoundError("Didn't find file: %s" % fname)
            vecs.append(import_petsc_vec(fname))

        return pd.DataFrame(vecs, columns=elements, index=self.get_index())


class Abinitio(object):
    """
    Class that represents a run. Everything that calculates data for a run
    should be done through this class.
    """

    def __init__(self, path: str):
        self.path = path

        self.time = get_file(join(path, "time.dat"))
        self.laser = Laser(join(path, "Laser.config"), self.time)
        self.dipole = Dipole(join(path, "Dipole.config"), self.time)
        self.wavefunctions = Wavefunctions(path)

    def time_dependence_data(self):
        efield = self.laser.efield
        dipoles = self.dipole[:]

        dipoles["efield"] = efield.tdata
        return dipoles

    def stft_data(self, *args, **kwargs):
        dipoles = [self.dipole.dipole(x) for x in self.dipole.names]
        stfts = [dipole.stft(*args, **kwargs) for dipole in dipoles]
        time, efield = self.laser.efield.stft(*args, **kwargs)

        data = pd.DataFrame(stfts, columns=self.dipole.names, index=time)
        data['efield'] = efield
