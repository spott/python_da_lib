from os.path import exists, join, dirname
from collections import namedtuple
from units import atomic
import numpy as np
import pandas as pd
from common import file_size, get_file, import_petsc_vec
from fourier import Fourier
import functools


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
        sp = [2, 2]

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
                        dirname(filename), line.split(" ")[1].strip(" \n"))

        self.shape_parameters = Laser.Shape(sp[0], sp[1])
        self.freq = atomic.from_wavelength(self.wavelength)
        self.zeros_ = []
        self.center_of_pulse_ = float()
        self.ef_ = None

    def __repr__(self):
        return "<Laser: {λ: %f, cycles: %d, intensity: %e}>" % (
            self.wavelength, self.cycles, self.intensity)

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
                ef = get_file(self.ef_filename)
            except IOError as e:
                print("Failed to read efield, got error {e}, instead making it.".format(e=e))
                ef = self.find_efield()
            if np.shape(self.time) != np.shape(ef):
                # there is a problem here: just recreate the efield:
                print(
                    "warning, the efield array length and time array length don't match"
                )
                ef = self.find_efield()
                # ef = np.append(
                #     ef, np.zeros(
                #         len(self.time) - len(ef), dtype='d'))

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

    @staticmethod
    def gaussian_pulse(freq, intensity, cycles, height, cep):
        freq_ = freq
        cycles_ = cycles
        height_ = height
        cep_ = cep
        intensity_ = np.sqrt(intensity / atomic.intensity)
        fwhm_time = np.pi * 2 * cycles_ / freq_
        mean = fwhm_time * np.sqrt(np.log(1. / height_))
        mean /= (2. * np.sqrt(np.log(2.)))

        if not(cep_ > np.pi * 2 or cep_ < -np.pi * 2):
            cycles_till_mean = int(mean * freq_ / 2*np.pi)
            remainder = cep_ / (2. * np.pi)
            if mean < cycles_till_mean * 2 * np.pi / freq_:
                cycles_till_mean += 1
            mean = (cycles_till_mean + remainder) * (np.pi * 2 / freq_)

        std_dev = fwhm_time / np.sqrt(8. * np.log(2.))

        def pulse(t):
            ef = np.exp(-(t - mean)**2 / (2. * std_dev**2))
            ef *= np.sin(freq_ * (t - mean/2.) + cep_)
            ef *= intensity_
            return ef
        return pulse

    def find_efield(self):
        print("making efield.")
        if self.shape == "gaussian":
            return Laser.gaussian_pulse(self.freq, self.intensity, self.cycles, self.height, self.cep)(self.time)
        elif self.shape == "sin_squared":
            ef = np.sin(self.freq * self.time / (self.cycles * 2))**2
            ef *= np.sin(self.freq * self.time)
            ef = ef * np.sqrt(self.intensity / atomic.intensity)
        return ef

    @property
    def period(self):
        return 2. * np.pi / atomic.from_wavelength(self.wavelength)


class Dipole(object):
    def __init__(self, filename, time):
        from string import ascii_lowercase
        from itertools import combinations_with_replacement
        from collections import OrderedDict

        self.time = time
        letters = iter(ascii_lowercase)
        numbers = iter(map(str, range(15)))
        self.decompositions = None
        self.l_decompositions = None
        self.folder = dirname(filename)

        self.pondermotive = False

        with open(filename, 'r') as dipole_conf:
            for line in dipole_conf:
                if "dipole_decomposition_l" in line and len(
                        line.strip(" ,\n").split(" ")) == 2:
                    str_splits = ["-1000"] + line.strip(" ,\n").split(
                        " ")[1].strip(',\n').split(",") + ["1000"]
                    splits = list(map(lambda x: int(x), str_splits))
                    self.l_decompositions = OrderedDict(
                        map(lambda x: (next(numbers), x),
                            zip(splits[:-1], splits[1:])))
                elif "dipole_decomposition" in line and len(
                        line.strip(" ,\n").split(" ")) == 2:
                    str_splits = ["-1000"] + line.strip(" ,\n").split(
                        " ")[1].strip(',\n').split(",") + ["1000"]
                    splits = list(map(lambda x: float(x), str_splits))
                    self.decompositions = OrderedDict(
                        map(lambda x: (next(letters), x),
                            zip(splits[:-1], splits[1:])))
                elif line.startswith("-dipole_ponder"):
                    self.pondermotive = True

        self.decomp_units = None
        # check if float or integer decomposition boundaries make more sense
        if self.decompositions:
            for val in self.decompositions.values():
                if float(int(round(val[0]))) != val[0] or float(
                        int(round(val[1]))) != val[1]:
                    # there is at least one float, so this is in
                    # terms of energy, not index:
                    self.decomp_units = "energy"
            if self.decomp_units is None:
                self.decomp_units = "index"

        if self.l_decompositions:
            self.names = [
                str(x[0]) + str(x[1]) + "_" + str(y[0]) + str(y[1])
                for x in combinations_with_replacement(
                    self.decompositions.keys(), 2) for y in
                combinations_with_replacement(self.l_decompositions.keys(), 2)
            ] + ["all"]
        elif self.decompositions:
            self.names = [
                str(x[0]) + str(x[1])
                for x in combinations_with_replacement(
                    self.decompositions.keys(), 2)
            ] + ["all"]
        else:
            self.names = ["all"]

        for x in self.names:
            term_filename = self.term_to_filename(x)
            if not exists(term_filename):
                raise FileNotFoundError("%s not found!" % term_filename)

    def __repr__(self):
        return "<Dipole: {decompositions: %r, pondermotive: %r}>" % (
            self.decompositions, self.pondermotive)

    def term_to_filename(self, term):
        if term.lower() == "all":
            return join(self.folder, "dipole.dat")

        return join(self.folder, "dipole_" + term + ".dat")

    def __getitem__(self, key):
        if type(key) is slice:
            start, stop, step = (key.start, key.stop, key.step)
            if key.stop is None:
                stop = len(self.names)
            if key.start is None:
                start = 0
            if key.step is None:
                step = 1
            if stop > len(self.names):
                raise IndexError("%s is a larger dipole than we have")
            elements = range(start, stop, step)
        else:
            if key not in self.names:
                raise IndexError("%s not a valid dipole term" % key)
            else:
                return self.dipole(key)

        dipoles = []
        for key in elements:
            dipoles.append(self.dipole(self.names[key]))

        data = list(map(lambda x: x.tdata, dipoles))
        return pd.DataFrame(
            list(zip(*data)),
            index=self.time,
            columns=list(map(lambda x: self.names[x], elements)))

    def __hash__(self):
        return hash(self.folder)

    #@functools.lru_cache(maxsize=128, typed=False)
    def dipole(self, term):
        fname = self.term_to_filename(term)

        size = int(file_size(fname) / 8)

        filetype = None
        if size == len(self.time):
            filetype = 'd'
        elif size == len(self.time) * 2:
            filetype = 'D'
        else:
            raise Exception("dipole file corrupted: %s" % term)

        dipole_moment = get_file(fname, filetype)
        s1 = term[0]
        s2 = term[1]
        if self.l_decompositions:
            s1 = (term[0], term[-2])
            s2 = (term[1], term[-1])
        if s1 != s2 and term != "all":
            print("doubling term: {term}".format(term=term))
            # we want 2*real part when the two sections don't match:
            dipole_moment = 2 * dipole_moment.real
        else:
            dipole_moment = dipole_moment.real

        return Fourier(self.time, dipole_moment)


class Wavefunctions(object):
    def __init__(self, path):
        from itertools import count
        self.path = path
        self.zeros = int()
        for z in count():
            self.zeros = z
            if not exists(join(path, "wf_" + str(z) + ".dat")):
                break

    def __repr__(self):
        return "<Wavefunctions: {zeros: %d}>" % self.zeros

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
            stop, start, step = (zslice.stop, zslice.start, zslice.step)
            if zslice.stop is None:
                stop = len(self)
            if zslice.start is None:
                start = 0
            if zslice.step is None:
                step = 1
            if start < 0:
                start = len(self) + start
            if stop < 0:
                stop = len(self) + stop + 1
            if step < 0:
                assert (start > stop)

            if stop > len(self):
                raise IndexError("requesting too many zeros")
            elements = range(start, stop, step)
        elif zslice < 0:
            elements = [len(self) + zslice + 1]
        else:
            elements = [zslice]

        vecs = []
        for zero in elements:
            if zero > self.zeros or zero < 0:
                raise IndexError("Zero %s not a zero!" % zero)
            fname = join(self.path, "wf_" + str(zero) + ".dat")
            if zero == self.zeros:
                fname = join(self.path, "wf_final.dat")
            if not exists(fname):
                raise FileNotFoundError("Didn't find file: %s" % fname)
            vecs.append(import_petsc_vec(fname))

        return pd.DataFrame(
            list(zip(*vecs)), columns=elements, index=self.get_index())


class Abinitio(object):
    """
    Class that represents a run. Everything that calculates data for a run
    should be done through this class.
    """

    def __init__(self, path: str, ignore_saved_data=False):
        self.path = path

        self.time = get_file(join(path, "time.dat"))
        self.laser = Laser(join(path, "Laser.config"), self.time)
        self.dipole = Dipole(join(path, "Dipole.config"), self.time)
        self.wavefunctions = Wavefunctions(path)

        self.data = None
        self.stfts = None
        self.susceptibilities = None

        self.arguments = None

        if exists(join(path, "abinitio_data.hdf")) and not ignore_saved_data:
            # self.stfts = pd.read_hdf(join(path, "abinitio_data.hdf"), "stfts")
            try:
                self.data = pd.read_hdf(
                    join(path, "abinitio_data.hdf"), "data_field")
                self.susceptibilities = pd.read_hdf(
                    join(path, "abinitio_data.hdf"), "susceptibilities")
            except Exception as e:
                print("savefile for {self.path} ran into error {e}, ignoring".format(self=self, e=e))
                self.data = None
                self.susceptibilities = None

    def __repr__(self):
        return "<Abinitio: folder: %r \n\t{%r,\n\t %r,\n\t %r}>" % (
            self.path, self.laser, self.dipole, self.wavefunctions)

    def time_dependence_data(self):
        efield = self.laser.efield
        dipoles = self.dipole[:]

        dipoles["efield"] = efield.tdata
        return dipoles

    def stft_data(self,
                  freq=1,
                  cycles=6,
                  dt=10,
                  dt_cycle_frac=None,
                  even=None,
                  **kwargs):
        import scipy.signal
        # if None, use cycles to find window_size
        window_size = kwargs[
            "window_size"] if "window_size" in kwargs else None
        # if None, use dt to find hop
        hop = kwargs["hop"] if "hop" in kwargs else None
        kwargs[
            "window_fn"] = scipy.signal.flattop if "window_fn" not in kwargs else kwargs[
                "window_fn"]
        kwargs["ra"] = [0, -1] if "ra" not in kwargs else kwargs["ra"]
        kwargs["filt"] = None if "filt" not in kwargs else kwargs["filt"]
        kwargs["symetric"] = True if "symetric" not in kwargs else kwargs[
            "symetric"]
        kwargs["zero_pad"] = None if "zero_pad" not in kwargs else kwargs[
            "zero_pad"]
        # dt = kwargs["dt"] if "dt" in kwargs else 10  # time between each windowed fourier transform
        # freq = kwargs["freq"] if "freq" in kwargs else 1  # the frequency of interest, as a multiple of fundamental
        # cycles = kwargs["cycles"] if "cycles" in kwargs else 6  # the size of the window
        # # print("Start of stft_data for %r" % self)
        # even = kwargs["even"] if "even" in kwargs else None

        if self.data is not None and self.stfts is not None:
            return self.data
        # the period of the laser:
        period = self.laser.period

        # the window size we want to use.  Either a given, or calculated from
        # the cycles and the period.  When calculated, we find out how many
        # time points are in n cycles of the electric field.
        if window_size is None:
            window_size = np.searchsorted(self.time, [period * cycles],
                                          "left")[0]
            if even is True:
                window_size += window_size % 2
            if even is False:
                window_size += 1 - window_size % 2

        zero_pad = kwargs["zero_pad"]

        if zero_pad is not None:
            if kwargs["zero_pad"] < 20:
                kwargs["zero_pad"] *= int(window_size / cycles)

            zero_pad = int(max(window_size, kwargs["zero_pad"]))
            kwargs["zero_pad"] = zero_pad
        else:
            zero_pad = window_size

        bin_freq_f = (zero_pad * cycles / window_size)
        bin_freq = int(round(bin_freq_f))
        nn = zero_pad / window_size
        print(
            "Zero pad ({zero_pad}) is {nn} times window_size({window_size}),"
            " and a bin_freq is {bin_freq_f}".format(zero_pad=zero_pad,nn=nn, window_size=window_size, bin_freq_f = bin_freq_f))
        # the size of the move between windowing functions
        if hop is None and dt_cycle_frac is None:
            hop = int(dt / (self.time[1] - self.time[0]))
        elif hop is None:
            hop = int(round(dt_cycle_frac * window_size / cycles))
        print("hop is {hop} and dt_cycle_frac is {dt_cycle_frac}".format(hop=hop,dt_cycle_frac=dt_cycle_frac))

        dipoles = [self.dipole.dipole(x) for x in self.dipole.names]

        print(
            "stft options: window_size={window_size}, hop={hop}, even={even}, dt={dt}, dt_cycle_frac={dt_cycle_frac}, cycles={cycles}, freq={freq}, time_size={time_size}, window_fn={window_fn!r}, ra={ra!r}, filt={filt!r}, symetric={symetric!r}, zero_pad={zero_pad!r}".
            format(
                window_size=window_size,
                hop=hop,
                even=even,
                dt=dt,
                dt_cycle_frac=dt_cycle_frac,
                cycles=cycles,
                freq=freq,
                time_size=len(self.time),
                **kwargs))

        stfts = [
            -dipole.stft(window_size, hop, **kwargs)[1].T for dipole in dipoles
        ]
        time, efield = self.laser.efield.stft(window_size, hop, **kwargs)
        print(" got stfts")

        data = pd.DataFrame(
            list(zip(*map(lambda x: x[freq * bin_freq], stfts))),
            columns=self.dipole.names,
            index=time)
        data['efield'] = efield.T[freq * bin_freq]

        self.stfts = stfts + [efield.T]
        self.data = data

        return self.data

    def get_raw_stfts(self, **kwargs):
        if self.stfts is None:
            self.stft_data(**kwargs)

        return self.stfts

    def stft_susceptibility(self, **kwargs):
        if self.susceptibilities is not None:
            return self.susceptibilities
        if self.data is None:
            self.stft_data(**kwargs)

        print("Start of stft_susceptibility for %r" % self)

        mi = pd.MultiIndex.from_arrays(
            [self.data.index, self.stft_time_cycles(self.data.index)],
            names=("time", "cycles"))
        self.susceptibilities = pd.DataFrame(index=self.data.index)
        for name in self.dipole.names:
            name_ = name
            s1 = name[0]
            s2 = name[1]
            if self.dipole.l_decompositions:
                s1 = (name[0], name[-2])
                s2 = (name[1], name[-1])

                if name != "all" and (s1 != s2):
                    name_ = "%s + %s^*" % (name,
                                           "{s2[0]}{s1[0]}_{s2[1]}{s1[1]}".format(s2=s2,s1=s1))
            elif name != "all" and (s1 != s2):
                name_ = "%s + %s^*" % (name, "{s2[0]}{s1[0]}".format(s2=s2,s1=s1))
            self.susceptibilities[name_] = self.data[name] / self.data[
                "efield"]
        self.susceptibilities["efield"] = self.data["efield"]
        self.susceptibilities.index = mi

        return self.susceptibilities

    def stft_time_cycles(self, t):
        single_cycle = self.laser.period
        t = np.array(t)
        t_new = t - self.laser.center_of_pulse
        t_new = t_new / single_cycle

        return t_new

    def save(
            self,
            folder=None, ):
        if folder is None:
            folder = self.path

        self.data.to_hdf(
            join(self.path, "abinitio_data.hdf"),
            key="data_field",
            mode='a',
            complib='blosc',
            data_columns=True)
        self.susceptibilities.to_hdf(
            join(self.path, "abinitio_data.hdf"),
            key="susceptibilities",
            mode='a',
            complib='blosc',
            data_columns=True)
