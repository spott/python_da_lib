"""
fourier module. contains the Fourier class.
"""
from __future__ import print_function
import numpy as np
from numpy import pi
import scipy.signal
import scipy.interpolate
from window import *


class Fourier(object):
    """
    A Fourier object takes a time series and a data series and takes the
    fourier transform of them and stores both in the object for later
    information retrieval.

    Attributes:
        time: the list of times.
        tdata: the data corresponding to the list of times.
        freq: the energies of the fourier transform.
        fdata: the fourier transform data
        df: the spacing between the frequency points.
    """

    def __init__(self, time, data, window=scipy.signal.boxcar):

        self.time = np.array(time)
        self.tdata = window(len(data)) * np.array(data)
        assert len(self.time) == len(
            self.tdata
        ), "Fourier.__init__: time and tdata must have same length"
        self.fdata_ = None
        self.freq_ = None
        self.df = None

    @property
    def fdata(self):
        if self.fdata_ is None:
            if self.tdata.dtype == np.dtype('d'):
                # real!
                self.fdata_ = 2 * np.fft.rfft(self.tdata) / len(self.tdata)
                self.freq_ = np.fft.rfftfreq(
                    len(self.tdata), (self.time[1] - self.time[0]) / (2. * pi))

            elif self.tdata.dtype == np.dtype('D'):
                self.fdata_ = 2 * np.fft.fft(self.tdata) / len(self.tdata)
                self.freq_ = np.fft.fftfreq(
                    len(self.tdata), (self.time[1] - self.time[0]) / (2. * pi))
            else:
                raise Exception(
                    "Fourier.__init__: Not using double complex, or double... this is wrong"
                )
            self.df = self.freq[2] - self.freq[1]
        return self.fdata_

    @property
    def freq(self):
        if self.fdata_ is None:
            self.fdata
        return self.freq_

    def nyquist(self, val=None):
        if val is None:
            return .5 / (self.time[1] - self.time[0])
        else:
            return val / (2. * np.pi * self.nyquist())

    def __call__(self, freq):
        if self.df is None:
            self.fdata
        point = freq / self.df
        i = int(np.round(point))
        return self.fdata[i]

    def interpolated_freq(self, freq):
        """ returns the frequency asked for, interpolated instead of
        rounded"""
        i = int(np.round(freq / self.df))
        iinitial = i - 10 if i > 10 else 0
        ifinal = i + 10 if len(self.freq) - i < 10 else len(self.freq)
        return scipy.interpolate.InterpolatedUnivariateSpline(
            self.freq[iinitial:ifinal],
            np.abs(self.fdata[iinitial:ifinal]))(freq)

    def integrated_freq(self, fn, a, b):
        """ integrates the frequency over the range. Uses an interpolant"""
        return scipy.interpolate.InterpolatedUnivariateSpline(
            self.freq, fn(self.fdata), k=1).integral(a, b)

    def __repr__(self):
        return "<Fourier: tdata shape = %s, time shape = %s >" % (
            repr(self.tdata.shape), repr(self.time.shape))

    def stft(self,
             window_size,
             hop,
             window_fn=scipy.signal.flattop,
             ra=[0, -1],
             filt=None,
             symetric=True,
             zero_pad=None):

        #print("stft options: window_size={window_size}, hop={hop}, window_fn={window_fn!r}, ra={ra!r}, filt={filt!r}, symetric={symetric!r}, zero_pad={zero_pad!r}")

        if ra[1] == -1:
            ra[1] = len(self.time)

        # ra[1] = ra[1] - window_size / 2.
        ra = np.searchsorted(self.time, ra)
        # better reconstruction with this trick +1)[:-1]
        w = window_fn(window_size, symetric)  # [:-1]
        # df = 2. * np.pi / self.time[window_size]
        time = np.array([
            self.time[j + int(window_size / 2)]
            for j in range(ra[0], ra[1] - window_size, hop)
        ])

        # remove the DC average
        print("#", end='')

        if self.tdata.dtype == np.dtype('d'):
            # real!
            fft_func = np.fft.rfft
            # self.fdata_ = 2 * np.fft.rfft(self.tdata) / len(self.tdata)
            # self.freq_ = np.fft.rfftfreq(
            #     len(self.tdata), (self.time[1] - self.time[0]) / (2. * pi))

        elif self.tdata.dtype == np.dtype('D'):
            fft_func = np.fft.fft
            # self.fdata_ = 2 * np.fft.fft(self.tdata) / len(self.tdata)
            # self.freq_ = np.fft.fftfreq(
            #     len(self.tdata), (self.time[1] - self.time[0]) / (2. * pi))

        if filt is not None and filt in ['constant', 'linear']:
            chi = np.array([
                fft_func(w * (scipy.signal.detrend(
                    self.tdata[i:i + window_size], type=filt)), zero_pad) * 2. /
                window_size for i in range(ra[0], ra[1] - window_size, hop)
            ])
        elif filt is None:
            chi = np.array([
                fft_func(w *
                         (self.tdata[i:i + window_size]), zero_pad) * 2. / window_size
                for i in range(ra[0], ra[1] - window_size, hop)
            ])
        else:
            b, a = scipy.signal.butter(
                filt[0], self.nyquist(filt[1]), btype='high')
            filtered_data = scipy.signal.lfilter(b, a, self.tdata)
            chi = np.array([
                fft_func(w *
                         (filtered_data[i:i + window_size]), zero_pad) * 2. / window_size
                for i in range(ra[0], ra[1] - window_size, hop)
            ])

        print(".", end='')
        assert len(time) == len(
            chi), "stft: time and chi have different lengths"
        return (time, chi)
