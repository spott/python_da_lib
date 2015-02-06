import math
import numpy as np


class atomic(object):
    ''' class that contains atomic unit conversions and constants '''
    press = 2.206825e11  # per torr #3.44386e-9
    kT = 3.2e-6  # one K
    intensity = 3.5094452e16
    c = 137.035999074
    e0 = 0.0795775

    @staticmethod
    def from_wavelength(l):
        return 45.56335/l

    @staticmethod
    def averaged_intensity(i, n):
        return (math.gamma(.5 + 2.*n) / (np.sqrt(np.pi) * math.gamma(1.+2.*n))) * (i/atomic.intensity)**n

    @staticmethod
    def temperature(kelvin):
        """ returns kT in Atomic Units for Kelvin given """
        return atomic.kT * kelvin

    @staticmethod
    def pressure(torr):
        """ returns the pressure in atomic units for pressure given """
        return atomic.press * torr

    #@staticmethod
    #def density(kelvin=300, torr=5):
        #""" density in number per bohr radius squared """
        #return atomic.pressure(torr)/atomic.temperature(kelvin)

    @staticmethod
    def density(torr=5):
        return (torr/5.)* 1.76e17 * (5.29e-5/1e4)**3

    @staticmethod
    def pondermotive (intensity_si, wavelength):
        return (intensity_si/atomic.intensity)/(4. * atomic.from_wavelength(wavelength))
