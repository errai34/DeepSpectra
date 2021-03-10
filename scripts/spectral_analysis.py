import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy import signal
from scipy.stats import norm

import sys

sys.path.append("/Users/ioanaciuca/Desktop/The_Payne/")
from The_Payne import utils


# example data
data = np.loadtxt(
    "/Users/ioanaciuca/Desktop/MockSpectra/Sync_Spectra_All/at12_aaaaa_t04534g2.24.spec"
)
wavelength = utils.load_wavelength_array()

waves = data[:, 0]
flux = data[:, 1] / data[:, 2]  # this is the normalised flux


def gauss_kern(fwhm):
    """ Returns a normalized 1D gauss kernel array for convolutions """
    k = 2.355
    sigma_s = (fwhm / k) ** 2  # sigma sqared where sigma = fwhm/k
    size = 2 * sigma_s
    size_grid = int(
        size
    )  # we limit the size of kernel, so it is as small as possible (or minimal size) for faster calculations
    if size_grid < 7:
        size_grid = 7
    x = scipy.mgrid[-size_grid : size_grid + 1]
    g = scipy.exp(-(x ** 2 / float(size)))
    return g / np.sum(g)


def conv_spectrum(wavein, flux, r=300000, target_r=22000):
    """Perform resolution degradation."""

    w_avg = np.average(wavein)
    s = w_avg / r
    s_target = w_avg / target_r
    s_conv = np.sqrt(s_target ** 2 - s ** 2)
    step = waves[1] - waves[0]
    kernel = gauss_kern(s_conv / step)
    blaze = signal.fftconvolve(flux, kernel, mode="same")

    return blaze


def ys_conv(wavein, flux, r=300000, target_r=22000):
    """Performs the same as conv_spectrum but with YS's approach... Will use whichever is faster."""

    step = wavein[1] - wavein[0]  # choose step size

    w_avg = np.average(wavein)
    s = w_avg / r
    s_target = w_avg / target_r
    s_conv = np.sqrt(s_target ** 2 - s ** 2)

    c_eff = s_conv / 2.355  # k=2.355 is 2*np.sqrt(2*np.log(2)). s_conv=fwhm/k

    size_grid = 100
    x = np.arange(size_grid * 2 + 1) - size_grid

    win = norm.pdf(x * step, scale=c_eff)
    win = win / np.sum(win)

    blaze_ys = signal.convolve(flux, win, mode="same")

    return blaze_ys


def interpolate(waveinterp, wavein, flux):
    """
    interpolate the spectrum to a wavelength space defined in space."""
    waveinterp = np.array(waveinterp)
    flux_interp = np.interp(waveinterp, wavein, flux)

    return flux_interp


def pipeline(waveinterp, wavein, flux):
    """Function to put it all together."""
    blaze = ys_conv(wavein, flux, r=300000, target_r=22000)
    waveinterp = waveinterp/10 #divide by 10 
    blaze_interp = interpolate(waveinterp, wavein, blaze)

    return waveinterp[::3], blaze_interp[::3]


if __name__ == "__main__":
    waveout, fluxout = pipeline(wavelength, waves, flux)
    assert len(fluxout) == len(waveout)
