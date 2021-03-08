from scipy import signal
from scipy.stats import norm
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd

Rin = 300000
Rfinal = 30000
N = int(Rin/Rfinal)

def get_spectrum(fn):
    """Function to read spectral data
       Args:
            fn: filename
       Returns:
            wavelength, flux

    """

    data=np.loadtxt(fn)

    data = data[200000:250000]
    normalized =  data[:,1]/data[:,2] #normalize spectrum

    flux = data[:,1]/data[:,2]
    wavelength = data[:,0]

    return wavelength, flux

def smooth_wave(wave, spec, outwave, sigma, nsigma=10, inres=0):
    """Smooth a spectrum in wavelength space.
    Args:
        param wave: Wavelength vector of the input spectrum.

        param spec: Flux vector of the input spectrum.

        param outwave: Desired output wavelength vector.

        param sigma: Desired resolution (*not* FWHM) in wavelength units.  This can be a
        vector of same length as ``wave``, in which case a wavelength dependent
        broadening is calculated

        param nsigma: (optional, default=10): Number of sigma away from the output wavelength to consider in the
        integral.  If less than zero, all wavelengths are used.  Setting this
        to some positive number decreses the scaling constant in the O(N_out *
        N_in) algorithm used here.

        param inres: (optional, default: 0.0)
        Resolution of the input, in either wavelength units or
        lambda/dlambda (c/v).  Ignored if <= 0.

    Returns:
        The output smoothed flux vector, same length as ``outwave``.
    """
    # sigma_eff is in angstroms
    if inres <= 0:
        sigma_eff_sq = sigma**2
    else:
        sigma_eff_sq = sigma**2 - (wave/inres)**2

    if np.any(sigma_eff_sq < 0):
        raise ValueError("Desired wavelength sigma is lower than the value "
                         "possible for this input spectrum.")

    sigma_eff = np.sqrt(sigma_eff_sq)
    flux = np.zeros(len(outwave))
    for i, w in enumerate(outwave):
        x = (wave - w) / sigma_eff
        if nsigma > 0:
            good = np.abs(x) < nsigma
            x = x[good]
            _spec = spec[good]
        else:
            _spec = spec
        f = np.exp(-0.5 * x**2)
        flux[i] = np.trapz(f * _spec, x) / np.trapz(f, x)
    return outwave, flux

map_of_int = {'a': '0', 'b': '1', 'c':'2',
             'd': '3', 'e': '4', 'f':'5',
             'g':'6', 'h': '7', 'i':'8', 'j':'9'}


def get_idx(fn, path):
    name = fn.replace(path, '').replace('.spec.gz', '')
    idx = name[5:10].translate(str.maketrans(map_of_int))
    return idx

def run_analysis(fn, path):

    wavelength, flux = get_spectrum(fn)
    sigma_blaze = wavelength/Rfinal
    outwave = wavelength[1::N]
    waves, blaze = smooth_wave(wavelength, flux, outwave=outwave, sigma=sigma_blaze, inres=Rin)

    idx = get_idx(fn, path)

    return waves, blaze, idx

#for galah

path='./Sync_Spectra_Galah_minibatch/'
#waves, blaze, idx = run_analysis('./Sync_Spectra_Galah_minibatch/at12_acigf_t04875g2.43.spec.gz', path)'

df = pd.DataFrame()

i=0
if __name__ == "__main__":
    for name in glob.glob('./Sync_Spectra_Galah_minibatch/*'):
        waves, blaze, idx = run_analysis(name, path)
        print("Writing spec:", i)
        df['spec'+str(idx)] = blaze
        i += 1

df.to_csv('jo_galah_batch_training.csv')
