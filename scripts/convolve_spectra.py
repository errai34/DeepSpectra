# -*- coding: utf-8 -*-

from scipy import signal
from scipy.stats import norm

filename = (“spec-%(d)5d-%(s)s_sp%(p)02d-%(f)03d.fits.gz” %\
                {‘d’: lmjd[i], ‘s’: planid[i].strip(),\
                 ‘p’: spid[i], ‘f’: fiberid[i]})
hdulist = fits.open(“lamost_DR5/” + filename)
flux = hdulist[0].data[0,:]
flux_err = 1./np.sqrt(hdulist[0].data[1,:]+ 1e-6)
wavelength = hdulist[0].data[2,:]
# make blaze
win = norm.pdf((np.arange(201)-100.)*(wavelength[1]-wavelength[0]),\
                    scale=50.)
win = win/np.sum(win)
blaze = signal.convolve(flux, win, mode=‘same’)
