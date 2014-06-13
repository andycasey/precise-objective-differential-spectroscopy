# coding: utf-8

""" Spectral fitting """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
from itertools import chain

import numpy as np
import scipy.optimize as op, scipy.ndimage as ndimage

import moog
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ModelSpectrum(object):

    moog_kwargs = {
        "wl_cont": 2.,
        "wl_edge": 2.
    }

    def __init__(self, model_atmosphere_filename, line_list_filename, elements,
        convolve=True, v_rad=True, outliers=False, continuum_order=-1):

        if isinstance(elements, (str, unicode)):
            elements = [elements]

        # Check the elements
        assert len(elements) > 0
        assert map(utils.element_to_species, elements)

        self.elements = elements
        self.line_list_filename = line_list_filename
        self.model_atmosphere_filename = model_atmosphere_filename

        # Determine the parameters
        self.parameters = []
        if v_rad:
            self.parameters.append("v_rad")
        if convolve:
            self.parameters.append("convolve")
        if outliers:
            self.parameters.extend(["Pb", "Vb", "Yb"])
        if continuum_order >= 0:
            self.parameters.extend(["c{0}".format(i) for i in range(continuum_order + 1)])

        self.parameters.extend(map(lambda x: "Z_{0}".format(x), self.elements))

        # Determine wl_min and wl_max
        try:
            wavelengths = np.loadtxt(line_list_filename, usecols=(0, ))
        except (ValueError, TypeError) as e:
            # First row is probably a comment
            wavelengths = np.loadtxt(line_list_filename, usecols=(0, ))

        self.moog_kwargs.update({
            "wl_min": np.min(wavelengths) - self.moog_kwargs["wl_edge"],
            "wl_max": np.max(wavelengths) + self.moog_kwargs["wl_edge"]
        })


    def optimise(self, data, p0=None):

        dispersion, flux, variance = data.dispersion, data.flux, data.variance

        # Slice the data, clean up non-finite fluxes, etc
        indices = np.searchsorted(data.dispersion,
            [self.moog_kwargs["wl_min"], self.moog_kwargs["wl_max"]])
        dispersion = data.dispersion.__getslice__(*indices)
        flux = data.flux.__getslice__(*indices)
        ivar = 1.0/data.variance.__getslice__(*indices)

        finite = np.isfinite(flux)

        # Approximate the wavelength sampling for MOOG
        self.moog_kwargs["wl_step"] = np.median(np.diff(dispersion))

        if p0 is None: p0 = {}
        opt_p0 = [p0.get(parameter, 0) for parameter in self.parameters]

        with moog.instance(prefix="moog") as moogsilent:

            def chi_sq(theta):
                
                theta_dict = dict(zip(self.parameters, theta))

                model_dispersion, model_flux = self._synthesise_(theta, moog_instance=moogsilent)
                # Due to rounding errors, the length of model_dispersion won't always
                # match dispersion, so we must interpolate
                model_flux = np.interp(dispersion, model_dispersion * (1. + theta_dict.get("v_rad", 0)/299792458e-3),
                    model_flux[0], left=np.nan, right=np.nan)

                convolve = abs(theta_dict.get("convolve", 0))
                if convolve > 0:
                    model_flux = ndimage.gaussian_filter(model_flux, convolve)
                
                chi_sq = (flux - model_flux)**2 * ivar
                finite = np.isfinite(chi_sq)
                if not np.any(finite):
                    return np.inf

                result = np.sum(chi_sq[finite])
                print(theta, result)

                return result

            final_p0 = op.fmin_powell(chi_sq, opt_p0, xtol=0.01)

        return final_p0


    def _synthesise_(self, theta, moog_instance=None, **kwargs):

        theta_dict = dict(zip(self.parameters, theta))
        
        # Parse abundances
        abundances = {}
        for parameter, value in theta_dict.iteritems():
            if parameter[:2] == "Z_":
                neutral_species = np.floor(utils.element_to_species(parameter[2:]))

                abundances[neutral_species] = [value]
                abundances[neutral_species + 0.1] = [value]

        # Synthesise and return a spectrum
        if moog_instance is not None:
            dispersion, flux = moog_instance.synth(self.model_atmosphere_filename,
                self.line_list_filename, abundances=abundances, **self.moog_kwargs)

        else:
            with moog.instance(prefix="moog") as moogsilent:
                # Flux is a list because we can request multiple spectra from MOOG in one synthesis
                dispersion, flux = moogsilent.synth(self.model_atmosphere_filename,
                    self.line_list_filename, **self.moog_kwargs)

        return (dispersion, flux)


a = ModelSpectrum("marcs-sun.model", "linelists/hermes/lin56683new", "Fe")
data = np.loadtxt("/Users/arc/observing/kurucz-1984-atlas/lm0564")

class spectrum(object):
    pass

b = spectrum()

b.dispersion = data[:,0] * 10.
b.flux = data[:,1]
b.variance =  np.array([0.0001] * len(data))



