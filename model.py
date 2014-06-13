# coding: utf-8

""" Spectral fitting """

__author__ = "Andy Casey <andy@astrowizici.st>"

import logging
from itertools import chain

import numpy as np
import scipy.optimize as op, scipy.ndimage as ndimage, scipy.interpolate as interpolate

import moog
import utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ApproximateModelSpectrum(object):

    # Here we define MOOG keyword arguments because all classes should be consistent
    moog_kwargs = {
        "wl_cont": 2.,
        "wl_edge": 2.,
        "parallel": True
    }
    
    def __init__(self, model_atmosphere_filename, line_list_filename, elements,
        convolve=False, v_rad=False, continuum=False, continuum_order=-1):
        """

        Convolve and v_rad can be True/False or actual values. If True, it's a
        free parameter. Otherwise fixed.

        Continuum can either be True (in which case continuum_order should be
        specified), or a lambda function that takes dispersion as a single value
        (in which case the continuum enters multiplicatively)

        """

        if isinstance(elements, (str, unicode)):
            elements = [elements]

        # Check the elements
        assert len(elements) > 0
        assert map(utils.element_to_species, elements)

        self.parameters = []
        self.fixed_parameters = {}

        self.elements = elements
        self.continuum_order = continuum_order
        self.line_list_filename = line_list_filename
        self.model_atmosphere_filename = model_atmosphere_filename

        # Determine the parameters
        if v_rad is True:
            self.parameters.append("v_rad")
        elif v_rad is not False and isinstance(v_rad, (float, int)):
            # Velocity is fixed.
            self.fixed_parameters["v_rad"] = v_rad

        if convolve is True:
            self.parameters.append("convolve")
        elif convolve is not False and isinstance(convolve, (float, int)) and convolve > 0:
            # Convolution is fixed.
            self.fixed_parameters["convolve"] = convolve

        if continuum is True:
            if continuum_order >= 0:
                self.parameters.extend(["c_{0}".format(i) for i in range(continuum_order + 1)])
                
            else:
                logging.warn("Continuum was specified as True but 0 > continuum_order")

        elif hasattr(continuum, "__call__"):
            self.fixed_parameters["continuum"] = continuum

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

        # Ensure we haven't done anything silly
        assert len(set(self.parameters).intersection(self.fixed_parameters.keys())) == 0
        assert len(set(self.parameters)) == len(self.parameters)


    def __call__(self, theta):
        """
        Return model dispersion and flux.
        """

        theta_dict = dict(zip(self.parameters, theta))
        get_parameter = lambda p, d: theta_dict.get(p, self.fixed_parameters.get(p, d)) 

        # Do abundances
        abundances = {}
        for element in self.elements:
            neutral_species = np.floor(utils.element_to_species(element))
            abundances[neutral_species] = [theta_dict["Z_"+element]]
            abundances[neutral_species + 0.1] = [theta_dict["Z_"+element]]

        # Synthesise flux
        with moog.instance() as moogsilent:
            model_dispersion, model_fluxes = moogsilent.synth(self.model_atmosphere_filename,
                self.line_list_filename, abundances=abundances, **self.moog_kwargs)

        # Select only the first spectrum
        model_flux = model_fluxes[0]

        # Any convolution to apply?
        convolve = abs(get_parameter("convolve", 0))
        if convolve > 0:
            ndimage.gaussian_filter(model_flux, convolve, output=model_flux)

        # Any radial velocity to apply?
        # TODO: YOU SHOULD DO THIS ON A DISPERSION THAT IS UNIFORM IN LOG-LAMBDA
        z = get_parameter("v_rad", 0)/299792458e-3
        model_dispersion *= 1. + z

        # Any continuum transformation to apply?
        if "continuum" in self.fixed_parameters:
            # Continuum here is a function that takes dispersion as a single input
            continuum = self.fixed_parameters["continuum"](model_dispersion)
            model_flux *= continuum

        elif self.continuum_order >= 0:
            continuum = np.polyval([theta_dict["c_{0}".format(i)] \
                for i in range(self.continuum_order + 1)], model_dispersion)
            model_flux *= continuum

        return (model_dispersion, model_flux)


    def optimise(self, data, p0=None, p0_abundance_range=[-2, -1, 0, 1, 2], xtol=0.01,
        maxiter=10, full_output=False, op_kwargs=None):

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
        p0 = [p0.get(parameter, 0) for parameter in self.parameters]
        p0_abundance_range = np.array(p0_abundance_range)

        if op_kwargs is None:
            op_kwargs = {"disp": False}
        else:
            if "xtol" in op_kwargs:
                del op_kwargs["xtol"]

        # Let's synthesize 5 spectra at once then use them for interpolation
        abundances = {}
        for element in self.elements:
            neutral_species = np.floor(utils.element_to_species(element))
            abundances[neutral_species] = p0_abundance_range
            abundances[neutral_species + 0.1] = p0_abundance_range

        niter, warnflag, converged, previous_abundances = 0, 0, False, None
        while not converged:
            
            with moog.instance() as moogsilent:
                model_dispersion, model_fluxes = moogsilent.synth(self.model_atmosphere_filename,
                    self.line_list_filename, abundances=abundances, **self.moog_kwargs)

            # Create an interpolator to interpolate between these spectra
            if len(self.elements) == 1:
                interpolator = interpolate.interp1d(abundances.values()[0], np.array(model_fluxes).T,
                    bounds_error=False)

            else:
                raise NotImplementedError("use LinearNDInterpolator")

            # Define a chi-sq function to fit to the data
            def chi_sq(theta):

                theta_dict = dict(zip(self.parameters, theta))
                get_parameter = lambda p, d: theta_dict.get(p, self.fixed_parameters.get(p, d)) 

                interpolated_flux = interpolator(*[theta_dict["Z_"+element] for element in self.elements])

                # Any convolution to apply?
                convolve = abs(get_parameter("convolve", 0))
                if convolve > 0:
                    ndimage.gaussian_filter(interpolated_flux, convolve, output=interpolated_flux)

                # Any radial velocity to apply?
                z = get_parameter("v_rad", 0)/299792458e-3

                # Must match model dispersion onto observed dispersion
                # TODO: YOU SHOULD DO THIS ON A DISPERSION THAT IS UNIFORM IN LOG-LAMBDA
                interpolated_flux = np.interp(dispersion, model_dispersion * (1. + z),
                    interpolated_flux, left=np.nan, right=np.nan)

                # Any continuum transformation to apply?
                if "continuum" in self.fixed_parameters:
                    # Continuum here is a function that takes dispersion as a single input
                    continuum = self.fixed_parameters["continuum"](dispersion)
                    interpolated_flux *= continuum

                elif self.continuum_order >= 0:
                    continuum = np.polyval([theta_dict["c_{0}".format(i)] \
                        for i in range(self.continuum_order + 1)], dispersion)
                    interpolated_flux *= continuum

                # Calculate the chi-sq
                chi_sq = (flux - interpolated_flux)**2 * ivar
                finite = np.isfinite(chi_sq)
                if not np.any(finite):
                    return np.inf

                return np.sum(chi_sq[finite])

            # Optimise!
            opt_p0 = op.fmin_powell(chi_sq, p0, xtol=xtol, **op_kwargs)
            if len(self.parameters) == 1:
                opt_p0 = [opt_p0]
            opt_p0_dict = dict(zip(self.parameters, opt_p0))

            # Check for convergence
            current_abundances = np.array([opt_p0_dict["Z_"+element] for element in self.elements])
            if previous_abundances is not None:
                if np.all(xtol >= np.abs(current_abundances - previous_abundances)):
                    # Achievement unlocked: convergence.
                    break

            # No convergence. Prepare for the next loop.
            previous_abundances = current_abundances

            # Half the p0_range and check for convergence
            for element in self.elements:
                neutral_species = np.floor(utils.element_to_species(element))

                opt_value = opt_p0_dict["Z_"+element]

                n_spectra = len(abundances[neutral_species])
                half_range = np.ptp(abundances[neutral_species])/2.
                new_abundance_range = np.linspace(opt_value - half_range/2., opt_value + half_range/2., n_spectra)
                abundances[neutral_species] = new_abundance_range
                abundances[neutral_species + 0.1] = new_abundance_range

            p0 = opt_p0

            niter += 1
            if niter >= maxiter:
                warnflag = 1
                logging.warn("Maximum number of iterations reached.")
                break

        if full_output:
            return opt_p0, warnflag

        return opt_p0







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

        with moog.instance(prefix="moog", debug=True) as moogsilent:

            def chi_sq(theta):
                
                theta_dict = dict(zip(self.parameters, theta))

                model_dispersion, model_flux = self._synthesise_(theta, moog_instance=moogsilent)

                # Take only the first spectrum
                model_flux = model_flux[0]
                convolve = abs(theta_dict.get("convolve", 0))
                if convolve > 0:
                    ndimage.gaussian_filter(model_flux, convolve, output=model_flux)
                
                # Due to rounding errors, the length of model_dispersion won't always
                # match dispersion, so we must interpolate
                model_flux = np.interp(dispersion, model_dispersion * (1. + theta_dict.get("v_rad", 0)/299792458e-3),
                model_flux, left=np.nan, right=np.nan)

                # Create a mask of which regions to use?

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
aa = ApproximateModelSpectrum("marcs-sun.model", "linelists/hermes/lin56683new", "Fe")
data = np.loadtxt("spectra/uvessun2.txt", skiprows=1)

class spectrum(object):
    pass

b = spectrum()

b.dispersion = data[:,0]
b.flux = data[:,1]
b.variance =  np.array([0.0001] * len(data))



