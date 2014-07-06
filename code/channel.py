# coding: utf-8

""" Spectral Channels. """

__author__ = "Andy Casey <andy@astrowizici.st>"

import cPickle as pickle
import logging

import emcee
import numpy as np
import numpy.lib.recfunctions as nprcf
from scipy import optimize as op, ndimage

__all__ = ["SpectralChannel"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _absorption_line_(wavelength, depth, sigma, x):
    return 1. - depth * np.exp(-(x - wavelength)**2 / (2 * sigma**2))

def channel_ln_prior(theta, parameters, transitions, wl_tolerance):
    theta_dict = dict(zip(parameters, theta))

    if 0 > theta_dict["smoothing_sigma"]:
        logger.debug("prior fails because of smoothing_sigma")
        return -np.inf 

    # Check outlier parameters.
    if not (1 > theta_dict.get("Po", 0.5) > 0) or 0 > theta_dict.get("Vo", 1):
        logger.debug("prior fails because of Vo or Po")
        return -np.inf

    # Check line depths and wavelengths.
    i = 0
    while "ld_{0}".format(i) in parameters:
        #if not (1 >= theta_dict.get("ld_{0}".format(i), 0.5) >= 0):
        #    logger.debug("prior fails because of ld_{0}".format(i))
        #    return -np.inf

        wavelength = theta_dict.get("wl_{0}".format(i), transitions[i]["rest_wavelength"])
        if np.abs(wavelength - transitions[i]["rest_wavelength"]) > wl_tolerance:
            logger.debug("prior fails because of wl_{0}".format(i))
            return -np.inf
        i += 1

    return 0


def channel_model(theta_dict, dispersion, transitions, intensities=None, wl_cont=2):

    if intensities is None:
        intensities = np.ones(len(dispersion))
    else:
        intensities = intensities.copy()

    # Add absorption lines.
    i, z, sigma = (0, theta_dict.get("z", 0), theta_dict["smoothing_sigma"])
    while "ld_{0}".format(i) in theta_dict:
        depth = theta_dict["ld_{0}".format(i)]
        wavelength = (1. + z) * theta_dict.get("wl_{0}".format(i), transitions[i]["rest_wavelength"])
        sigma = theta_dict["smoothing_sigma"] * wavelength/dispersion[0]
        indices = dispersion.searchsorted([wavelength - wl_cont, wavelength + wl_cont])
        x = dispersion.__getslice__(*indices)
        y = intensities.__getslice__(*indices)
        intensities.__setslice__(indices[0], indices[1],
            y * _absorption_line_(wavelength, depth, sigma, x))
        i += 1

    # Fill nans.
    #intensities[intensities == 1.] = np.nan

    # Scale continuum.
    i, coefficients = 0, []
    while "c_{0}".format(i) in theta_dict:
        coefficients.append(theta_dict["c_{0}".format(i)])
        i += 1
    if i > 0:
        intensities *= np.polyval(coefficients, dispersion)

    return intensities


def channel_ln_likelihood(theta, dispersion, flux, variance, parameters, transitions,
    intensities, wl_tolerance, wl_cont):

    theta_dict = dict(zip(parameters, theta))
    model_flux = channel_model(theta_dict, dispersion, transitions, intensities, wl_cont)

    # Calculate likelihood.
    ivar = 1.0/variance
    signal_ln_like = -0.5 * ((model_flux - flux)**2 * ivar - np.log(ivar))
    if "Po" in parameters:
        Po, Yo, Vo = [theta_dict[each] for each in ["Po", "Yo", "Vo"]]
        ivar = 1.0/(variance + Vo)
        outlier_ln_like = -0.5 * ((Yo - flux)**2 * ivar - np.log(ivar))

        ln_like = np.logaddexp(np.log(1-Po) + signal_ln_like, np.log(Po) + outlier_ln_like)
    else:
        ln_like = signal_ln_like

    finite = np.isfinite(ln_like)
    return np.sum(ln_like[finite])


def channel_ln_probability(theta, dispersion, flux, variance, parameters, transitions,
    intensities, wl_tolerance=0, wl_cont=2):

    ln_prior = channel_ln_prior(theta, parameters, transitions, wl_tolerance)
    if not np.isfinite(ln_prior):
        return -np.inf
    return ln_prior + channel_ln_likelihood(theta, dispersion, flux, variance, parameters,
        transitions, intensities, wl_tolerance, wl_cont)


class SpectralChannel(object):

    upper_smoothing_sigma = 0.5

    def __init__(self, data, transitions, mask=None, redshift=False, continuum_order=-1,
        outliers=False, wl_tolerance=0, wl_cont=1):

        self.dispersion = data.dispersion
        self.data = data.flux
        self._finite = np.isfinite(self.data)
        self.variance = data.variance

        self.transitions = transitions

        # Check that we have the fields we want. If not, add them.
        additional_fields = ("equivalent_width", "abundance")
        for field in additional_fields:
            if field not in self.transitions.dtype.names:
                self.transitions = nprcf.append_fields(self.transitions, field,
                    [np.nan] * len(self.transitions), usemask=False)

        self.wl_cont = wl_cont
        self.redshift = redshift
        self.outliers = outliers
        self.wl_tolerance = wl_tolerance
        self.continuum_order = continuum_order
        self.mask = mask

        # Create parameter list.
        self.parameters = []
        if redshift: self.parameters.append("z")
        #self.parameters.append("smoothing_sigma")
        self.parameters.extend(["c_{0}".format(i) for i in range(continuum_order + 1)])

        # Outliers?
        if outliers:
            self.parameters.extend(["Po", "Vo", "Yo"])
            
        # Line depths.
        self.parameters.extend(["ld_{0}".format(i) for i in range(len(transitions))])

        # Line sigmas
        self.parameters.extend(["sigma_{0}".format(i) for i in range(len(transitions))])

        # Are wavelengths allowed to vary?
        if wl_tolerance > 0:
            self.parameters.extend(["wl_{0}".format(i) for i in range(len(transitions))])

        # Check for nearby-ish lines.
        sorted_wavelengths = np.sort(self.transitions["rest_wavelength"])
        distances = np.diff(sorted_wavelengths)
        if wl_cont > 0 and wl_cont > np.min(distances):
            nearby_line_indices = np.where(wl_cont > distances)[0]
            for nearby_index in nearby_line_indices:
                logging.warn("Transitions at {0:.2f} and {1:.2f} are very close ({2:.2f} < {3:.2f} Angstroms)".format(
                    sorted_wavelengths[nearby_index], sorted_wavelengths[nearby_index + 1], 
                    float(np.abs(np.diff(sorted_wavelengths[nearby_index:nearby_index+2]))), wl_cont))



    def _fit(self, theta, full_output=False, verbose=True, fail_value=9e99):

        theta_dict = dict(zip(self.parameters, theta))
        if self.outliers:
            Po, Yo, Vo = [theta_dict[each] for each in ("Po", "Yo", "Vo")]
            if not (1 > Po > 0) or 0 > Vo:
                return fail_value

        # Create continuum shape.
        continuum = np.ones(len(self.data))
        if self.continuum_order > -1:
            continuum_coefficients = [theta[self.parameters.index("c_{0}".format(i))] \
                for i in range(self.continuum_order + 1)]
            continuum *= np.polyval(continuum_coefficients, self.dispersion)
        
        # Add absorption lines.
        for i, transition in enumerate(self.transitions):
            wavelength = theta_dict["wl_{0}".format(i)] if self.wl_tolerance > 0 else transition["rest_wavelength"]
            depth, sigma = theta_dict["ld_{0}".format(i)], theta_dict["sigma_{0}".format(i)]
            if not (1 >= depth >= 0) or not (self.upper_smoothing_sigma > sigma > 0) \
            or abs(wavelength - transition["rest_wavelength"]) > self.wl_tolerance:
                return fail_value

            z = args[0] if self.redshift else 0
            wavelength *= (1. + z)
            if self.wl_cont > 0:
                indices = self.dispersion.searchsorted([
                    wavelength - self.wl_cont,
                    wavelength + self.wl_cont
                ])
                x = self.dispersion.__getslice__(*indices)
                y = continuum.__getslice__(*indices)
                continuum.__setslice__(indices[0], indices[1],
                    y * _absorption_line_(wavelength, depth, sigma, x)
                )

            else:
                continuum *= _absorption_line_(wavelength, depth, sigma, self.dispersion)

        ivar = 1.0/self.variance
        chi_sq = (continuum - self.data)**2 * ivar
        
        if self.outliers:

            signal_ln_like = -0.5 * (chi_sq - np.log(ivar))

            ivar = 1.0/(self.variance + Vo)
            outlier_ln_like = -0.5 * ((continuum - Yo)**2 * ivar - np.log(ivar))

            # Make this the negative log-likelihood, since this function will only be used in optimisation.
            ln_like = np.logaddexp(np.log(1-Po) + signal_ln_like, np.log(Po) + outlier_ln_like)
            finite = np.isfinite(ln_like) * np.isfinite(self.mask)
            ln_like = -np.sum(ln_like[finite])

            print(ln_like)
            if verbose:
                print(ln_like)
            
            if full_output:
                return (ln_like, continuum)
            return ln_like

        # No outlier modelling; just return chi-sq.
        chi_sq = np.sum(chi_sq[np.isfinite(self.mask)])
        print(chi_sq)
        if verbose:
            print(chi_sq)
        if full_output:
            return (chi_sq, continuum)
        return chi_sq


    def optimise(self, max_iterations=3, initial_clip_iter=5, initial_clip_limits=(0.2, 3.0), verbose=False, line_kwargs=None, channel_kwargs=None, plot_filename=None,
        plot_clobber=False):
        # Optimise the channel and line parameters.

        op_line_kwargs = { "xtol": 1e-8, "maxfun": 10e3, "maxiter": 10e3,
            "full_output": True, "disp": False }
        op_channel_kwargs = { "ftol": 1e-3, "maxfun": 10e4, "maxiter": 10e4,
            "full_output": True, "disp": False } 

        if line_kwargs is not None:
            op_line_kwargs.update(line_kwargs)
        if channel_kwargs is not None:
            op_channel_kwargs.update(channel_kwargs)
        
        # We might not have all these parameters, but we can set some defaults.
        default_p0 = {
            "z": 0,
            "smoothing_sigma": 0.05,
            "Po": 0.5,
            "Yo": np.median(self.data[self._finite]),
            "Vo": np.std(self.data[self._finite])**2
        }

        # Wavelengths.
        default_p0.update(dict(zip(
            ["wl_{0}".format(i) for i in range(len(self.transitions))],
            [each["rest_wavelength"] for each in self.transitions]
        )))
        default_p0.update(dict(zip(
            ["sigma_{0}".format(i) for i in range(len(self.transitions))],
            [default_p0["smoothing_sigma"]] * len(self.transitions)
        )))

        # Continuum coefficients.
        if self.continuum_order >= 0:
            continuum_mask = np.isfinite(self.mask)
            continuum_coefficients = np.polyfit(self.dispersion, self.data, self.continuum_order)
            
            # Perform some crude sigma-clipping because we expect absorption features.
            if self.continuum_order > -1:
                lower_clip, upper_clip = initial_clip_limits
                for i in xrange(initial_clip_iter):

                    difference = np.polyval(continuum_coefficients, self.dispersion) - self.data
                    sigma = np.std(difference)
                    exclude = (difference > lower_clip * sigma) + (difference < -upper_clip*sigma)

                    # Remove the excluded points.
                    continuum_coefficients = np.polyfit(self.dispersion[~exclude], self.data[~exclude],
                        self.continuum_order)

            # Save the continuum coefficients.
            default_p0.update(dict(zip(
                ["c_{0}".format(i) for i in range(self.continuum_order + 1)],
                continuum_coefficients
            )))

        continuum_shape = np.ones(len(self.data))
        if self.continuum_order > -1:
            continuum_shape *= np.polyval(continuum_coefficients, self.dispersion)

        # We will iterate between the global and local line fitting.
        for i in xrange(max_iterations):

            xopts = []
            # Fit each line using the continuum we've set.
            for j, transition in enumerate(self.transitions):

                def fit_absorption_line(args, continuum, full_output, fail_value=np.inf):
                    
                    model = continuum.copy()

                    if self.wl_tolerance > 0:
                        depth, sigma, wavelength = args

                    else:
                        depth, sigma = args
                        wavelength = transition["rest_wavelength"]

                    if not (1 > depth > 0) \
                    or not (self.upper_smoothing_sigma > sigma > 0) \
                    or abs(wavelength - transition["rest_wavelength"]) > (self.wl_tolerance - 1e-3):
                        return fail_value

                    z = args[0] if self.redshift else 0
                    wavelength *= (1. + z)
                    
                    if self.wl_cont > 0:
                        indices = self.dispersion.searchsorted([
                            wavelength - self.wl_cont,
                            wavelength + self.wl_cont
                        ])
                        x = self.dispersion.__getslice__(*indices)
                        y = model.__getslice__(*indices)

                        model.__setslice__(indices[0], indices[1],
                            y * _absorption_line_(wavelength, depth, sigma, x)
                        )

                    else:
                        model *= _absorption_line_(wavelength, depth, sigma, self.dispersion)

                    chi_sqi = (model - self.data)**2/self.variance
                    chi_sq = chi_sqi[np.isfinite(chi_sqi * self.mask)].sum()
                    if full_output:
                        return (chi_sq, model)
                    return chi_sq

                if i == 0:
                    index = self.dispersion.searchsorted(transition["rest_wavelength"])

                    line_p0 = [1. - self.data[index]/continuum_shape[index], default_p0["sigma_{0}".format(j)]]
                    if self.wl_tolerance > 0:
                        line_p0.append(transition["rest_wavelength"])
                else:
                    line_p0 = np.array([default_p0["ld_{0}".format(j)], default_p0["sigma_{0}".format(j)], default_p0["wl_{0}".format(j)]])

                index = self.dispersion.searchsorted(transition["rest_wavelength"])
                line_p0 = np.array([1. - self.data[index]/continuum_shape[index], default_p0["smoothing_sigma"], transition["rest_wavelength"]])

                xopt, fopt, niter, nfuncs, warnflag = op.fmin(fit_absorption_line, line_p0,
                    args=(continuum_shape, False), **op_line_kwargs)
                xopts.append(xopt)
                if warnflag > 0:
                    message = [
                        "Che problem?",
                        "Maximum number of function evaluations made",
                        "Maximum number of iterations made"
                    ]
                    logging.warn("{0} for transition at {1}".format(message[warnflag], transition["rest_wavelength"])) 

                # Update the default_p0 values.
                default_p0["ld_{0}".format(j)] = np.clip(xopt[0], 0, 1)
                default_p0["sigma_{0}".format(j)] = xopt[1]
                if self.wl_tolerance > 0:
                    default_p0["wl_{0}".format(j)] = xopt[2]

            # Optimise the parameters globally if required.
            # Note: the smoothing kernel is always a global parameter, but the median of the line sigmas
            #       is a sufficient approximation. So here we are just checking for either redshift or
            #       continuum coefficients.
            channel_p0 = np.array([default_p0[parameter] for parameter in self.parameters])

            if i == 0:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.plot(self.dispersion, self._fit(channel_p0, True, False)[1], 'r')
                ax.plot(self.dispersion, self.data, 'k')
                plt.savefig("moo.pdf")

            if self.continuum_order > -1 or "z" in self.parameters or force:
                print("doing it")

                bounds = []
                for k, parameter in enumerate(self.parameters):
                    if parameter.startswith("wl_"):
                        index = int(parameter.split("_")[1])
                        bounds.append([
                            self.transitions[index]["rest_wavelength"] - self.wl_tolerance,
                            self.transitions[index]["rest_wavelength"] + self.wl_tolerance
                        ])

                    elif parameter.startswith("sigma_"):
                        bounds.append([0., self.upper_smoothing_sigma])

                    elif parameter.startswith("ld_") or parameter == "Po":
                        bounds.append([0., 1.])

                    elif parameter == "Vo":
                        bounds.append([0, None])

                    elif parameter == "Yo":
                        bounds.append([0, None])

                    else:
                        bounds.append((None, None))


                #xopt, fopt, info = op.fmin_l_bfgs_b(self._fit, channel_p0,
                #    args=(False, verbose), m=100, pgtol=1e-32, approx_grad=True, bounds=bounds)
                xopt, fopt, iterations, funcalls, warnflag = op.fmin(self._fit, channel_p0,
                    args=(False, verbose), **op_channel_kwargs)
                
                if warnflag > 0:
                    message = [
                        "Che problem?",
                        "Maximum number of function evaluations made",
                        "Maximum number of iterations made"
                    ]
                    logging.warn("{0} for channel fit. Optimised values may be (even more) inaccurate.".format(
                        message[warnflag]))

            else:
                xopt = channel_p0


            # Update the default_p0 values with our optimised values for the next iteration.
            default_p0.update(dict(zip(self.parameters, xopt)))

        """
        fig, ax = plt.subplots()
        ax.plot(self.dispersion, self.data,'k')
        try:
            ax.plot(self.dispersion, self._fit(channel_p0, True)[1], 'r')
        except:
            None
        ax.plot(self.dispersion, self._fit(xopt, True)[1], 'b')
        raise a
        """

        fopt, model = self._fit(xopt, True)
        plot_filename = "moo2.pdf"
        if plot_filename is not None:

            # Produce a plot.
            """
            continuum = continuum_shape.copy()
            for transition, result in zip(self.transitions, line_fits):
                if self.wl_tolerance > 0:
                    depth, sigma, wavelength = result["x"]
                else:
                    depth, sigma = result["x"]
                    wavelength = transition["rest_wavelength"]

                continuum *= self._absorption_line_(wavelength, depth, sigma)
            """
            ax.plot(self.dispersion, self.data, 'k')
            ax.plot(self.dispersion, model, 'b')
            ax.set_xlim(self.dispersion[0], self.dispersion[-1])
            fig.savefig(plot_filename, clobber=plot_clobber)

        raise a
        # Infer dat shit.
        walkers = 200
        ndim = len(self.parameters)
        initial_pos = np.array([xopt + 1e-4 * np.random.randn(ndim) for i in range(walkers)])
        dat_inference = lambda x: -self._fit(x, False)
        sampler = emcee.EnsembleSampler(walkers, ndim, dat_inference)

        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(initial_pos, iterations=400)):
            print("mean acceptance: {0:.2f}, max ln prob: {1:.2f}".format(np.mean(sampler.acceptance_fraction),
                np.max(sampler.lnprobability[:, i])))

        raise a

        # Calculate equivalent widths in the same units as the dispersion.
        logging.warn("Assuming spectral dispersion units are Angstroms. Measured equivalent widths stored as mA.")
        self.transitions["equivalent_width"] = np.array([xopt[self.parameters.index("ld_{0}".format(i))] \
            for i in range(len(self.transitions))]) * xopt[0] * 1000. * 2.65 # TODO: Check integral.
        return (xopt, fopt, model)


    def infer(self, p0, walkers, burn, sample, threads=1):
        """
        Infer the parameters of this spectral channel.
        """

        # Initialise about p0
        ndim = len(self.parameters)
        initial_pos = np.array([p0 + 1e-4 * np.random.randn(ndim) for i in range(walkers)])
        sampler = emcee.EnsembleSampler(walkers, ndim, channel_ln_probability,
            args=(self.dispersion, self.data, self.variance, self.parameters, self.transitions,
                None, self.wl_tolerance, self.wl_cont), threads=threads)

        mean_acceptance_fractions = np.zeros(burn + sample)
        for i, (pos, lnprob, rstate) in enumerate(sampler.sample(initial_pos, iterations=burn)):
            mean_acceptance_fractions[i] = np.mean(sampler.acceptance_fraction)
            logger.info(u"Sampler has finished step {0:.0f} with <a_f> = {1:.3f}, maximum log probability"\
                " in last step was {2:.3e}".format(i + 1, mean_acceptance_fractions[i],
                    np.max(sampler.lnprobability[:, i])))

            if mean_acceptance_fractions[i] in (0, 1):
                raise RuntimeError("mean acceptance fraction is {0:.0f}!".format(mean_acceptance_fractions[i]))

        # Reset the chains and sample.
        logger.info("Resetting chain...")
        chain, lnprobability = sampler.chain, sampler.lnprobability
        sampler.reset()

        logger.info("Sampling posterior...")
        for j, state in enumerate(sampler.sample(pos, iterations=sample)):
            mean_acceptance_fractions[i + j + 1] = np.mean(sampler.acceptance_fraction)

        # Concatenate the existing chain and lnprobability with the posterior samples.
        chain = np.concatenate([chain, sampler.chain], axis=1)
        lnprobability = np.concatenate([lnprobability, sampler.lnprobability], axis=1)

        # Get the maximum likelihood theta.
        ml_index = np.argmax(lnprobability.reshape(-1))
        ml_values = chain.reshape(-1, ndim)[ml_index]

        # Get the quantiles.
        posteriors = {}
        for parameter_name, (ml_value, quantile_16, quantile_84) in zip(self.parameters, 
            map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
                zip(*np.percentile(sampler.chain.reshape(-1, ndim), [16, 50, 84], axis=0)))):
            posteriors[parameter_name] = (ml_value, quantile_16, quantile_84)

        # Send back additional information.
        info = {
            "chain": chain,
            "lnprobability": lnprobability
        }
        return (posteriors, sampler, info)



if __name__ == "__main__":

    class spectrum(object):
        pass

    data = np.loadtxt("spectra/uvessun1.txt", skiprows=1)

    blue_channel = spectrum()
    blue_channel.dispersion = data[:,0]
    blue_channel.flux = data[:, 1]
    blue_channel.variance =  np.array([0.0001] * len(blue_channel.dispersion))

    with open("transitions.pkl", "rb") as fp:
        transitions = pickle.load(fp)

    # Get just blue channel ones
    indices = (4895 > transitions["rest_wavelength"]) * (transitions["rest_wavelength"] > 4705)

    blue = SpectralChannel(blue_channel, transitions[indices], wl_tolerance=0.10, continuum_order=-1, outliers=True)

    p0 = blue.optimise()

    raise a

    posterior, sampler, info = blue.infer(p0[0], 400, 400, 100, 8)

    # Plot projections and chains?
    ml_theta_dict = dict([(key, posterior[key][0]) for key in posterior.keys()])
    model_flux_emcee = channel_model(ml_theta_dict, blue.dispersion, blue.transitions)
    model_flux_opt = channel_model(dict(zip(blue.parameters, p0[0])), blue.dispersion, blue.transitions)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(blue.dispersion, blue.data, 'k')
    ax.plot(blue.dispersion, model_flux_opt, 'r')
    ax.plot(blue.dispersion, model_flux_emcee, 'b')


