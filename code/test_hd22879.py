import cPickle as pickle

from stellar_parameters import Star
from channel import SpectralChannel




class spectrum(object):
    pass

import sick
spec = sick.specutils.Spectrum.load("spectra/hermes-sun.fits")
spec = sick.specutils.Spectrum.load("spectra/uvessun1.txt")


blue_channel = spectrum()
blue_channel.dispersion = spec.disp
blue_channel.flux = spec.flux
blue_channel.variance =  spec.variance


with open("transitions.pkl", "rb") as fp:
    transitions = pickle.load(fp)


with open("sousa-transitions.pkl", "rb") as fp:
    transitions = pickle.load(fp)


# Get just blue channel ones
transition_indices = (blue_channel.dispersion[-1] > transitions["rest_wavelength"]) * (transitions["rest_wavelength"] > blue_channel.dispersion[0])

use_regions = np.array([
    [4731.3, 4731.65],
    [4742.65, 4742.93],
    [4757.95, 4748.31],
    [4759.1, 4759.56],
    [4764.43, 4764.47],
    [4778.08, 4778.41],
    [4779.78, 4780.2],
    [4781.59, 4781.92],
    [4788.41, 4789],
    [4789.91, 4790.19],
    [4795.24, 4795.66],
    [4798.39, 4798.64],
    [4802.69, 4803.2],
    [4805.3, 4805.71],
    [4807.95, 4808.35],
    [4820.23, 4820.6],
    [4847.89, 4848.02],
    [4869.85, 4870.3],
    [4873.88, 4874.19],
    [4884.95, 4885.25],
    [4889.9, 4892.67],
    [4894.7, 4895.0]
])

#use_regions = np.array([
#    [4705, 4850.],
#    [4880., 5000.]
#])

mask = np.empty(len(blue_channel.dispersion))
mask[:] = np.nan
for row in use_regions:
    indices = blue_channel.dispersion.searchsorted(row)
    mask[indices[0]:indices[1] + 1] = 1.

print(np.sum(np.isfinite(mask)))

blue = SpectralChannel(blue_channel, transitions[transition_indices], mask=mask, redshift=False, continuum_order=-1,
    wl_tolerance=0.10, wl_cont=2, outliers=True)

xopt = blue.optimise(plot_filename="blue_optimise.pdf", plot_clobber=True)


star = Star("/Users/arc/atmospheres/castelli-kurucz-2004/a???at*.dat", channels=[blue])
star.infer({"Teff": 5700., "logg": 4.0, "[M/H]": 0.1, "xi": 0.9}, walkers=200, burn=450, sample=50)




