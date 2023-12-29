"""functions for quantifying spot intensity"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize, OptimizeResult
from skimage.filters import difference_of_gaussians
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from zms2.spots.quantification_utils import radial_center3d, gaussian3d_sym, gaussian3d_sym_log_likelihood, create_distance_grid
from zms2.spots.detection import extract_spot_voxels_from_zarr


def quantify_spots(df, method='total_intensity', single_core=False, **kwargs):
    """quantify fluorescence intensity of spots using various methods. single_core = bool, true bypasses
    multiprocessing"""

    if method == 'total_intensity':
        func = np.sum
        columns = ['total_intensity']

    elif method == 'gauss3d_dog':
        func = partial(gauss3d_dog, **kwargs)
        columns = ['gauss3d_dog', 'xc', 'yc', 'zc', 'sigma_x', 'sigma_y', 'sigma_z', 'amplitude', 'offset', 'std_offset']

    elif method == 'radial_dog':
        func = radial_dog
        columns = ['radial_dog', 'xc', 'yc', 'zc', 'sigma', 'offset', 'std_offset']

    else:
        raise ValueError('invalid method for quantify_spots')

    if single_core:
        res = df.data.apply(func)
    else:
        with Pool() as pool:
            # res = pool.imap_unordered(func, df.data.values.tolist())
            res = pool.map(func, tqdm(df.data.values.tolist()))
    res_df = pd.DataFrame(res, index=df.index, columns=columns)

    df = pd.concat((df, res_df), axis=1)

    # commenting out because of how fill in traces works now.
    # remove spots with inten=0...this can't be a real spot.
    #df = df[df.get(method) > 0]

    return df


def gauss3d_dog(voxel, background_method='shell', shell_radius_min=4, shell_radius_max=6, low_sigma=0.75,
                fit_method='lsq'):
    """quantify spot intensity by estimating the center of the spot with gaussian fitting, estimating local
    background levels, then summing pixel intensities above background in a sphere around the spot center.
    spot localization is done by default in a DoG-filtered image to remove structured background. """
    # fit 3D gaussian to the filtered data
    if fit_method == 'lsq':
        # preprocess the voxel by DoG filter (optional, pass low_sigma=0 to bypass) and normalization
        filtered_data = preprocess_spot_voxel(voxel, low_sigma)
        res = fit_gaussian3d(filtered_data)
    elif fit_method == 'mle':
        # maximum likelihood estimator with Poisson likelihood. don't filter.
        res = fit_gaussian3d_mle(voxel)
    else:
        raise ValueError('invalid fit_method')

    # collect best fit params
    p_dog = res.x
    xc, yc, zc, sigma_x, sigma_y, sigma_z, dog_amplitude, dog_offset = p_dog

    # compute grid of distances from spot center, for use in intensity computations.
    # TODO: include physical distances for z vs xy
    distance_grid = create_distance_grid(xc, yc, zc, voxel.shape)

    # extract the spot pixels and the pixels within the inner shell, closer to the spot center
    spot_pixels = voxel[distance_grid < shell_radius_min]

    # compute background ('offset') that will be subtracted from spot in quantification
    if background_method == 'shell':
        offset, std_offset = compute_spot_background_shell(voxel, distance_grid, shell_radius_min, shell_radius_max)
        amplitude = 0.0
    elif background_method == 'gauss':
        offset, amplitude = compute_spot_background_gauss(voxel, xc, yc, zc, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z)
        std_offset = np.NaN
    else:
        raise ValueError('invalid background_method')

    # subtract background from spot pixels
    spot_pixels = subtract_background(spot_pixels, offset)

    # sum up intensity
    intensity = np.sum(spot_pixels)

    return intensity, xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, offset, std_offset


def radial_dog(voxel, background_method='shell', shell_radius_min=4, shell_radius_max=6, low_sigma=0.75):
    # preprocess the voxel by DoG filter (optional, pass low_sigma=0 to bypass) and normalization
    filtered_data = preprocess_spot_voxel(voxel, low_sigma)

    # run radialcenter3D localization to get spot center and sigma estimate
    rc, sigma = radial_center3d(filtered_data)

    # collect best fit params
    xc, yc, zc = rc

    # compute grid of distances from spot center, for use in intensity computations.
    # TODO: include physical distances for z vs xy. Add amplitude to output
    distance_grid = create_distance_grid(xc, yc, zc, voxel.shape)

    # extract the spot pixels and the pixels within the inner shell, closer to the spot center
    spot_pixels = voxel[distance_grid < shell_radius_min]

    # compute background ('offset') that will be subtracted from spot in quantification
    if background_method == 'shell':
        offset, std_offset = compute_spot_background_shell(voxel, distance_grid, shell_radius_min, shell_radius_max)
    elif background_method == 'gauss':
        offset, amplitude = compute_spot_background_gauss(voxel, xc, yc, zc)
        std_offset = np.NaN
    else:
        raise ValueError('invalid background_method')

    # subtract background from spot pixels
    spot_pixels = subtract_background(spot_pixels, offset)

    intensity = np.sum(spot_pixels)

    return intensity, xc, yc, zc, sigma, offset, std_offset


def subtract_background(spot_pixels, offset):
    original_dtype = spot_pixels.dtype
    spot_pixels = spot_pixels.astype('float32')
    spot_pixels -= offset
    spot_pixels[spot_pixels < 0] = 0
    spot_pixels = spot_pixels.astype(original_dtype)

    return spot_pixels


def preprocess_spot_voxel(voxel, low_sigma=0.75):
    # apply dog filter and scale to 0-1
    if low_sigma > 0:
        filtered_data = difference_of_gaussians(voxel, low_sigma=low_sigma)
    else:
        filtered_data = voxel
    filtered_data = (filtered_data - np.min(filtered_data)) / (np.max(filtered_data) - np.min(filtered_data))

    return filtered_data


def compute_spot_background_shell(voxel, distance_grid, shell_radius_min=4, shell_radius_max=6):
    """estimate background via one of two methods."""
    shell_pixels = voxel[np.array((distance_grid >= shell_radius_min) & (distance_grid <= shell_radius_max))]

    # compute background as mean of shell pixels
    offset = np.mean(shell_pixels)

    # compute standard deviation of background to use in uncertainty estimates
    std_offset = np.std(shell_pixels)

    return offset, std_offset


def compute_spot_background_gauss(voxel, xc, yc, zc, sigma_x=None, sigma_y=None, sigma_z=None):
    # fit the original spot data with fixed center and sigma params to get offset and amplitude
    if None in [sigma_x, sigma_y, sigma_z]:
        res = fit_gaussian3d_offset_amp_sigmas(voxel, xc, yc, zc)
    else:
        res = fit_gaussian3d_offset_amp(voxel, xc, yc, zc, sigma_x, sigma_y, sigma_z)

    # collect fit params
    p = res.x
    offset = p[1]
    amplitude = p[0]

    return offset, amplitude


def fit_gaussian3d(data):
    """Returns (xc, yc, sigma_x, sigma_y, sigma_xy, amplitude, offset
        the gaussian parameters of a 3D distribution found by a fit"""
    params = get_initial_gaussian_param_estimates_3d_sym(data)

    # avoid value errors in the initial parameter estimates. if the initial guesses are negative or nan,
    # something went wrong, and we should skip this spot. Pass a fake result class with nans for parameters.
    if not check_if_initial_params_are_fine(params):
        result = OptimizeResult()
        result.x = np.full(8, np.nan)

        return result

    def error_function(p): return np.ravel(
        gaussian3d_sym(np.indices(data.shape)[2], np.indices(data.shape)[1], np.indices(data.shape)[0], p[0], p[1],
                       p[2], p[3], p[4], p[5], p[6], p[7]) - data)

    result = least_squares(error_function, params, bounds=(0, np.inf), method='trf')

    return result


def check_if_initial_params_are_fine(params):
    params_are_fine = True
    if np.sum(params <= 0) > 0:
        params_are_fine = False
    if np.sum(np.isnan(params)) > 0:
        params_are_fine = False

    return params_are_fine


def fit_gaussian3d_offset_amp(data, xc, yc, zc, sigma_x, sigma_y, sigma_z):
    """fit symmetric 3D gaussian with fixed center and width params to obtain offset and amplitude"""
    initial_offset = np.mean(data)
    initial_amplitude = np.std(data)
    params = [initial_amplitude, initial_offset]

    # define error function of residuals
    def error_function(p): return np.ravel(
        gaussian3d_sym(np.indices(data.shape)[2], np.indices(data.shape)[1],
                       np.indices(data.shape)[0], xc, yc, zc, sigma_x, sigma_y, sigma_z, p[0], p[1])
        - data)

    # call the optimizer
    result = least_squares(error_function, params, bounds=(0, np.inf), method='trf')

    return result


def fit_gaussian3d_offset_amp_sigmas(data, xc, yc, zc):
    """fit symmetric 3D gaussian with fixed center params to obtain sigmas, offset, and amplitude"""
    initial_offset = np.mean(data)
    initial_amplitude = np.std(data)
    initial_sigma = 1.0
    params = [initial_sigma, initial_sigma, initial_sigma, initial_amplitude, initial_offset]

    # define error function of residuals
    def error_function(p): return np.ravel(
        gaussian3d_sym(np.indices(data.shape)[2], np.indices(data.shape)[1],
                       np.indices(data.shape)[0], xc, yc, zc, p[0], p[1], p[2], p[3], p[4])
        - data)

    # call the optimizer
    result = least_squares(error_function, params, bounds=(0, np.inf), method='trf')

    return result


def fit_gaussian3d_mle(data):
    """"""
    params = get_initial_gaussian_param_estimates_3d_sym(data)
    bounds = ((0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf))
    options = {'eps': 0.1}
    result = minimize(gaussian3d_sym_log_likelihood, params, bounds=bounds, jac=None, options=options, args=data)

    return result


def get_initial_gaussian_param_estimates_3d_sym(data):
    offset = np.mean(data)
    amplitude = np.std(data)
    total = data.sum()
    # note: correcting the error in 2D version in terms of dimension. down=y.
    zgrid, ygrid, xgrid = np.indices(data.shape)
    xc = (xgrid * data).sum() / total
    yc = (ygrid * data).sum() / total
    zc = (zgrid * data).sum() / total
    # trying simpler estimates for now
    sigma_x = 1.0
    sigma_y = 1.0
    sigma_z = 2.0
    p = np.array([xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, offset])

    return p


def quantify_spot_uncertainties_from_voxel_position(df, method, path_to_zarr):
    func = partial(quantify_spot_uncertainties_from_voxel_position_1row, method=method, path_to_zarr=path_to_zarr)
    result = df.apply(func, axis=1)

    return result


def quantify_spot_uncertainties_from_voxel_position_1row(row, method, path_to_zarr):
    location = row.get(['t', 'z', 'y', 'x'])
    shifted_df = assemble_shifted_spot_ensemble(path_to_zarr, location)
    shifted_df = quantify_spots(shifted_df, method=method)

    mean = np.mean(shifted_df.get(method).values)
    std_dev = np.nanstd(shifted_df.get(method).values)
    series = pd.Series([mean, std_dev])

    return series


def assemble_shifted_spot_ensemble(path_to_zarr, location):
    location = np.array(location)
    shifted_locations = shift_locations(location)
    df = extract_spot_voxels_from_zarr(path_to_zarr, shifted_locations)

    return df


def shift_locations(location):
    # tzyx
    shifted_locations = np.repeat(np.expand_dims(location, axis=0), 7, axis=0)
    shifted_locations[1, 1] += 1    # z + 1
    shifted_locations[2, 1] -= 1    # z - 1
    shifted_locations[3, 2] += 1    # y + 1
    shifted_locations[4, 2] -= 1    # y - 1
    shifted_locations[5, 3] += 1    # x + 1
    shifted_locations[6, 3] -= 1    # x - 1

    return shifted_locations


if __name__ == "__main__":
    # path_to_segments_zarr = r'/media/brandon/Data1/Data/Somitogenesis/Lightsheet Z1/2021-11-11/male_ms2_3244_female_mcp_med_3231_histone/embryo_1/tracking_cluster/segments.zarr'
    # path_to_spots_df = r'/media/brandon/Data1/Data/Somitogenesis/Lightsheet Z1/2021-11-11/male_ms2_3244_female_mcp_med_3231_histone/embryo_1/tracking_cluster/spots_dilation.pkl'
    # df = pd.read_pickle(r'/media/brandon/Data1/Data/Somitogenesis/Lightsheet Z1/2021-11-11/male_ms2_3244_female_mcp_med_3231_histone/embryo_1/tracking_cluster/spots_dilation_curated.pkl')
    df = pd.read_pickle(r'/media/brandon/Data1/Somitogenesis/Dorado/spots_curated_gauss_vol.pkl')
    # df = pd.read_pickle(r'/media/brandon/Data1/Somitogenesis/Dorado/manual_spot_df_16253_quant.pkl')

    df = df.drop('gauss_params', axis=1)
    # df = quantify_spots(df, method='gauss3D_dog')
    df = quantify_spots(df, method='radial_dog_mult')

    # df = assign_nucleus(path_to_spots_df, path_to_segments_zarr)
    # df = quantify_spots(df)

    df.to_pickle(r'/media/brandon/Data1/Somitogenesis/Dorado/spots_curated_radial_dog_mult.pkl')
