"""functions for spot detection"""

import numpy as np
import zarr
import dask.array as da
import pickle
from tqdm import tqdm
import cupy as cp
from cucim.skimage.filters import gaussian, difference_of_gaussians
import cucim.skimage.morphology as cpmorph
import cucim.skimage.measure as cpmeas
from skimage.measure import regionprops as regionprops_np
import pandas as pd
from pathlib import Path


def run_spot_detection(path_to_spot_data, timepoints=None, sigma_blur=5.74, skin_sigma_blur=10.0,
                       skin_thresh=10 ** -1.76,
                       erosion_size=0, xor_size=0, sigma_dog_low=0.68, spot_thresh=0.001,
                       path_to_spots=r'./spots.pkl', cpu_only=False):
    """ input path to zarr file and parameters, run spot detection, output a dataframe with spots info. parameters
        can be optimized with the tune_params.py functions in interactives.
        Sketch of algorithm:    -spot mask is made from a simple combination of blurring, DoG, and thresholding.
                                    results will be filtered later with a CNN classifier.
                                -large, bright background regions of mcp expression in the skin is removed with one
                                    of two algorithms: (1) blur + threshold, (2) geometrically, by masking the whole
                                    tissue and then converting to a thin strip of its boundary
                                -^^ these two steps are default done by passing each zslice individually to the gpu
                                    and back, with dask being used to orchestrate the rechunking and scheduling of
                                    transfers.
                                -left over regions are labelled and info is collected with straightforward numpy version
                                    of regionprops.
                                -results are stored in a pandas dataframe, with columns:
                                    -data:     3D numpy array of voxel containing potential spot
                                    -spot_id:  unique spot identifier that comes from regionprops
                                    -t:        scan number of spot
                                    -z, y, x:  centroid of spot in whole image array, from regionprops
                                    -manual_classification: can be set to bool (true spot vs false spot) with manual
                                                            classification, for use in training CNN classifier. created
                                                            here as None """

    """ preliminaries"""
    # check for CUDA device
    if not cpu_only:
        if cp.cuda.runtime.getDeviceCount() == 0:
            raise ValueError('cpu_only=False but no CUDA devices found. Re-run with cpu_only=True.')

    # create a folder for saving individual time point results. useful for both memory management and in case something
    # crashes.
    spots_by_time_point_dir = Path(path_to_spots).parent / 'spots_by_time_point'
    spots_by_time_point_dir.mkdir()

    # open zarr array of spot data. likely not in memory if dataset is large. dims=TZYX
    spot_data = zarr.open(path_to_spot_data, mode='r')

    # create output DataFrame where we will store spot info
    #df = pd.DataFrame(columns=['data', 'spot_id', 't', 'z', 'y', 'x', 'manual_classification'])
    #dtypes_dir = {'data': object, 'spot_id': np.uint32, 't': np.uint16, 'z': np.uint16, 'y': np.uint16, 'x': np.uint16}
    #df = df.astype(dtypes_dir)

    # if timepoints is not specified, run all timepoints in the spot data array.
    if timepoints is None:
        timepoints = range(spot_data.shape[0])

    """loop over time and run spot detection. default run uses gpu for initial spot mask creation."""
    for timepoint in tqdm(timepoints):
        # load one timepoint into a dask array
        data_da = da.from_array(spot_data[timepoint],
                                chunks=(1, spot_data.shape[2], spot_data.shape[3]))

        # create spot labels, either with gpu version (default) or cpu version. both return numpy array.
        if cpu_only:
            raise NotImplementedError
        else:
            labels = create_spot_labels_gpu(data_da, sigma_blur, skin_sigma_blur, skin_thresh, erosion_size, xor_size,
                                            sigma_dog_low, spot_thresh)

        # from label matrix extract spot data.
        props = regionprops_np(labels)
        im = data_da.compute()
        df_list = []
        for p in props:
            this_df = extract_spot_data(p, im=im, voxel_size=(9, 11, 11), timepoint=timepoint)
            df_list.append(this_df)


        # save results to a temporary dir
        if len(df_list) > 0:
            df = pd.concat(df_list, axis=0)
            dtypes_dir = {'data': object, 'spot_id': np.uint32, 't': np.uint16, 'z': np.uint16, 'y': np.uint16,
                          'x': np.uint16}
            df = df.astype(dtypes_dir)
            df.to_pickle(spots_by_time_point_dir / f'test_t{timepoint}.pkl')

    # read in all the individual spot dataframes and concat into one big one and save
    df_list = []
    for timepoint in timepoints:
        try:
            this_df = pd.read_pickle(spots_by_time_point_dir / f'test_t{timepoint}.pkl')
            df_list.append(this_df)
        except FileNotFoundError:
            continue

    df = pd.concat(df_list, ignore_index=True)
    dtypes_dir = {'data': object, 'spot_id': np.uint32, 't': np.uint16, 'z': np.uint16, 'y': np.uint16, 'x': np.uint16}
    df = df.astype(dtypes_dir)
    df.to_pickle(path_to_spots)

    return df


def create_spot_labels_gpu(data_da, sigma_blur=5.74, skin_sigma_blur=10.0, skin_thresh=10 ** -1.76,
                           erosion_size=0, xor_size=0, sigma_dog_low=0.68, spot_thresh=0.001):
    # execute gpu-backed spot mask function. return 3D cupy array.
    # note flexible structure: one function gpu_process can take a cupy-based function as "process" argument
    # and a list of params. this approach is used to create spot masks and skin masks
    spot_mask = gpu_process(data_da, params=[sigma_blur, sigma_dog_low, spot_thresh],
                            process=make_spot_mask).compute()

    # execute gpu-backed skin mask function. return 3D cupy array.
    skin_mask = gpu_process(data_da, params=[skin_sigma_blur, skin_thresh, erosion_size, xor_size],
                            process=make_skin_mask).compute()

    # apply skin mask to spot mask and create label matrix
    spot_mask = spot_mask * skin_mask
    labels = cpmeas.label(spot_mask)

    # release cupy arrays
    skin_mask = cp.asnumpy(skin_mask)
    spot_mask = cp.asnumpy(spot_mask)

    # numpy version of regionprops
    labels = cp.asnumpy(labels)

    return labels


def make_spot_mask(arr, params):
    # unpack params
    _sigma_blur = params[0]
    _sigma_low = params[1]
    _thresh = params[2]

    # do spot filtering and thresholding slice by slice, due to gpu memory constraints
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    if _sigma_blur > 0:
        arr = gaussian(arr, sigma=_sigma_blur)

    if _sigma_low > 0:
        arr = difference_of_gaussians(arr, low_sigma=_sigma_low)

    # create and apply a mask
    if _thresh > 0:
        arr = arr > _thresh

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def to_gpu(arr):
    return cp.asarray(arr)


def gpu_process(darr, params, process=make_spot_mask):
    alpha = 0.5
    # lazy move to gpu
    arr_cu = darr.map_blocks(to_gpu, dtype=np.float32)

    # lazy apply filter
    filt = da.map_blocks(process, arr_cu, params, dtype=np.float32)

    # return cupy array
    return filt


def make_skin_mask(arr, params):
    # unpack params
    _sigma_blur, _thresh, _erosion_size, _xor_size = params

    # do filtering and thresholding slice by slice, due to gpu memory constraints
    # unpack this zslice
    arr = arr[0]

    # apply some filters
    if _sigma_blur > 0:
        arr = gaussian(arr, sigma=_sigma_blur)

    # create and apply a mask
    if _thresh > 0:
        arr = arr > _thresh

    # erode mask to get back to boundary
    if _erosion_size > 0:
        arr = cpmorph.binary_erosion(arr, footprint=cpmorph.disk(_erosion_size))

    # then subtract off another erosion
    if _xor_size > 0:
        arr = cp.logical_xor(arr, cpmorph.binary_erosion(arr, footprint=cpmorph.disk(_xor_size)))

    # invert mask
    arr = cp.logical_not(arr)

    # reshape into 3D arr
    arr = cp.expand_dims(arr, axis=0)

    return arr


def extract_spot_data(these_props, im, voxel_size=(9, 11, 11), timepoint=0):
    """from an instance of regionprops + raw image, extract some properties"""
    locations = np.int16(these_props['centroid'])
    this_dict = dict()
    # only include spots whose voxel lies completely in the image
    if not check_voxel_boundary(im.shape, locations, voxel_size):
        z0, y0, x0 = locations
        z, y, x = voxel_size
        this_voxel = im[(z0 - (z - 1) // 2):(z0 + (z - 1) // 2 + 1), (y0 - (y - 1) // 2):(y0 + (y - 1) // 2) + 1,
                     (x0 - (x - 1) // 2):(x0 + (x - 1) // 2 + 1)]
        this_dict['data'] = [this_voxel]
        this_dict['spot_id'] = these_props['label']
        this_dict['t'] = timepoint
        this_dict['z'] = locations[0]
        this_dict['y'] = locations[1]
        this_dict['x'] = locations[2]
        this_dict['manual_classification'] = None

    this_df = pd.DataFrame.from_dict(this_dict)
    return this_df


def check_voxel_boundary(im_shape, location, voxel_size):
    """check whether the spot's voxel lives entirely within the image."""
    z0, y0, x0 = location
    z, y, x = voxel_size
    hit_boundary = ((z0 - (z - 1) // 2) <= 0
                    or (z0 + (z - 1) // 2 + 1) >= im_shape[0]
                    or (y0 - (y - 1) // 2) <= 0
                    or (y0 + (y - 1) // 2 + 1) >= im_shape[1]
                    or (x0 - (x - 1) // 2) <= 0
                    or (x0 + (x - 1) // 2 + 1) >= im_shape[2])

    return hit_boundary


def extract_locations_arr_from_spots(df):
    locations = np.zeros((len(df), 3))
    for i, row in enumerate(df.iterrows()):
        locations[i] = np.array([row.z, row.y, row.x])

    return locations


def extract_spot_voxels_from_zarr(path_to_zarr, locations, voxel_size=(9, 11, 11)):
    """pull out voxel of mcp channel data from zarr at 4D points specified by locations. alternative to
    extract_spot_data above, which uses numpy array. Using this for making manual traces. go straight to creating a
    dataframe"""

    # open a 4D zarr array (out of memory)
    data = zarr.open(path_to_zarr, 'r')

    # extract shape of each time point. used for checking if spot is near a boundary
    im_shape = data.shape[1:]

    # create a dataframe for storing the output
    df = pd.DataFrame()
    for i in range(len(locations)):
        if not check_voxel_boundary(im_shape, locations[i, 1:], voxel_size):
            t0, z0, y0, x0 = locations[i]
            t0 = int(t0)
            z0 = int(z0)
            y0 = int(y0)
            x0 = int(x0);
            z, y, x = voxel_size
            this_voxel = data[t0, (z0 - (z - 1) // 2):(z0 + (z - 1) // 2 + 1), (y0 - (y - 1) // 2):(y0 + (y - 1) // 2) + 1,
                         (x0 - (x - 1) // 2):(x0 + (x - 1) // 2 + 1)]
            if i == 0:
                df['data'] = [this_voxel]
            else:
                df.loc[len(df)] = [this_voxel]

    return df


def add_voxel_to_manual_spots_df(path_to_manual_spots_df, path_to_zarr):
    manual_spots_df = pd.read_pickle(path_to_manual_spots_df)
    locations = manual_spots_df.get(['t', 'z', 'y', 'x']).values
    data_df = extract_spot_voxels_from_zarr(path_to_zarr, locations)
    df = pd.concat((data_df, manual_spots_df), axis=1)
    df.to_pickle(path_to_manual_spots_df)

    return

