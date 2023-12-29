"""functions for assembling MS2 traces from quantified spots DataFrame and segmented nuclei"""

import numpy as np
import zarr
import pandas as pd
from zms2.spots.detection import extract_spot_voxels_from_zarr
from zms2.traces.trace_analysis import binarize_trace, extract_traces
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
from zms2.spots.quantification import quantify_spots
from zms2.spots.classification import run_batch_prediction


def assign_nucleus(df, path_to_segments_zarr, look_for_nearby_nuclei=True, dxy=5, dz=3, single_cpu=False):
    """for each spot in a DatFrame, assign it to a nucleus in a segmented label matrix stored as a zarr array.

    look_for_nearby_nuclei=True engages an algorithm for dealing with spots that fall right outside the segmented
    nucleus region. The algorithm looks for nuclei that fall with dx, dy, dz of the spot centroid.

    single_cpu=True bypass multiprocessor calls."""
    # load necessary files
    segments = zarr.open(path_to_segments_zarr, 'r')

    # create new column for nucleus id. delete existing column if it exists.
    if "nucleus_id" in df.keys().values:
        df = df.drop('nucleus_id', axis=1)
    # df["nucleus_id"] = -1
    # Reset indices of df in case they're duplicated
    df = df.reset_index(drop=True)

    # create a partial function call to the assign_nucleus subroutine with additional parameters passed
    partial_func = partial(assign_nucleus_1spot, segments=segments, look_for_nearby_nuclei=look_for_nearby_nuclei,
                           dz=dz, dxy=dxy)

    if single_cpu:
        tmp_df = df.assign(spot_location=df.loc[:, ['t', 'z', 'y', 'x']].values.tolist())
        result = tmp_df.spot_location.apply(partial_func)
    else:
        with Pool() as pool:
            result = pool.map(partial_func, tqdm(df.loc[:, ['t', 'z', 'y', 'x']].values))
    # result_df = pd.DataFrame(result, columns=['nucleus_id'])
    # df = pd.concat((df, result_df), axis=1)
    df['nucleus_id'] = result

    return df


def assign_nucleus_1spot(spot_location, segments, look_for_nearby_nuclei, dz, dxy):
    t, z, y, x = spot_location
    this_nuc_id = segments[t, z, y, x]
    dx = dxy
    dy = dxy
    if look_for_nearby_nuclei:
        if this_nuc_id > 0:
            return this_nuc_id

        else:
            # spot is not in a nuclear segment. it could be that the spot is on the edge and is just missed.
            # Look for nearby nuclei to assign the spot to.
            # get the nuclear ids of in the box
            try:
                these_seg_ids = segments[t, z - dz: z + dz + 1,
                                y - dy: y + dy + 1,
                                x - dx: x + dx + 1]
            except RuntimeError:
                print('error!')
                return np.NaN

            # it's possible that the spot is on the boarder of the dataset. in this rare case, just skip the spot.
            # this case will show up as these_seg_ids having the wrong shape
            deviation_from_correct_shape = np.array(these_seg_ids.shape) - np.array(
                [2 * dz + 1, 2 * dy + 1, 2 * dx + 1])
            if np.prod(deviation_from_correct_shape) == 0:
                return np.NaN

            # see how many options for nearby nuclei we have. proceed accordingly.
            possible_nuc_ids = np.unique(these_seg_ids)
            if len(possible_nuc_ids) == 1:
                # no nearby nuclei. double check that the spot was assigned to background (nucleus 0),
                # then assign this spot to NaN.
                if possible_nuc_ids != 0:
                    raise ValueError(
                        'something went wrong in assign_nucleus. if theres only one possible_nuc_id, it should be 0')
                return np.NaN
            elif len(possible_nuc_ids) == 2:
                # success! found 1 nearby nucleus. assigning spot to that nucleus.
                this_nuc_id = possible_nuc_ids[possible_nuc_ids > 0]
                return this_nuc_id
            else:
                # we found multiple nearby nuclei.
                # do a brute force search over all possible nearby nuclear ids and find the closest one
                non_zero_nuc_ids = possible_nuc_ids[possible_nuc_ids > 0]
                this_nuc_id = find_closest_nearby_nucleus(non_zero_nuc_ids, these_seg_ids, z, y, x, this_nuc_id)

                return this_nuc_id
    else:
        return this_nuc_id


def find_closest_nearby_nucleus(non_zero_nuc_ids, these_seg_ids, z, y, x, this_nuc_id):
    closest_distance = 1 / np.finfo(float).eps  # big number
    for i in range(len(non_zero_nuc_ids)):
        these_pixel_locs = np.asarray(these_seg_ids == non_zero_nuc_ids[i]).nonzero()

        # you could get multiple pixels of the same nucleus id. loop over all of them and find the closest pixel.
        for j in range(len(these_pixel_locs[0])):
            this_distance = np.sqrt((these_pixel_locs[0][j] - z) ** 2 + (these_pixel_locs[1][j] - y) ** 2
                                    + (these_pixel_locs[2][j] - x) ** 2)

            # check if this is the closest distance measured so far
            if this_distance < closest_distance:
                closest_distance = this_distance
                this_nuc_id = non_zero_nuc_ids[i]

    return this_nuc_id


def fill_in_traces(df, thresh, path_to_zarr, path_to_model, method='radial_dog'):
    """"TODO: working on avoiding iterating over spots multiple times during multiple iterations.
    Still have bugs:  tmp_df.loc[(np.array([max_sigma < 3.0]) * np.array(max_sigma > 0.5)).flatten(), 'passed_filters'] = False
    ValueError: cannot set a frame with no defined index and a scalar"""
    df.t = df.t.astype('float32')
    if 'fill_in_spot' not in df.keys():
        df['fill_in_spot'] = False
    if 'passed_filters' not in df.keys():
        df['passed_filters'] = True

    traces = extract_traces(df, method=method)
    nucleus_ids = np.unique(df.nucleus_id)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    assert len(nucleus_ids) == len(traces)
    tmp_df = pd.DataFrame(columns=['data', 'spot_id', 't', 'z', 'y', 'x', 'manual_classification'])
    tmp_df.t = tmp_df.t.astype('uint16')
    for i, trace in enumerate(traces):
        t_arr, inten_arr, _nuc_id = trace
        # inten_arr needs to be at least 3 time points---the length of the convolution window
        if len(inten_arr) < 3:
            continue
        sub_df = df[df.nucleus_id == nucleus_ids[i]]
        new_df = fill_in_trace(inten_arr, t_arr, thresh=thresh, path_to_zarr=path_to_zarr, sub_df=sub_df,
                               path_to_model=path_to_model, method=method)
        tmp_df = pd.concat((tmp_df, new_df), axis=0, ignore_index=True)

    if len(tmp_df) > 0:
        # run classification
        tmp_df = run_batch_prediction(tmp_df, path_to_model=path_to_model)

        # filter spots here
        #tmp_df = tmp_df[tmp_df.prob > 0.7]

        # run quantify spots
        tmp_df = quantify_spots(tmp_df, method=method)

        tmp_df['passed_filters'] = True
        if len(tmp_df) > 0:
            # filter by sigmas
            max_sigma = np.max(tmp_df[['sigma_x', 'sigma_y', 'sigma_z']], axis=1).values
            tmp_df.loc[np.array([max_sigma > 3.0]).flatten(), 'passed_filters'] = False
            tmp_df.loc[np.array([max_sigma < 0.5]).flatten(), 'passed_filters'] = False
            #tmp_df = tmp_df[(np.array([max_sigma < 3.0]) * np.array(max_sigma > 0.5)).flatten()]
            min_sigma = np.min(tmp_df[['sigma_x', 'sigma_y', 'sigma_z']], axis=1).values
            #tmp_df = tmp_df[np.array([min_sigma > 0.5]).flatten()]
            tmp_df.loc[np.array([min_sigma < 0.5]).flatten(), 'passed_filters'] = False

            # filter by zero intensity
            tmp_df.loc[tmp_df.get(method) == 0.0, 'passed_filters'] = False
            tmp_df.loc[tmp_df.get(method) == 0.0, method] = np.NaN
            #tmp_df = tmp_df[tmp_df.get(method) > 0]


    # create new spot ids
    tmp_df['spot_id'] = np.arange(df.spot_id.max() + 1, df.spot_id.max() + len(tmp_df) + 1)

    # label the new spots as fill in spots
    tmp_df['fill_in_spot'] = True

    # assemble into full df
    df = pd.concat((df, tmp_df), axis=0, ignore_index=True)

    return df


def fill_in_trace(inten_arr, t_arr, thresh, path_to_zarr, sub_df, path_to_model, method='radial_dog'):
    # binarize trace
    state = binarize_trace(inten_arr, t_arr, thresh)

    on_ids = np.where(state)[0]
    # ignore the first time point to make things simpler
    on_ids = on_ids[np.array(on_ids) > 0]
    new_df = pd.DataFrame(columns=['data', 't', 'z', 'y', 'x', 'nucleus_id'])

    # loop over on ids. if there is no detected spot (inten_arr == 0), try to pick the previous time point and use its
    # com to extract a spot voxel.
    tmp_df = None
    for i in range(len(on_ids)):
        if inten_arr[on_ids[i]] == 0:
            # check previous spot
            if inten_arr[on_ids[i] - 1] > 0:
                # only use neighboring spot if it passed all filters
                #if all(sub_df[sub_df.t == t_arr[on_ids[i] - 1]].passed_filters):
                #    tmp_df = get_adjacent_spot_data(sub_df, t_arr[on_ids[i]], t_arr[on_ids[i]] - 1, path_to_zarr)
                tmp_df = get_adjacent_spot_data(sub_df, t_arr[on_ids[i]], t_arr[on_ids[i]] - 1, path_to_zarr)

            # check next spot
            elif inten_arr[on_ids[i] + 1] > 0:
                #if all(sub_df[sub_df.t == t_arr[on_ids[i] + 1]].passed_filters):
                #    tmp_df = get_adjacent_spot_data(sub_df, t_arr[on_ids[i]], t_arr[on_ids[i]] + 1, path_to_zarr)
                tmp_df = get_adjacent_spot_data(sub_df, t_arr[on_ids[i]], t_arr[on_ids[i]] + 1, path_to_zarr)

            # if no adjacent spots, give it up for a lost cause
            else:
                continue
            if tmp_df is not None:
                new_df = pd.concat((new_df, tmp_df), axis=0, ignore_index=True)

    return new_df


def get_adjacent_spot_data(sub_df, current_time_point, adjacent_time_point, path_to_zarr):
    sub_sub_df = sub_df[sub_df.t == adjacent_time_point]
    if len(sub_sub_df) > 1:
        sub_sub_df = sub_sub_df[sub_sub_df.prob == np.max(sub_sub_df.prob)]
    z = sub_sub_df.z.values[0]
    y = sub_sub_df.y.values[0]
    x = sub_sub_df.x.values[0]

    locations = np.array([current_time_point, z, y, x])
    locations = np.expand_dims(locations, axis=0)
    tmp_df = extract_spot_voxels_from_zarr(path_to_zarr, locations)

    # pass time point
    tmp_df['t'] = current_time_point

    # compute centroid and use to update in new data frame
    zc, yc, xc = centroid(sub_sub_df.data.values[0])
    lz, ly, lx = sub_sub_df.data.values[0].shape
    new_z = np.uint16(z + zc - (lz - 1) / 2)
    new_y = np.uint16(y + yc - (ly - 1) / 2)
    new_x = np.uint16(x + xc - (lx - 1) / 2)
    tmp_df['z'] = new_z
    tmp_df['y'] = new_y
    tmp_df['x'] = new_x

    # pass nucleus id
    tmp_df['nucleus_id'] = sub_sub_df.nucleus_id.values[0]

    return tmp_df


def centroid(im):
    centroids = np.zeros(len(im.shape))
    indices = np.indices(im.shape)
    for i in range(len(indices)):
        these_indices = indices[i]
        centroids[i] = np.sum(im * these_indices) / np.sum(im)

    return centroids
