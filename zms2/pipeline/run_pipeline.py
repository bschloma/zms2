"""run the full spot pipeline"""
from zms2.spots.detection import run_spot_detection
from zms2.spots.classification import run_batch_prediction, run_batch_prediction_by_time_point
from zms2.spots.quantification import quantify_spots
from zms2.traces.trace_assembly import assign_nucleus, fill_in_traces
import pandas as pd
import numpy as np


def run_pipeline(path_to_raw_data,
                 steps,
                 save_dir=None,
                 timepoints=None,
                 sigma_blur=0.0,
                 sigma_dog_1=0.68,
                 spot_thresh=0.001,
                 skin_sigma_blur=5.74,
                 skin_thresh=10 ** -1.76,
                 erosion_size=0,
                 xor_size=0,
                 cpu_only=False,
                 path_to_segments=None,
                 path_to_model=None,
                 prob_thresh=0,
                 method=None,
                 single_cpu=False,
                 look_for_nearby_nuclei=True,
                 dxy=5,
                 dz=3,
                 fill_trace_thresh=None,
                 n_fill_iterations=1,
                 **kwargs
                 ):
    valid_steps = ['detection', 'classification', 'quantification', 'assign_nucleus', 'fill_in_traces']
    assert all([step in valid_steps for step in steps])

    if 'detection' in steps:
        path_to_spots = save_dir + '/spots_raw.pkl'
        df = run_spot_detection(path_to_raw_data, timepoints=timepoints, sigma_blur=sigma_blur,
                                skin_thresh=skin_thresh, erosion_size=erosion_size, xor_size=xor_size,
                                skin_sigma_blur=skin_sigma_blur,
                                sigma_dog_low=sigma_dog_1, spot_thresh=spot_thresh,
                                path_to_spots=path_to_spots, cpu_only=cpu_only)

    if 'classification' in steps:
        path_to_spots = save_dir + '/spots_raw.pkl'
        path_to_spots_culled = save_dir + '/spots_culled.pkl'
        df = pd.read_pickle(path_to_spots)
        # df = run_batch_prediction(df, path_to_model)
        df = run_batch_prediction_by_time_point(df, path_to_model)
        df.to_pickle(path_to_spots)
        df_culled = df[df.prob > prob_thresh]
        df_culled.to_pickle(path_to_spots_culled)

    if 'quantification' in steps:
        path_to_spots_culled = save_dir + '/spots_culled.pkl'
        path_to_spots_quant = save_dir + '/spots_quant.pkl'
        df = pd.read_pickle(path_to_spots_culled)
        df = quantify_spots(df, method=method, **kwargs)
        df.to_pickle(path_to_spots_quant)

    if 'assign_nucleus' in steps:
        path_to_spots_quant = save_dir + '/spots_quant.pkl'
        df = pd.read_pickle(path_to_spots_quant)
        df = assign_nucleus(df, path_to_segments, look_for_nearby_nuclei=look_for_nearby_nuclei, dxy=dxy, dz=dz,
                            single_cpu=single_cpu)
        df.to_pickle(path_to_spots_quant)

    if 'fill_in_traces' in steps:
        path_to_spots_quant = save_dir + '/spots_quant.pkl'
        path_to_spots_filled = save_dir + '/spots_filled.pkl'

        df = pd.read_pickle(path_to_spots_quant)

        # filter for only spots assigned to a nucleus
        df = df[~np.isnan(df.nucleus_id)]
        df = df[df.nucleus_id > 0]

        # call fill in traces
        for n in range(n_fill_iterations):
            # if n == 0:
            #     print(f'number of true spots in dataframe = {len(df)}')
            # else:
            #     print(f'number of true spots in dataframe = {len(df[df.passed_filters])}')
            df = fill_in_traces(df, thresh=fill_trace_thresh, path_to_zarr=path_to_raw_data,
                                path_to_model=path_to_model,
                                method=method)
        df.to_pickle(path_to_spots_filled)

    return df
