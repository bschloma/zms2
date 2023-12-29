"""trace_analysis.py
Functions for analyzing traces. keep this minimal with the dependencies, i.e. just numpy, pandas, and matplotlib"""

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from numpy.polynomial.polynomial import Polynomial


def extract_traces(df, method='radial_dog'):
    """from spots DataFrame with nuclear ids, create a list containing a pair of time array and intensity array for
    each nucleus."""
    nucleus_ids = np.unique(df.nucleus_id)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    traces = []
    for n in range(len(nucleus_ids)):
        sub_df = df[df.nucleus_id == nucleus_ids[n]]
        t_arr = np.arange(np.min(sub_df.t), np.max(sub_df.t) + 1)
        inten_arr = np.zeros_like(t_arr)
        for i in range(len(inten_arr)):
            if np.isin(t_arr[i], sub_df.t):
                # if there's only one spot per time point, yay! assign that spot's inten to the inten array
                if np.sum(sub_df.t == t_arr[i]) == 1:
                    inten_arr[i] = sub_df[sub_df.t == t_arr[i]].get(method).values
                # if there are multiple spots per time point, pick the one with the highest level of classifier prob
                else:
                    sub_sub_df = sub_df[sub_df.t == t_arr[i]]
                    sub_sub_df = sub_sub_df[sub_sub_df.prob == np.nanmax(sub_sub_df.prob)]
                    if len(sub_sub_df) > 1:
                        # something bizarre is happening where prob is identical for multiple spots. In this case, pick
                        # the brightest spot.
                        sub_sub_df = sub_sub_df[sub_sub_df.get(method) == np.nanmax(sub_sub_df.get(method))]
                    if len(sub_sub_df) > 1:
                        # if the two spots have the same value (i.e. 0), store the first one
                        sub_sub_df = sub_sub_df.iloc[0]
                    elif len(sub_sub_df) == 0:
                        continue
                    #try:
                        # all cases seem to be covered, but just in case, keep this in a try block so analysis can
                        # proceed.
                    inten_arr[i] = sub_sub_df.get(method)
                    #except ValueError:
                    #    continue

        traces.append([t_arr, inten_arr, nucleus_ids[n]])

    return traces


def remove_blips(df, method='gauss3d_dog', min_number_of_points=3, interval=11):
    """filter traces by removing isolated blips, keeping only blocks of activity. blocks are defined by regions having
    at least min_number_of_points in a window of length=interval"""

    # for interval to be odd. makes window easier
    if np.mod(interval, 2) != 0:
        interval = interval + 1

    # create a true copy of the df to return, which we will edit
    filtered_df = df.copy()

    # remove background
    filtered_df = filtered_df.drop(df[df.nucleus_id <= 0].index)
    
    # extract traces
    traces = extract_traces(filtered_df, method=method)

    for trace in traces:
        t_arr, inten_arr, nucleus_id = trace
        
        """do the filtering. loop over each time point, define a window based on interval, and count number of data 
        points in interval"""
        wing = (interval - 1) / 2  # wing = half of window not including center point
        for j in range(len(t_arr)):
            # only look at actual data points (inten_arr > 0)
            if inten_arr[j] > 0:
                # most cases: window is totally contained in time array
                if t_arr[j] - wing >= 0 and t_arr[j] + wing <= np.max(t_arr):
                    this_window = np.where((t_arr[j] - wing <= t_arr) & (t_arr <= t_arr[j] + wing))
                # edge case: early times. window = first time point:t_arr[j] + wing
                elif t_arr[j] - wing < 0 and t_arr[j] + wing <= np.max(t_arr):
                    this_window = np.where((t_arr[0] <= t_arr) & (t_arr <= t_arr[j] + wing))
                # edge case: late times. window = t_arr[j] - wing:last time point
                elif t_arr[j] - wing >= 0 and t_arr[j] + wing > np.max(t_arr):
                    this_window = np.where((t_arr[j] - wing <= t_arr) & (t_arr <= t_arr[-1]))
                # weird case: interval is longer than whole time series. pick a smaller window
                else:
                    print('remove_blips: pick a smaller window')
                    return

                # collect intensities and count how many are > 0
                window_intens = inten_arr[this_window]
                num_points_in_interval = np.sum(window_intens > 0)
                
                # if less than minimum, set that intensity value to 0
                if num_points_in_interval < min_number_of_points:
                    sel = np.array(filtered_df.t == t_arr[j]) * np.array(filtered_df.nucleus_id == nucleus_id)
                    idx = filtered_df[sel].index.values[0]
                    filtered_df.at[idx, method] = 0.0

    return filtered_df


def enforce_1spot_per_nucleus(df, method='gauss3d_dog'):
    nucleus_ids = np.unique(df.nucleus_id)
    nucleus_ids = nucleus_ids[nucleus_ids > 0]
    culled_list = []
    for n in range(len(nucleus_ids)):
        sub_df = df[df.nucleus_id == nucleus_ids[n]]
        for t in sub_df.t.unique():
            sub_sub_df = sub_df[sub_df.t == t]
            if len(sub_sub_df) > 1:
            # if there are multiple spots per time point, pick the one with the highest level of classifier prob
                keep_indices = sub_sub_df[sub_sub_df.prob == np.nanmax(sub_sub_df.prob)].index.values
                if len(keep_indices) > 1:
                    # something bizarre is happening where prob is identical for multiple spots. In this case, pick
                    # the brightest spot.
                    sub_sub_df = sub_sub_df[keep_indices]
                    keep_indices = sub_sub_df[sub_sub_df.get(method) == np.nanmax(sub_sub_df.get(method))]
                    if len(keep_indices) > 1:
                        # if the two spots have the same value (i.e. 0), store the first one
                        keep_indices = keep_indices[0]
                elif len(keep_indices) == 0:
                    continue
            else:
                keep_indices = sub_sub_df.index
                
            assert len(keep_indices) == 1
            keep_index = keep_indices[0]
            culled_list.append(sub_sub_df.loc[keep_index].values)
    
    culled_df = pd.DataFrame(culled_list, columns=df.columns)
    
    return culled_df
    

def compute_moving_average(ms2, t_arr, window_size):
    # compute moving average via convolution
    moving_average = np.convolve(ms2, np.ones(window_size) / window_size, mode='same')

    return moving_average


def binarize_trace(ms2, t_arr, thresh, window_size=3):
    moving_average = compute_moving_average(ms2, t_arr, window_size)
    state = moving_average >= thresh

    return state


def get_on_and_off_times(state, t_arr):
    on_times = t_arr[np.where(np.diff(state.astype('int')) == 1)]
    off_times = t_arr[np.where(np.diff(state.astype('int')) == -1)]

    return on_times, off_times


def get_burst_durations(on_times, off_times):
    if len(on_times) == 0 or len(off_times) == 0:
        return []

    if on_times[0] < off_times[0]:
        if len(on_times) == len(off_times):
            burst_durations = off_times - on_times
        elif len(on_times) == len(off_times) + 1:
            burst_durations = off_times - on_times[:-1]
        else:
            raise ValueError
    elif on_times[0] > off_times[0]:
        if len(on_times) == len(off_times):
            burst_durations = off_times[1:] - on_times[:-1]
        elif len(on_times) == len(off_times) - 1:
            burst_durations = off_times[1:] - on_times
        else:
            raise ValueError
    else:
        raise ValueError

    return burst_durations


def get_burst_inactive_durations(on_times, off_times):
    if len(on_times) == 0 or len(off_times) == 0:
        return []

    if on_times[0] > off_times[0]:
        if len(on_times) == len(off_times):
            burst_inactive_durations = on_times - off_times
        elif len(off_times) == len(on_times) + 1:
            burst_inactive_durations = on_times - off_times[:-1]
        else:
            raise ValueError('first event is on --> off')
    elif on_times[0] < off_times[0]:
        if len(on_times) == len(off_times):
            burst_inactive_durations = on_times[1:] - off_times[:-1]
        elif len(on_times) == len(off_times) + 1:
            burst_inactive_durations = on_times[1:] - off_times
        else:
            raise ValueError('first event is off --> on')
    else:
        raise ValueError

    return burst_inactive_durations


def predict_protein(ms2, t_arr):
    # params - time in minutes
    transcription_rate = 33
    mrna_decay_rate = 0.23
    translation_rate = 4.5
    protein_decay_rate = 0.23
    fp_maturation_rate = 1.0
    fp_decay_rate = 0.1

    ms2 = ms2 / np.max(ms2)
    protein = np.zeros(len(t_arr))
    mrna = np.zeros(len(t_arr))
    fp = np.zeros(len(t_arr))

    for i in range(1, len(protein)):
            dt = t_arr[i] - t_arr[i - 1]
            mrna[i] = mrna[i - 1] + dt * (transcription_rate * ms2[i - 1] - mrna_decay_rate * mrna[i - 1])
            protein[i] = protein[i - 1] + dt * (translation_rate * mrna[i - 1] - protein_decay_rate * protein[i - 1])
            fp[i] = fp[i - 1] + dt * (fp_maturation_rate * protein[i - 1] - fp_decay_rate * fp[i - 1])
    
    return mrna, protein, fp


def predict_protein_v2(ms2, t_arr, Tmax, t_eval, 
                       transcription_rate = 33, 
                       mrna_decay_rate = 0.23, 
                       translation_rate = 4.5, 
                       protein_decay_rate = 0.23, 
                       fp_maturation_rate = 1.0, 
                       fp_decay_rate = 0.1):
    


    def mrna_production(t, ms2, t_arr, time_tol=0.3):
        if np.min(np.abs(t - t_arr)) < time_tol:
            production = ms2[np.abs(t - t_arr) == np.min(np.abs(t - t_arr))][0]
        else:
            production = 0
            
        return production
    
    def derivative(t, y, ms2, t_arr, transcription_rate, translation_rate, fp_maturation_rate, 
                   mrna_decay_rate, protein_decay_rate, fp_decay_rate, time_tol=0.3):
        mrna, protein, fp = y
    
        mrna_derivative = transcription_rate * mrna_production(t, ms2, t_arr, time_tol) - mrna_decay_rate *  mrna
        protein_derivative = translation_rate * mrna - protein_decay_rate * protein
        fp_derivative = fp_maturation_rate * protein - fp_decay_rate * fp
        
        return [mrna_derivative, protein_derivative, fp_derivative]
    
    sol = solve_ivp(derivative, [0, Tmax], [0, 0, 0], t_eval=t_eval, 
                    args=(ms2, t_arr, transcription_rate, translation_rate, 
                          fp_maturation_rate, mrna_decay_rate, protein_decay_rate, 
                          fp_decay_rate), max_step=0.25)
    
    mrna, protein, fp = sol.y
    
    return mrna, protein, fp

def predict_protein_for_all_nuclei(df, tracks, method='gauss3d_dog'):
    traces = extract_traces(df, method)
    protein_df = pd.DataFrame(columns=['fp', 'nucleus_id', 't', 'z', 'y', 'x'])

    for trace in traces:
        t_arr, ms2, nucleus_id = trace
        sub_tracks = tracks[tracks.track_id == nucleus_id]

        mrna, protein, fp = predict_protein_v2(ms2, t_arr, Tmax=sub_tracks.t.max(), t_eval=sub_tracks.t)
                 
        tmp_df = pd.DataFrame()
        tmp_df['fp'] = fp
        tmp_df['pred_protein'] = protein
        tmp_df['pred_mrna'] = mrna
        tmp_df['nucleus_id'] = nucleus_id
        tmp_df['t'] = sub_tracks.t.values
        tmp_df['z'] = sub_tracks.z.values
        tmp_df['y'] = sub_tracks.y.values
        tmp_df['x'] = sub_tracks.x.values

        protein_df = pd.concat((protein_df, tmp_df), axis=0)
    
    return protein_df
        
    
def compute_trace_uncertainty(t_arr, offset_arr, inten_arr, degree=4, false_positive=0.04, false_negative=0.08, n_pixels_per_spot=251):
    # poly fit
    coefs = Polynomial.fit(t_arr, offset_arr, deg=degree).convert().coef
    s = np.sqrt(np.mean((poly_eval(t_arr, coefs) - offset_arr) ** 2)) * n_pixels_per_spot * np.sqrt(1 + 1 / n_pixels_per_spot)    
    has_spot = np.array((inten_arr > 0), dtype='float32')

    # case 1: has spots
    trace_uncertainty_1 = has_spot * (np.sqrt(s ** 2 * (1 - false_positive) + inten_arr ** 2 * false_positive * (1 - false_positive)))
    
    # case 2: no spot
    trace_uncertainty_2 = (1 - has_spot) * np.mean(inten_arr[inten_arr > 0]) * np.sqrt(false_negative * (1 - false_negative))
    
    # combine them
    trace_uncertainty = trace_uncertainty_1 + trace_uncertainty_2
    
    # poly = PolynomialFeatures(degree=degree, include_bias=False)
    # poly_features = poly.fit_transform(t_arr.reshape([-1, 1]))
    # model = LinearRegression()

    # model.fit(poly_features, t_arr)
    
    # pred = model.predict(poly_features)
    
    # trace uncertainty is given by fluctuations around polynomial fit + a term that accounts for the probability of being a false negative.
    #trace_uncertainty = np.sqrt(2 * np.mean((pred - offset_arr) ** 2)) * np.ones_like(inten_arr) + np.mean(inten_arr) * false_negative * np.array(inten_arr > 0, dtype='uint8')
    
    
    return trace_uncertainty
    
    
def poly_eval(x, coefs):
    y = np.zeros_like(x)
    for i in range(len(y)):
        for n in range(len(coefs)):
            y[i] += coefs[n] * x[i] ** n
    
    return y
    
    
    
    
    
    
    
    