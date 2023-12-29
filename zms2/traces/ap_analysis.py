#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:28:32 2022

@author: brandon
"""
import numpy as np
from functools import partial
import pandas as pd


def get_ap(loc, ap):
    """loc = (t,z,y,x) of interest, ap = ap DF"""
    t = loc[0]
    zyx = loc[1:]
    these_ap_locs = ap[ap.t == t].values[:,1:]
    distances = distance(zyx, these_ap_locs)
    ap_value = np.argwhere(distances == np.nanmin(distances))
    if len(ap_value) > 1:
        ap_value = ap_value[0]
    
    if len(ap_value) == 0:
        print(distances)
    
    return ap_value
    

def distance(x, y):
    """ x = 1xd, y = Nxd"""
    return np.sqrt(np.sum((x - y) ** 2, axis=1))


def get_ap_for_nucleus(nucleus, tracks, ap):
    """ nucleus = int, tracks = tracks DF for just this nucleus, ap = ap DF"""
    t_values = tracks.t.values
    ap_values = np.zeros(len(t_values))
    for t in range(len(t_values)):
        sub_tracks = tracks[tracks.t == t_values[t]]
        this_loc = (t_values[t], sub_tracks.z.values[0], sub_tracks.y.values[0], sub_tracks.x.values[0])
        
        ap_values[t] = get_ap(this_loc, ap)
        
    return ap_values


def get_ap_for_spots(df, ap):
    """df = spots df, ap = ap df"""
    aps = np.zeros(len(df))
    locs = df.loc[:, ['t', 'z', 'y', 'x']].values.tolist()
    for i in range(len(locs)):
        aps[i] = get_ap(locs[i], ap)
    
    df['ap'] = aps
    
    return df


def get_ap_for_somites(somites, ap):
    """somites = somite df, ap = ap df"""
    aps = np.zeros(len(somites))
    for i in range(len(somites)):
        loc = somites.iloc[i].values
        aps[i] = get_ap(loc, ap)

    somites['ap'] = aps
    
    return somites


def filter_spots_by_distance_from_somite(df, somites, distance_thresh=-20):
    """need df.aps"""
    time_points = np.unique(df.t)
    sel = np.zeros(len(df))
    drop_indices = []    
    for t in time_points:
        this_somite_ap = np.unique(somites[np.abs(somites.t - t) 
                                 == np.min(np.abs(somites.t - t))].ap.values)[0]
        sub_df = df[df.t==t]
        these_indices = sub_df[(sub_df.ap - this_somite_ap) < distance_thresh].index
        drop_indices.append(these_indices)
        
    flat_list = [item for sublist in drop_indices for item in sublist]
    df = df.drop(flat_list)

    return df
        

def filter_spots_by_perp_distance_from_ap_axis(df, ap, dxy=0.4, dz=2, distance_thresh=100):
    """compute distance ap axis and throw out spots that are farther away"""
    partial_func = partial(compute_distance_to_ap_1row, ap=ap, dxy=dxy, dz=dz)
    dists = df.apply(partial_func, axis=1)
    df = df[dists < distance_thresh]
    
    return df, dists
        

def compute_distance_to_ap_1row(sub_df, ap, dxy=0.4, dz=2):
    loc = sub_df.get(['t', 'z', 'y', 'x']).values.tolist()
    t = loc[0]
    zyx = loc[1:]
    zyx = np.array([dz, dxy, dxy]) * np.array(zyx)
    these_ap_locs = ap[ap.t == t].values[:,1:]
    these_ap_locs = np.array([dz, dxy, dxy]) * np.array(these_ap_locs)
    distances = distance(zyx, these_ap_locs) 
    min_distance = np.min(distances)

    return min_distance
       

def bin_aps(df, bins):
    _counts, bins = np.histogram(df.ap, bins)
    bins = bins[1:]
    partial_func = partial(get_ap_bin, bins=bins)
    binned_aps = df.ap.apply(partial_func)
    df['binned_ap'] = binned_aps.values
    
    return df
    

def get_ap_bin(this_ap, bins):
    this_bin = np.where(np.abs(this_ap - bins) == np.nanmin(np.abs(this_ap - bins)))[0][0]

    return this_bin
    
    

def get_microns_between_ap_bins(ap, dz=2, dyx=0.4):
    t_arr = np.unique(ap.t)
    microns = np.zeros((len(t_arr), 100))   # assume 100 ap bins
    for j, t in enumerate(t_arr):
        zyx = np.array(ap[ap.t==t].get(['z', 'y', 'x']))
        for i in range(1, len(zyx)):
            this_point = zyx[i]
            previous_point = zyx[i-1]
            this_point = np.array([dz, dyx, dyx]) * np.array(this_point)
            previous_point = np.array([dz, dyx, dyx]) * np.array(previous_point)
            microns[j, i] = np.sqrt(np.sum((this_point - previous_point) ** 2))
        
    return microns    
    

def get_ap_um(df, ap, dz=2, dyx=0.4):
    """for each spot in df, which has been assigned an ap bin, add another 
    column, ap_um, that is the actual distance in microns along the ap axis.
    can also use for somites. just need t and ap."""
    # get microns along ap axis at each time point. 
    # microns.shape = (num time points, num ap bins)
    microns = np.cumsum(get_microns_between_ap_bins(ap, dz=dz, dyx=dyx), axis=1)
    
    def get_um_1row(row):
        t = int(row.t)
        ap = int(row.ap)
        return microns[t, ap]
        
    df['ap_um'] = df.get(['t', 'ap']).apply(get_um_1row, axis=1).values
    
    return df
    

def bin_aps_um(df, bins):
    _counts, bins = np.histogram(df.ap_um, bins)
    bins = bins[1:]
    partial_func = partial(get_ap_bin, bins=bins)
    binned_aps = df.ap_um.apply(partial_func)
    df['binned_ap'] = binned_aps.values
    
    return df

    