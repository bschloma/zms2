import napari
import zarr
import numpy as np
from scipy.interpolate import splprep, splev
import pandas as pd

# degree of spline
degree = 2

# path for final interpolated points dataframe
path_to_interpolated_points_df = r'/media/brandon/Data1/Somitogenesis/Dorado/ap.pkl'

# raw data. use histone marker for midline
path_to_ds = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.zarr'
root = zarr.open(path_to_ds, 'r')
h2b = root['H2b']['H2b']
num_time_points = h2b.shape[0]
time_array = np.arange(0, num_time_points)

# for now, load in manually placed points
path_to_points = r'/media/brandon/Data1/Somitogenesis/Dorado/ap_raw.pkl'
df = pd.read_pickle(path_to_points)
labelled_time_points = np.unique(df.t)
num_labelled_time_points = len(labelled_time_points)
points = df.values

# ap bins
num_bins = 100
bins = np.linspace(0, 1, num_bins)

# interpolate points with B spline
# array to store interpolated points
labelled_interpolated_points_array = np.zeros((num_labelled_time_points * num_bins, 4))

# interp each time point separately first
counter = 0
for t in range(len(labelled_time_points)):
    these_points = df[df.t == labelled_time_points[t]].values[:, 1:]
    tck, u = splprep(these_points.T, k=degree)

    # interpolate at bins
    interpolated_points = splev(bins, tck)

    # collect back into an array format that napari likes
    for i in range(num_bins):
        labelled_interpolated_points_array[counter, 0] = labelled_time_points[t]
        labelled_interpolated_points_array[counter, 1] = interpolated_points[0][i]
        labelled_interpolated_points_array[counter, 2] = interpolated_points[1][i]
        labelled_interpolated_points_array[counter, 3] = interpolated_points[2][i]
        counter += 1

# now interpolated over time for each bin independently
interpolated_points_array = np.zeros((num_bins * num_time_points, 4))
counter = 0
for i in range(num_bins):
    these_points = labelled_interpolated_points_array[i:(num_bins * num_time_points):num_bins, 1:]

    # use time iteslf as the parameter
    u = (labelled_time_points - np.min(labelled_time_points)) / (np.max(labelled_time_points) - np.min(labelled_time_points))
    tck, u = splprep(these_points.T, u=u, k=1)

    u_eval = (time_array - np.min(time_array)) / (np.max(time_array - np.min(time_array)))
    these_interpolated_points = splev(u_eval, tck)

    # loop over time and extract points into properly shaped array
    for j in range(num_time_points):
        interpolated_points_array[counter, 0] = time_array[j]
        interpolated_points_array[counter, 1] = these_interpolated_points[0][j]
        interpolated_points_array[counter, 2] = these_interpolated_points[1][j]
        interpolated_points_array[counter, 3] = these_interpolated_points[2][j]
        counter += 1


interpolated_points_array = interpolated_points_array.astype('uint16')
interpolated_points_df = pd.DataFrame(interpolated_points_array, columns=['t', 'z', 'y', 'x'])
interpolated_points_df.to_pickle(path_to_interpolated_points_df)

# launch napari
viewer = napari.view_image(h2b)
viewer.add_points(points, size=36)
viewer.add_points(interpolated_points_array, size=28, face_color='b')
