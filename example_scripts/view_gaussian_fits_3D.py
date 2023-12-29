import napari
import numpy as np
import pandas as pd
from zms2.spots.quantification_utils import gaussian3d_sym, gaussian3d_sym_bg

# load some spots. here it's from a manual trace.
df = pd.read_pickle(r'/media/brandon/Data1/Somitogenesis/Dorado/gauss_001_v2_sigma_filt/filtered_df.pkl')
df = df[df.nucleus_id == 3187]
df = df.sort_values(by='t')

# create one array with all the voxels in it and one with all the gaussian fits
all_voxels = np.zeros((len(df),) + df.iloc[0].data.shape)
all_shells = np.zeros((len(df),) + df.iloc[0].data.shape)
all_gauss_fits = np.zeros((len(df),) + df.iloc[0].data.shape)

xarr = np.arange(all_gauss_fits[0].shape[2])
yarr = np.arange(all_gauss_fits[0].shape[1])
zarr = np.arange(all_gauss_fits[0].shape[0])

zgrid, ygrid, xgrid = np.meshgrid(zarr, yarr, xarr, indexing='ij')

for i in range(len(df)):
    this_data = df.iloc[i].data
    all_voxels[i] = this_data

    # gaussian fit
    this_df = df.iloc[i]
    this_fit = gaussian3d_sym(xgrid, ygrid, zgrid, this_df.xc, this_df.yc, this_df.zc, this_df.sigma_x, this_df.sigma_y,
                              this_df.sigma_z, 3000, this_df.offset)
    all_gauss_fits[i] = this_fit

    xc_ind = np.round(this_df.xc).astype(int)
    yc_ind = np.round(this_df.yc).astype(int)
    zc_ind = np.round(this_df.zc).astype(int)

    zgrid, ygrid, xgrid = np.indices(this_data.shape)
    delta_zgrid = zgrid - zc_ind
    delta_ygrid = ygrid - yc_ind
    delta_xgrid = xgrid - xc_ind

    # compute grid of distances from spot center.
    distance_grid = np.sqrt(delta_xgrid ** 2 + delta_ygrid ** 2 + delta_zgrid ** 2)
    shell_pixels = this_data.copy()
    shell_pixels[np.array((distance_grid < 3) | (distance_grid > 4))] = 0
    all_shells[i] = shell_pixels

viewer = napari.view_image(all_voxels)
viewer.add_image(all_gauss_fits)
viewer.add_image(all_shells)



