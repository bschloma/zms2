import numpy as np
import pandas as pd
from scipy.signal import convolve2d


def gaussian3d_sym(x, y, z, xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, offset):
    """3D symmetric gaussian"""
    g = offset + amplitude * np.exp(-(x - xc) ** 2 / 2 / sigma_x / sigma_x - (y - yc) ** 2 / 2 / sigma_y / sigma_y - (
            z - zc) ** 2 / 2 / sigma_z / sigma_z)

    return g


def gaussian3d_sym_bg(x, y, z, xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, bg_amplitude, kx, ky, kz, offset):
    """3D symmetric gaussian on exponential background"""
    g = offset + amplitude * np.exp(-(x - xc) ** 2 / 2 / sigma_x / sigma_x - (y - yc) ** 2 / 2 / sigma_y / sigma_y - (
            z - zc) ** 2 / 2 / sigma_z / sigma_z) + bg_amplitude * np.exp(-(x * kx + y * ky + z * kz))

    return g


def gaussian3d_sym_log_likelihood(p, data):
    """log likelihood of 3D symmetric gaussian with Poisson noise"""
    g = gaussian3d_sym(np.indices(data.shape)[2], np.indices(data.shape)[1], np.indices(data.shape)[0], p[0], p[1],
                       p[2], p[3], p[4], p[5], p[6], p[7])
    g = np.ravel(g)
    data = np.ravel(data)
    pixel_log_likelihood = data * np.log(g) - g
    log_likelihood = -1 * np.sum(pixel_log_likelihood)

    return log_likelihood


def generate_synthetic_spots(num_spots, voxel_shape, amplitude=400, offset=200, bg_amplitude=0, sigma_x=1.0, sigma_y=1.0, sigma_z=2.0, shell_radius_min=4, shell_radius_max=6, return_background=False):
    """create fake spots, consisting of a gaussian spot, optional exponential background, and Poisson noise"""
    if type(num_spots) != int:
        num_spots = int(num_spots)
    # get dimensions of voxels
    lz, ly, lx = voxel_shape

    # preallocate array for fake spots
    fake_spots_arr = np.zeros((num_spots, lz, ly, lx))

    # set centers of fake spots to be at the center of the voxel plus gaussian noise
    z0 = (lz - 1) / 2
    y0 = (ly - 1) / 2
    x0 = (lx - 1) / 2
    center_std = 1.0
    center_offsets = np.random.normal(loc=0, scale=center_std, size=(num_spots, 3))
    true_centers = np.repeat(np.expand_dims(np.array([z0, y0, x0]), axis=0), repeats=num_spots, axis=0) + center_offsets

    # create grid of points for creating gaussians
    z_arr = np.arange(lz)
    y_arr = np.arange(ly)
    x_arr = np.arange(lx)
    zgrid, ygrid, xgrid = np.meshgrid(z_arr, y_arr, x_arr, indexing='ij')

    # only used if return_background is True
    shell_vars = np.zeros(num_spots)
    shell_means = np.zeros(num_spots)
    true_intens = np.zeros(num_spots)
    for i in range(num_spots):
        this_fake_spot = generate_synthetic_spot(true_centers[i], zgrid, ygrid, xgrid, amplitude, bg_amplitude,
                                                 offset, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z)
        fake_spots_arr[i] = this_fake_spot

        if return_background:
            # get ground truth background
            zc, yc, xc = true_centers[i]
            distance_grid = create_distance_grid(xc, yc, zc, this_fake_spot.shape)
            shell_pixels = this_fake_spot[np.array((distance_grid >= shell_radius_min) & (distance_grid <= shell_radius_max))]
            shell_vars[i] = np.var(shell_pixels)
            shell_means[i] = np.mean(shell_pixels)

            noiseless_spot = gaussian3d_sym(xgrid, ygrid, zgrid, xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, offset=0)
            true_intens[i] = np.sum(noiseless_spot[distance_grid < shell_radius_min])

    if return_background:
        return fake_spots_arr, shell_means, shell_vars, true_intens
    else:
        return fake_spots_arr


def generate_synthetic_spot(true_center, zgrid, ygrid, xgrid, amplitude, bg_amplitude, offset, sigma_x=1.0, sigma_y=1.0,
                            sigma_z=2.0):
    """ note the lack of 's'. create a fake spot assuming a gaussian psf, exponential background, and poisson noise"""
    # unpack true center
    zc, yc, xc = true_center

    # create the true gaussian spot
    g = gaussian3d_sym(xgrid, ygrid, zgrid, xc, yc, zc, sigma_x, sigma_y, sigma_z, amplitude, offset)

    # random background: exponential with random direction but fixed length scale
    # magnitude of wavevector for decay
    if bg_amplitude > 0:
        k = 0.1
        kx = np.random.normal(loc=0, scale=1 / np.max(xgrid))
        ky = np.random.normal(loc=0, scale=1 / np.max(ygrid))
        kz2 = k ** 2 - kx ** 2 - ky ** 2 > 0
        if kz2 > 0:
            kz = np.sqrt(kz2)
        # if the random draws don't satisfy the wavevector length, just make the gradient in the x direction
        else:
            kz = 0
            ky = 0
            kx = k

        # add variable background
        g += bg_amplitude * np.exp(-xgrid * kx - ygrid * ky - zgrid * kz)

    # add poisson noise
    fake_spot = np.random.poisson(g, size=g.shape)

    return fake_spot


def create_synthetic_spot_df(fake_spots):
    """convert an array of fake spots into a DataFrame, to be analyzed like real data."""
    df = pd.DataFrame()
    df['data'] = pd.Series(dtype=object)
    for i in range(len(fake_spots)):
        # tmp = pd.Series([fake_spots[i]], name='data', dtype=object)
        df.loc[len(df.index)] = [fake_spots[i]]

    df['nucleus_id'] = 1  # fake nuc id

    return df


def radial_center3d(I, zscaleratio=1.0):
    """Python implementation of Raghu's radialcenter algorithm in 3D. See his particle tracking page.
    # Ported from Matlab code. I kept variable names the same for consistency."""
    # swap axes to go from our convention (zyx) to Raghu's convention (yxz)
    I = np.swapaxes(I, 0, 1)
    I = np.swapaxes(I, 1, 2)

    # number of grid points
    Ny, Nx, Nz = I.shape

    # grid coordinates are -n:n, where Nx (or Ny) = 2*n+1
    # grid midpoint coordinates are -n+0.5:n-0.5;
    # note that in radialcenter.m I avoid repmat for speed; here I'll use it
    # for the z-repetition, for simplicity.
    xm_onerow = np.expand_dims(np.arange(-(Nx - 1) / 2.0 + 0.5, (Nx - 1) / 2.0 - 0.5 + 1), axis=0)
    xm2D = np.repeat(xm_onerow, Ny - 1, axis=0)
    xm2D = np.expand_dims(xm2D, axis=2)
    xm = np.repeat(xm2D, Nz - 1, axis=2)

    ym_onecol = np.expand_dims(np.arange(-(Ny - 1) / 2.0 + 0.5, (Ny - 1) / 2.0 - 0.5 + 1), axis=1)
    ym2D = np.repeat(ym_onecol, Nx - 1, axis=1)
    ym2D = np.expand_dims(ym2D, axis=2)
    ym = np.repeat(ym2D, Nz - 1, axis=2)

    # in this block fixed bug in Raghus code that swapped x and y dims.
    zm_onedepth = np.arange(-(Nz - 1) / 2.0 + 0.5, (Nz - 1) / 2.0 - 0.5 + 1)
    zm_onedepth = np.expand_dims(zm_onedepth, axis=0)
    zm = np.repeat(zm_onedepth, Nx - 1, axis=0)
    zm = np.expand_dims(zm, axis=0)
    zm = np.repeat(zm, Ny - 1, axis=0)

    # % For each slice, calculate the gradient in the x-y plane
    # % Nx-1 x Ny-1 arrays: dImag2D2, dI2Dx, dI2Dy
    dI2Dx = np.zeros((Ny - 1, Nx - 1, Nz))
    dI2Dy = np.zeros((Ny - 1, Nx - 1, Nz))
    for j in range(Nz):
        # % Calculate derivatives along 45-degree shifted coordinates (u and v)
        # % Note that y increases "downward" (increasing row number) -- we'll deal
        # % with this when calculating "m" below.
        dIdu = I[:Ny - 1, 1:Nx, j] - I[1:Ny, :Nx - 1, j]
        dIdv = I[:Ny - 1, :Nx - 1, j] - I[1:Ny, 1:Nx, j]

        # % Smoothing --
        h = np.ones((3, 3)) / 9  # % simple 3x3 averaging filter
        fdu = convolve2d(dIdu, h, 'same')
        fdv = convolve2d(dIdv, h, 'same')
        # % Note that we need a 45 degree rotation of
        # % the u,v components to express the slope in the x-y coordinate system.
        # % The negative sign "flips" the array to account for y increasing
        # % "downward"
        dI2Dx[:, :, j] = fdu - fdv
        dI2Dy[:, :, j] = -(fdv + fdu)

    # % For each pair of slices, calculate the gradient in z, and the average of
    # % the xy-plane gradients
    dIx = np.zeros((Ny - 1, Nx - 1, Nz - 1))
    dIy = np.zeros((Ny - 1, Nx - 1, Nz - 1))
    dIz = np.zeros((Ny - 1, Nx - 1, Nz - 1))
    for j in range(Nz - 1):
        dIx[:, :, j] = 0.5 * (dI2Dx[:, :, j] + dI2Dx[:, :, j + 1])
        dIy[:, :, j] = 0.5 * (dI2Dy[:, :, j] + dI2Dy[:, :, j + 1])
        dIz[:, :, j] = (1 / zscaleratio) * 0.25 * (I[:Ny - 1, :Nx - 1, j + 1] - I[:Ny - 1, :Nx - 1, j] +
                                                   I[:Ny - 1, 1:Nx, j + 1] - I[:Ny - 1, 1:Nx, j] +
                                                   I[1:Ny, :Nx - 1, j + 1] - I[1:Ny, :Nx - 1, j] +
                                                   I[1:Ny, 1:Nx, j + 1] - I[1:Ny, 1:Nx, j])

    dImag2 = dIx * dIx + dIy * dIy + dIz * dIz
    sqdImag2 = np.sqrt(dImag2)
    mx = dIx / sqdImag2  # % x-component of unit vector in the gradient direction
    my = dIy / sqdImag2  # % x-component of unit vector in the gradient direction
    mz = dIz / sqdImag2  # % x-component of unit vector in the gradient direction

    # % Weighting: weight by square of gradient magnitude and inverse
    # % distance to gradient intensity centroid.
    sdI2 = np.sum(dImag2)
    prodx = dImag2 * xm
    prody = dImag2 * ym
    prodz = dImag2 * zm
    xcentroid = np.sum(prodx) / sdI2
    ycentroid = np.sum(prody) / sdI2
    zcentroid = np.sum(prodz) / sdI2
    w = dImag2 / np.sqrt(
        (xm - xcentroid) * (xm - xcentroid) + (ym - ycentroid) * (ym - ycentroid) + (zm - zcentroid) * (zm - zcentroid))

    # % least-squares minimization to determine the translated coordinate
    # % system origin (xc, yc, zc) such that gradient lines have
    # % the minimal total distance^2 to the origin:
    rc = ls_radial_center_fit3d(xm, ym, zm, mx, my, mz, w)

    # %%
    # % Return output relative to upper left coordinate, slice 1
    rc[0] = rc[0] + (Nx + 1) / 2.0
    rc[1] = rc[1] + (Ny + 1) / 2.0
    rc[2] = rc[2] + (Nz + 1) / 2.0

    # % A rough measure of the particle width.
    # % Not at all connected to center determination, but may be useful for tracking applications;
    # % could eliminate for (very slightly) greater speed
    Isub = I - np.min(I)
    py, px, pz = np.indices((Ny, Nx, Nz))

    xoffset = px - rc[0]
    yoffset = py - rc[1]
    zoffset = pz - rc[2]
    r2 = xoffset * xoffset + yoffset * yoffset + zoffset * zoffset
    sigma = np.sqrt(np.sum(Isub * r2) / np.sum(Isub)) / 2  # % second moment is 2*Gaussian width

    return rc, sigma


def ls_radial_center_fit3d(xm, ym, zm, mx, my, mz, wk):
    # % least squares solution to determine the radial symmetry center

    # % inputs mx, my, mz, w are defined on a grid
    # % w are the weights for each point
    mz2pmy2w = (mz * mz + my * my) * wk
    mx2pmz2w = (mx * mx + mz * mz) * wk
    my2pmx2w = (my * my + mx * mx) * wk
    mxyw = mx * my * wk
    mxzw = mx * mz * wk
    myzw = my * mz * wk
    A = np.array([[np.sum(mz2pmy2w), -np.sum(mxyw), -np.sum(mxzw)],
                  [-np.sum(mxyw), np.sum(mx2pmz2w), -np.sum(myzw)],
                  [-np.sum(mxzw), -np.sum(myzw), np.sum(my2pmx2w)]])
    xmz2pmy2w = xm * mz2pmy2w
    ymx2pmz2w = ym * mx2pmz2w
    zmy2pmx2w = zm * my2pmx2w
    ymxyw = ym * mxyw
    zmxzw = zm * mxzw
    xmxyw = xm * mxyw
    zmyzw = zm * myzw
    xmxzw = xm * mxzw
    ymyzw = ym * myzw
    rhs = np.array([np.sum(xmz2pmy2w - ymxyw - zmxzw),
                    np.sum(-xmxyw + ymx2pmz2w - zmyzw),
                    np.sum(-xmxzw - ymyzw + zmy2pmx2w)])
    # % best fit minimal distance, relative to image center
    rc = np.linalg.inv(A) @ rhs

    return rc


def create_distance_grid(xc, yc, zc, voxel_shape):
    xc_ind = np.round(xc).astype(int)
    yc_ind = np.round(yc).astype(int)
    zc_ind = np.round(zc).astype(int)

    zgrid, ygrid, xgrid = np.indices(voxel_shape)
    delta_zgrid = zgrid - zc_ind
    delta_ygrid = ygrid - yc_ind
    delta_xgrid = xgrid - xc_ind

    distance_grid = np.sqrt(delta_xgrid ** 2 + delta_ygrid ** 2 + delta_zgrid ** 2)

    return distance_grid