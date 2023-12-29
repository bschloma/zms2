import numpy as np
from PIL import Image
from zms2.spots.detection import extract_spot_voxels_from_zarr


path_to_mcp = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.norm.segmentation.mn1.culled.zarr/MCP/MCP'
path_to_h2b = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.norm.segmentation.mn1.culled.zarr/H2b/H2b'
save_dir = r'/home/brandon/Documents/somitogenesis/paper_1/draft_1/figures/single_cell_traces/spot_mips'

t = [81, 83, 86]
z = [69, 69, 71]
y = [790, 793, 796]
x = [432, 433, 430]

voxel_size = (3, 11, 11)
for i in range(len(t)):
    this_t = t[i]
    this_z = z[i]
    this_y = y[i]
    this_x = x[i]
    location = np.expand_dims(np.array([this_t, this_z, this_y, this_x]), axis=0)

    # mcp channel
    tmp_df = extract_spot_voxels_from_zarr(path_to_mcp, location, voxel_size=voxel_size)
    this_im = tmp_df.data.values[0]
    this_mip = np.max(this_im, axis=0)
    Image.fromarray(this_mip).save(save_dir + f'/mcp_t{this_t}.tif')

    # h2b channel
    tmp_df = extract_spot_voxels_from_zarr(path_to_h2b, location, voxel_size=voxel_size)
    this_im = tmp_df.data.values[0]
    this_mip = np.max(this_im, axis=0)
    this_mip = (this_mip * (2 ** 16 - 1)).astype('uint16')
    Image.fromarray(this_mip).save(save_dir + f'/h2b_t{this_t}.tif')

