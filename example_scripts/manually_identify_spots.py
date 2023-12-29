import napari
import pandas as pd


def get_df_from_points(nucleus, save_dir=None):
    points = viewer.layers[-1].data
    df = pd.DataFrame()
    df['t'] = points[:, 0]
    df['z'] = points[:, 1]
    df['y'] = points[:, 2]
    df['x'] = points[:, 3]
    df['nucleus_id'] = nucleus

    if save_dir is not None:
        save_path = save_dir + f'/manual_points_df_{nucleus}.pkl'
        df.to_pickle(save_path)

    return df

# load the raw data
filename = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.norm.segmentation.mn1.culled.zarr'

# load the segments
segments_file = r'/media/brandon/Data1/Somitogenesis/Dorado/segments.zarr'

# path to locations to save
path_to_location_df_dir = r'/media/brandon/Data1/Somitogenesis/Dorado/manual_traces'


viewer = napari.Viewer()
viewer.open(filename + '/MCP/MCP')
viewer.open(filename + '/H2b/H2b')
viewer.open(segments_file + '/Segments/Segments')

