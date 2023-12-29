import napari
import pandas as pd
import numpy as np


# load the raw data
filename = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.norm.segmentation.mn1.culled.zarr'

# load the segments
segments_file = r'/media/brandon/Data1/Somitogenesis/Dorado/segments.zarr/Segments/Segments'

# load the spots
path_to_spots_df = r'/media/brandon/Data1/Somitogenesis/Dorado/manual_traces/all_manual_spots_quant.pkl'

df = pd.read_pickle(path_to_spots_df)

points = np.array(df.get(['t', 'z', 'y', 'x']))
viewer = napari.Viewer()
viewer.open(filename + '/MCP/MCP')
viewer.open(filename + '/H2b/H2b')
viewer.open(segments_file)
viewer.add_points(points)

