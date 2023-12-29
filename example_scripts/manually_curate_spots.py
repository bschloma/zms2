import pickle
import zarr
import dask.array as da
from zms2.interactives.spot_curator import spot_curator

# load images
filename = r'/media/brandon/Data1/Somitogenesis/Lightsheet_Z1/2021-11-11/timeseries.norm.pred.zarr'

z = zarr.open(filename, mode='r')
timepoint = 100
mcp = da.array(z.MCP.MCP[timepoint], dtype='int32')   # just one timepoint
h2b = da.array(z.Prediction.Prediction[timepoint])

# load spots data
spots_dir = r'/media/brandon/Data1/Somitogenesis/Lightsheet_Z1/2021-11-11/tracking_cluster/spots_dilation_transfer_learning'
spots_file = spots_dir + '/spots_t' + str(timepoint)
with open(spots_file, "rb") as fp:
    spots = pickle.load(fp)

# dir to save spots in
spots_name = r'spots_t' + str(timepoint)
spots, viewer = spot_curator(mcp, spots, h2b, spots_dir, spots_name)
