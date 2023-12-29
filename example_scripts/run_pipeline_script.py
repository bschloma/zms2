import time
from zms2.pipeline.run_pipeline import run_pipeline

# paths to the data
path_to_raw_data = r'/media/brandon/Data1/Somitogenesis/Dorado/fused.fulltime.cropped.norm.segmentation.mn1.culled.zarr/MCP/MCP'
timepoints = None   # run pipeline on all time points
# timepoints = [0, 1]  # alternatively, provide a list of which timepoints to include

# list of which steps of the pipeline to run
steps = ['detection', 'classification', 'quantification', 'assign_nucleus', 'fill_in_traces']

# spot detection params
sigma_blur = 0.0
sigma_dog_1 = 0.68
spot_thresh = 0.001

# skin mask params
skin_sigma_blur = 5.74
skin_thresh = 10 ** -1.76
erosion_size = 0
xor_size = 0

# other parameters
path_to_model = r'/media/brandon/Data1/Somitogenesis/Dorado/kfold_models/model_1.h5'
method = 'gauss3d_dog'
path_to_segments = r'/media/brandon/Data1/Somitogenesis/Dorado/segments.zarr'
save_dir = r'/media/brandon/Data1/Somitogenesis/Dorado/gauss_001_v2'
look_for_nearby_nuclei = True
cpu_only = False
single_cpu = False
dxy = 5
dz = 3
fill_in_trace_thresh = 1.0
prob_thresh = 0.7
n_fill_iterations = 5
kwargs = {}

start_time = time.perf_counter()
df = run_pipeline(path_to_raw_data=path_to_raw_data,
                  steps=steps,
                  save_dir=save_dir,
                  timepoints=timepoints,
                  sigma_blur=sigma_blur,
                  sigma_dog_1=sigma_dog_1,
                  spot_thresh=spot_thresh,
                  skin_sigma_blur=skin_sigma_blur,
                  skin_thresh=skin_thresh,
                  erosion_size=erosion_size,
                  xor_size=xor_size,
                  cpu_only=cpu_only,
                  path_to_segments=path_to_segments,
                  path_to_model=path_to_model,
                  prob_thresh=prob_thresh,
                  method=method,
                  single_cpu=single_cpu,
                  look_for_nearby_nuclei=look_for_nearby_nuclei,
                  dxy=dxy,
                  dz=dz,
                  fill_trace_thresh=fill_in_trace_thresh,
                  n_fill_iterations=n_fill_iterations,
                  **kwargs
                  )
finish_time = time.perf_counter()
run_time = finish_time - start_time
print('it took ' + str(run_time) + ' seconds to run the whole pipeline')