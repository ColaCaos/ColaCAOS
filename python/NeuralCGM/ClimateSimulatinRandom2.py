import gcsfs
import jax
import numpy as np
import pickle
import xarray 


from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm
import matplotlib.pyplot as plt

# To force GPU
#jax.config.update('jax_platform_name', 'gpu')

gcs = gcsfs.GCSFileSystem(token='anon')

model_name = 'v1/deterministic_2_8_deg.pkl'  #@param ['v1/deterministic_0_7_deg.pkl', 'v1/deterministic_1_4_deg.pkl', 'v1/deterministic_2_8_deg.pkl', 'v1/stochastic_1_4_deg.pkl', 'v1_precip/stochastic_precip_2_8_deg.pkl', 'v1_precip/stochastic_evap_2_8_deg'] {type: "string"}

with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

demo_start_time = '2020-02-14'
demo_end_time = '2020-02-18'
data_inner_steps = 24  # process every 24th hour

# sliced_era5 = (
#     full_era5
#     [model.input_variables + model.forcing_variables]
#     .pipe(
#         xarray_utils.selective_temporal_shift,
#         variables=model.forcing_variables,
#         time_shift='24 hours',
#     )
#     .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
#     .compute()
# )

# Save to NetCDF format on your local PC
output_path = 'C:\\Users\\alber\\Downloads\\era5_subset.nc'  # Escape backslashes
# sliced_era5.to_netcdf(output_path)
# print(f"Data saved to {output_path}")


# Reload the dataset
reloaded_era5 = xarray.open_dataset(output_path)

print("reloaded dataset")
 
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

print("process dataset")

# Process the dataset for the model
eval_era5 = xarray_utils.regrid(reloaded_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
eval_era5 = eval_era5.astype("float32")


# Step 1: Extract the u_component_of_wind data
u_wind = eval_era5['u_component_of_wind']
v_wind = eval_era5['v_component_of_wind']

# Step 2: Calculate the variance (1% of the maximum value of u_component_of_wind)
max_u_wind = u_wind.max().item()  # Get the maximum value as a scalar
max_v_wind = v_wind.max().item()  # Get the maximum value as a scalar
varianceU = 1 * max_u_wind  # 1% of the maximum value
varianceV = 1 * max_v_wind  # 1% of the maximum value
# Step 3: Generate Gaussian random perturbation
# The perturbation will have mean 0 and the calculated variance
perturbationU = np.random.normal(
    loc=0,  # Mean
    scale=np.sqrt(varianceU),  # Standard deviation
    size=u_wind.shape  # Shape matches the original data
)
perturbationV = np.random.normal(
    loc=0,  # Mean
    scale=np.sqrt(varianceV),  # Standard deviation
    size=u_wind.shape  # Shape matches the original data
)
# Step 4: Add the perturbation to the u_component_of_wind data
perturbed_u_wind = u_wind + perturbationU
perturbed_v_wind = v_wind + perturbationV
# Step 5: Update the dataset with the perturbed u_component_of_wind
eval_era5['u_component_of_wind'] = xarray.DataArray(
    data=perturbed_u_wind,
    dims=u_wind.dims,
    coords=u_wind.coords,
    attrs=u_wind.attrs  # Preserve metadata
)
eval_era5['v_component_of_wind'] = xarray.DataArray(
    data=perturbed_v_wind,
    dims=v_wind.dims,
    coords=v_wind.coords,
    attrs=v_wind.attrs  # Preserve metadata
)
print("Perturbation introduced in u_component_of_wind")


inner_steps = 24  # save model outputs once every 24 hours
outer_steps = 4 * 24 // inner_steps  # total of 4 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

print("initialize model state")

# initialize model state
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
rng_key = jax.random.key(42)  # optional for deterministic models
initial_state = model.encode(inputs, input_forcings, rng_key)

# use persistence for forcing variables (SST and sea ice cover)
all_forcings = model.forcings_from_xarray(eval_era5)

print("forecasting")
# make forecast
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=False,
)

print("forecasting finished")

predictions_ds = model.data_to_xarray(predictions, times=times)

print("predictions_ds")

# Selecting ERA5 targets from exactly the same time slice
target_trajectory = model.inputs_from_xarray(
    eval_era5
    .thin(time=(inner_steps // data_inner_steps))
    .isel(time=slice(outer_steps))
)
target_data_ds = model.data_to_xarray(target_trajectory, times=times)

combined_ds = xarray.concat([target_data_ds, predictions_ds], 'model')
combined_ds.coords['model'] = ['ERA5', 'NeuralGCM']

print("starting visualization")
# Visualize ERA5 vs NeuralGCM trajectories
#combined_ds.specific_humidity.sel(level=850).plot(
#    x='longitude', y='latitude', row='time', col='model', robust=True, aspect=2, size=2
#);
#plt.show()

# Visualize ERA5 vs NeuralGCM trajectories
combined_ds.temperature.sel(level=850).plot(
    x='longitude', y='latitude', row='time', col='model', robust=True, aspect=2, size=2
);
# Show the plot
plt.show()