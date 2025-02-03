import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
import pickle
import xarray
import matplotlib.pyplot as plt

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

# -----------------------------------------------------------------------------
# --- Optional: To force GPU usage ---
# jax.config.update('jax_platform_name', 'gpu')

# -----------------------------------------------------------------------------
# 1) LOAD NEURALGCM MODEL
# -----------------------------------------------------------------------------
gcs = gcsfs.GCSFileSystem(token='anon')

model_name = 'v1/deterministic_2_8_deg.pkl'
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

# -----------------------------------------------------------------------------
# 2) LOAD ERA5 DATA (subset or your local copy)
# -----------------------------------------------------------------------------
output_path = 'C:\\Users\\alber\\Downloads\\era5_subset.nc'
reloaded_era5 = xarray.open_dataset(output_path)
print("Reloaded ERA5 dataset:", reloaded_era5)

# -----------------------------------------------------------------------------
# 3) REGRID ERA5 DATA ONTO NEURALGCM GRID
# -----------------------------------------------------------------------------
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)

regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

print("Regridding ERA5 data...")
eval_era5 = xarray_utils.regrid(reloaded_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
eval_era5 = eval_era5.astype("float32")

print(eval_era5.time.values)
print(eval_era5.isel(time=slice(0,5)).time.values)



# -----------------------------------------------------------------------------
# 4) FORECAST SETTINGS
# -----------------------------------------------------------------------------
# For this example, let's assume we only have times [0,1,2,3]. 
# So each step is 1 hour and we do 4 steps total.
inner_steps = 1
outer_steps = 4
timedelta = np.timedelta64(inner_steps, 'h')
times = np.arange(outer_steps)  # -> [0,1,2,3]

# -----------------------------------------------------------------------------
# 5) EXTRACT INPUTS & FORCINGS, ADD RANDOM PERTURBATION
# -----------------------------------------------------------------------------
# We'll use the time=0 slice as the initial inputs.
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))

print("Inputs (before perturbation):", inputs)

# Create random noise for `specific_humidity`
rng_key = jax.random.PRNGKey(42)  # or any seed you like
if "specific_humidity" in inputs:
    shape = inputs["specific_humidity"].shape
    noise_std = 0.0002
    noise = noise_std * jax.random.normal(rng_key, shape=shape)
    # Add the noise and clip negative values
    perturbed_sh = inputs["specific_humidity"] + noise
    perturbed_sh = jnp.clip(perturbed_sh, 0, None)
    inputs["specific_humidity"] = perturbed_sh

print("Inputs (after perturbation):", inputs)

initial_state = model.encode(inputs, input_forcings, rng_key)

# Prepare all forcings for the forecast
# We'll grab times [0..4], which are 5 slices if you actually have them
all_forcings = model.forcings_from_xarray(eval_era5.isel(time=slice(0, outer_steps + 1)))

print("all_forcings shape:",
      all_forcings["specific_humidity"].shape)  # Expect (5, level, lat, lon) ...

# -----------------------------------------------------------------------------
# 6) UNROLL THE FORECAST
# -----------------------------------------------------------------------------
print("Forecasting with NeuralGCM...")
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
)

print("predictions[specific_humidity].shape:",
      predictions["specific_humidity"].shape)


predictions_ds = model.data_to_xarray(predictions, times=times)
print("Predictions dataset:", predictions_ds)


print(predictions['specific_humidity'][1])  # time index 1
print(predictions['specific_humidity'][2])
print(predictions['specific_humidity'][3])

# -----------------------------------------------------------------------------
# 7) COLLECT ERA5 TARGETS FOR THE SAME TIMES
# -----------------------------------------------------------------------------
# If your data actually has 4 hourly timesteps at [0,1,2,3],
# simply select them directly. No need for .thin(time=24).
target_trajectory = model.inputs_from_xarray(
    eval_era5.isel(time=slice(outer_steps))  # picks times [0,1,2,3]
)
target_data_ds = model.data_to_xarray(target_trajectory, times=times)

combined_ds = xarray.concat([target_data_ds, predictions_ds], dim='model')
combined_ds.coords['model'] = ['ERA5', 'NeuralGCM']

print(combined_ds)

# -----------------------------------------------------------------------------
# 8) SAVE AND PLOT RESULTS
# -----------------------------------------------------------------------------
predictions_ds.to_netcdf('forecast_predictions.nc')
print("Predictions saved to 'forecast_predictions.nc'")

combined_ds.specific_humidity.sel(level=850).plot(
    x='longitude',
    y='latitude',
    row='time',
    col='model',
    robust=True,
    cmap='viridis',
    aspect=2,
    size=2
)
plt.show()
