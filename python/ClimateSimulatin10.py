import gcsfs
import jax
import numpy as np
import pickle
import xarray
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
import neuralgcm
import matplotlib.pyplot as plt

# JAX settings to force GPU usage if needed
# jax.config.update('jax_platform_name', 'gpu')

# Load the model
gcs = gcsfs.GCSFileSystem(token='anon')
model_name = 'v1/deterministic_2_8_deg.pkl'
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
    ckpt = pickle.load(f)
model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

# Load ERA5 dataset
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(gcs.get_mapper(era5_path), chunks=None)

output_path = 'C:\\Users\\alber\\Downloads\\era5_subset.nc'
reloaded_era5 = xarray.open_dataset(output_path)

# Define regridder
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)

# Preprocess ERA5 data
eval_era5 = xarray_utils.regrid(reloaded_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)
eval_era5 = eval_era5.astype("float32")

# Model parameters
inner_steps = 24  # Save model outputs every 24 hours
outer_steps = 8 * 24 // inner_steps  # Total of 8 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # Time axis in hours

# Initial inputs and forcings
inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

# Temporarily disable JIT for debugging
def unroll_model(initial_state, forcings, steps, timedelta):
    return model.unroll(initial_state, forcings, steps=steps, timedelta=timedelta, start_with_input=True)

# Generate forecasts with slightly perturbed initial conditions
num_simulations = 2  # Start with 2 simulations for debugging
perturbation_scale = 0.01  # Scale of the perturbations
forecasts = []

rng_key = jax.random.PRNGKey(0)  # Base random key

for sim_number in range(num_simulations):
    print(f"Starting simulation {sim_number + 1} of {num_simulations}")
    
    # Split RNG for this simulation
    rng_key, subkey = jax.random.split(rng_key)
    
    # Apply perturbation to each array in the inputs dictionary
    perturbed_inputs = {
        key: value + perturbation_scale * jax.random.normal(subkey, value.shape)
        for key, value in inputs.items()
    }

    # Debug perturbed inputs
    for key, value in perturbed_inputs.items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    # Encode initial state
    initial_state = model.encode(perturbed_inputs, input_forcings, subkey)

    # Debug initial state and forcings
    print(f"Initial state shape: {initial_state.shape}")
    print("All forcings:")
    for key, value in all_forcings.items():
        print(f"{key}: shape {value.shape}, dtype {value.dtype}")

    print(f"Steps: {outer_steps}, Timedelta: {timedelta}")

    # Run the model for all steps in one call
    try:
        final_state, predictions = unroll_model(
            initial_state,
            all_forcings,
            steps=outer_steps,
            timedelta=timedelta
        )
    except ValueError as e:
        print(f"Error during model.unroll: {e}")
        raise

    # Collect results
    forecasts.append(model.data_to_xarray(predictions, times=times))
    print(f"Finished simulation {sim_number + 1} of {num_simulations}")

    # Free memory
    del perturbed_inputs, initial_state, predictions

# Plot the forecasts for each simulation as separate world maps
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, (forecast, ax) in enumerate(zip(forecasts, axes)):
    try:
        # Ensure the forecast data has the required dimensions and coordinates
        forecast_variable = forecast.specific_humidity.sel(level=850).isel(time=-1)
        
        # Plot with a world map projection
        forecast_variable.plot(
            x='longitude',
            y='latitude',
            robust=True,
            ax=ax,
            cmap='viridis',  # Choose a clear colormap for world maps
            cbar_kwargs={"shrink": 0.6}
        )
        ax.set_title(f"Simulation {i+1}")
    except Exception as e:
        print(f"Failed to plot simulation {i+1}: {e}")
        ax.set_title(f"Simulation {i+1} - Error")
        ax.axis('off')  # Hide the axis for plots with errors

# Adjust layout
plt.tight_layout()
plt.suptitle("Specific Humidity at 850 hPa for 10 Simulations", fontsize=16, y=1.02)
plt.show()
