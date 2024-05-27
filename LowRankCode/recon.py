# %%

import yaml
import argparse
import numpy as np
import os
import wandb
import sys
from tqdm import tqdm


# %%
def load_config(file_path):
    """
    Load and parse a YAML configuration file.

    Parameters:
    - file_path: str, path to the YAML configuration file.

    Returns:
    - config: dict, configuration parameters.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process some configuration and GPU index."
    )
    parser.add_argument(
        "--gpu_index", type=int, required=True, help="Index of the GPU to use"
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    # Set up GPU for JAX
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Set the Pytorch device
    device = "cpu"

    # Import all things JAX related
    import jax.numpy as jnp
    from jax import lax
    from jax import random
    from flax import linen as nn
    import jax
    import optax
    from flax.linen import avg_pool

    test = jnp.zeros((1, 1))

    # Import all things Torch related
    import torch
    import torchvision

    # Import the custom functions
    sys.path.append("/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope")
    import sdc_jax as sdc

    # Load the configuration
    config = load_config(args.config_file_path)

    # Accessing the configuration parameters
    calibration_folder_location = config["calibration"]["folder_location"]
    psf_name = config["calibration"]["psf_name"]
    calibration_wavelengths_file = config["calibration"]["calibration_wavelengths_file"]
    filter_cube_file = config["calibration"]["filter_cube_file"]

    datafolder = config["data"]["datafolder"]
    sample = config["data"]["sample"]

    bits = config["measurement_processing"]["bits"]
    crop_indices = config["measurement_processing"]["crop_indices"]
    wvmin = config["measurement_processing"]["wvmin"]
    wvmax = config["measurement_processing"]["wvmax"]
    wvstep = config["measurement_processing"]["wvstep"]
    downsample_factor = config["measurement_processing"]["downsample_factor"]

    kmax = config["reconstruction"]["kmax"]
    step_size = config["reconstruction"]["step_size"]
    thr = config["reconstruction"]["thr"]
    xytv = config["reconstruction"]["xytv"]
    lamtv = config["reconstruction"]["lamtv"]
    use_low_rank = config["reconstruction"]["use_low_rank"]
    use_one_hot = config["reconstruction"]["use_one_hot"]
    rank = config["reconstruction"]["rank"]

    kprint = config["wandb"]["kprint"]
    project_name = config["wandb"]["project_name"]
    run_name = config["wandb"]["run_name"]
    save_location = config["wandb"]["save_location"]

    # Print the loaded configuration (for demonstration purposes)
    print(f"Using GPU index: {args.gpu_index}")
    print(f"Configuration loaded from {args.config_file_path}")

    # Load the measurement data
    try:
        sample_meas = sdc.importTiff(datafolder, "meas.tiff") / (2**bits - 1)
    except:
        sample_meas = torch.mean(
            sdc.tif_loader(os.path.join(datafolder, "measurements")) / (2**bits - 1), 0
        )

    # Load the background image
    try:
        background = sdc.importTiff(datafolder, "bg.tiff") / 2**bits
    except:
        print("No background image found, continuing without background subtraction")
        background = torch.zeros(sample_meas.shape)

    # Processd the measurement data
    meas = sdc.cropci((sample_meas - background).clip(0, 1), crop_indices)
    meas = meas / meas.max()

    # Load wavelength calibration and downsample to spectral resolution of filter cube
    try:
        wv = torch.load(
            os.path.join(calibration_folder_location, calibration_wavelengths_file),
            map_location="cpu",
        )
    except:
        wv = np.load(
            os.path.join(calibration_folder_location, calibration_wavelengths_file)
        )
        wv = torch.tensor(wv)
    wavelengths = np.arange(wvmin, wvmax + wvstep, wvstep)

    # Load and crop filter cube
    try:
        normalized_filter_cube = torch.load(
            os.path.join(calibration_folder_location, filter_cube_file),
            map_location="cpu",
        )
    except:
        normalized_filter_cube = np.load(
            os.path.join(calibration_folder_location, filter_cube_file)
        )
        normalized_filter_cube = torch.tensor(normalized_filter_cube)
    filterstack = sdc.cropci(normalized_filter_cube, crop_indices)
    msum = sdc.sumFilterArray(filterstack, wv, wvmin, wvmax, wvstep)
    spectral_filter = msum / torch.amax(msum)
    spectral_filter = spectral_filter - torch.amin(spectral_filter, 0, keepdim=True)[0]

    # Load and crop PSF
    sensor_psf = torch.load(
        os.path.join(calibration_folder_location, psf_name), map_location="cpu"
    )
    ccrop = torchvision.transforms.CenterCrop(spectral_filter.shape[1:])
    psf = ccrop(sensor_psf)
    psf = psf / torch.sum(psf)
    psf = psf.clip(0)

    # Load ground truth image
    try:
        gt = sdc.importTiff(datafolder, "gt.tiff")
        if gt.shape[-1] != 3:
            gt = gt / 2**bits
            gt = torchvision.transforms.functional.rotate(
                gt.unsqueeze(0), -90
            ).squeeze()
        gt = gt / torch.max(gt)
    except:
        print("No ground truth image found, continuing without ground truth")
        gt = torch.zeros(meas.shape)

    # Downsample and move everything to GPU
    m = avg_pool(
        jnp.array(spectral_filter)[..., None],
        (downsample_factor, downsample_factor),
        (downsample_factor, downsample_factor),
        "VALID",
    ).squeeze()
    psf = avg_pool(
        jnp.array(psf)[None, ..., None],
        (downsample_factor, downsample_factor),
        (downsample_factor, downsample_factor),
        "VALID",
    ).squeeze()
    meas = avg_pool(
        jnp.array(meas)[None, ..., None],
        (downsample_factor, downsample_factor),
        (downsample_factor, downsample_factor),
        "VALID",
    ).squeeze()
    gt = gt.to(device)

    # Set up fourier space of PSF
    xk = jnp.zeros(m.shape)
    padding = (
        (0, 0, 0),
        (
            np.ceil(xk.shape[1] / 2).astype(int),
            np.floor(xk.shape[1] / 2).astype(int),
            0,
        ),
        (
            np.ceil(xk.shape[2] / 2).astype(int),
            np.floor(xk.shape[2] / 2).astype(int),
            0,
        ),
    )
    hpad = jax.lax.pad(psf[None, ...], 0.0, padding).squeeze()
    hfftpad = jnp.fft.fft2(hpad)

    # Calculate the adjoint
    xk = sdc.jax_adjoint_model(meas, m, hfftpad, padding)

    # initialize the optimizer
    optimizer = optax.adam(learning_rate=step_size, b1=0.9, b2=0.999)

    # If using low rank, calculate the low rank approximation
    if use_low_rank:
        # Reshape xk for SVD
        W = xk.shape[0]
        X = xk.shape[2]
        Y = xk.shape[1]
        xk_reshaped = xk.reshape(xk.shape[0], -1)  # Shape (Lambda, X*Y)

        # Perform SVD
        U, S, VT = jnp.linalg.svd(xk_reshaped, full_matrices=False)

        # Keep only the first rank components
        U = U[:, :rank]
        S = S[:rank]
        VT = VT[:rank, :]

        # Combine the right singular vectors with the singular values
        V = jnp.diag(S) @ VT

        # Initialize optimizer states for U and V separately
        opt_state_U = optimizer.init(U)
        opt_state_V = optimizer.init(V)

        if use_one_hot:
            # define weights
            weights = V[:]

            # define the starting temperature
            temperature = 1.0

            # Initialize optimizer state for weights
            opt_state_weights = optimizer.init(weights)

            # Define the loss function
            loss_func = sdc.one_hot_loss

            # define loss and gradient functions
            loss_and_grad = jax.jit(jax.value_and_grad(loss_func, (0, 1, 2)))

        else:
            # define the loss function
            loss_func = sdc.low_rank_loss

            # define loss and gradient functions
            loss_and_grad = jax.jit(jax.value_and_grad(loss_func, (0, 1)))

    else:
        # Initialize optimizer state
        opt_state = optimizer.init(xk)

        # define the loss function
        loss_func = sdc.loss

        # define loss and gradient functions
        loss_and_grad = jax.jit(jax.value_and_grad(loss_func, (0)))

    # Initialize wandb
    wandb.init(
        # Set the project name
        project=project_name,
        # Set the run name
        name=run_name,
        # Track the config
        config=config,
    )

    for k in tqdm(range(kmax)):
        # initialize the wandb log
        wandb_log = {}

        # log the measurement, psf, and ground truth at the start of the reconstruction
        if k == 0:
            wandb_log = sdc.wandb_log_meas(wandb_log, meas)
            wandb_log = sdc.wandb_log_psf(wandb_log, psf)
            wandb_log = sdc.wandb_log_ground_truth(wandb_log, gt)

            # Calculate the initial reconstruction
            if use_low_rank:
                if use_one_hot:
                    xk = sdc.one_hot_reconstruction(U, V, weights, temperature).reshape(
                        W, Y, X
                    )
                else:
                    xk = sdc.low_rank_reconstruction(U, V).reshape(W, Y, X)

        # log the simulated measurement, false color reconstruction, and low rank components at kprint intervals
        if k % kprint == 0:
            wandb_log = sdc.wandb_log_sim_meas(
                wandb_log, sdc.jax_forward_model(xk, m, hfftpad)
            )
            wandb_log = sdc.wandb_log_false_color_recon(
                wandb_log, xk / jnp.max(xk) * jnp.sum(xk, 0)[None, ...], wavelengths
            )
            if use_low_rank:
                wandb_log = sdc.wandb_log_low_rank_components(wandb_log, U, wavelengths)

        if use_low_rank:
            if use_one_hot:
                if k > 500:
                    temperature *= 0.994

                # calculate the loss and gradients
                loss, (grad_U, grad_V, grad_weights) = loss_and_grad(
                    U, V, weights, meas, hfftpad, m, thr, xytv, lamtv, temperature
                )

                # Get updates to U, V, and weights
                updates_U, opt_state_U = optimizer.update(grad_U, opt_state_U, U)
                updates_V, opt_state_V = optimizer.update(grad_V, opt_state_V, V)
                updates_weights, opt_state_weights = optimizer.update(
                    grad_weights, opt_state_weights, weights
                )

                # Remove any nans
                updates_U = jnp.nan_to_num(updates_U)
                updates_V = jnp.nan_to_num(updates_V)
                updates_weights = jnp.nan_to_num(updates_weights)

                # Apply updates to U, V, and weights
                U = optax.apply_updates(U, updates_U)
                V = optax.apply_updates(V, updates_V)
                weights = optax.apply_updates(weights, updates_weights)

                # Clip U, V, and weights to be non-negative
                U = jnp.clip(U, 0, None)
                V = jnp.clip(V, 0, None)
                weights = jnp.clip(weights, 0, None)

                # Calculate the new xk
                xk = sdc.one_hot_reconstruction(U, V, weights, temperature).reshape(
                    W, Y, X
                )

            else:
                # calculate the loss and gradients
                loss, (grad_U, grad_V) = loss_and_grad(
                    U, V, meas, hfftpad, m, thr, xytv, lamtv
                )

                # Get updates to U and V
                updates_U, opt_state_U = optimizer.update(grad_U, opt_state_U, U)
                updates_V, opt_state_V = optimizer.update(grad_V, opt_state_V, V)

                # Remove any nans
                updates_U = jnp.nan_to_num(updates_U)
                updates_V = jnp.nan_to_num(updates_V)

                # Apply updates to U and V
                U = optax.apply_updates(U, updates_U)
                V = optax.apply_updates(V, updates_V)

                # Clip U and V to be non-negative
                U = jnp.clip(U, 0, None)
                V = jnp.clip(V, 0, None)

                # Calculate the new xk
                xk = sdc.low_rank_reconstruction(U, V).reshape(W, Y, X)

        else:
            # calculate the loss and gradients
            loss, grad = loss_and_grad(xk, meas, hfftpad, m, thr, xytv, lamtv)

            # Get updates
            updates, opt_state = optimizer.update(grad, opt_state, xk)

            # Remove any nans
            updates = jnp.nan_to_num(updates)

            # Apply updates
            xk = optax.apply_updates(xk, updates)

            # Clip xk to be non-negative
            xk = jnp.clip(xk, 0, None)

        # log the mse of the measurement and simulated measurement
        wandb_log["data_loss"] = (
            jnp.linalg.norm((sdc.jax_forward_model(xk, m, hfftpad) - meas).ravel(), 2)
            ** 2
        )

        # log the custom loss
        wandb_log["loss"] = loss

        # log everything to wandb
        wandb.log(wandb_log)

# %%
