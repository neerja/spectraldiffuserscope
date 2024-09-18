# %%
import yaml
import argparse
import numpy as np
import os
import wandb
import sys
from tqdm import tqdm
import pickle
import jax.numpy as jnp
from flax.linen import avg_pool
import jax
import optax
from jax import lax
import sdc_jax as sdc
# %%
def load_config(file_path):
    """
    Load and parse a YAML configuration file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Dictionary containing configuration parameters.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def initialize_data(config):
    """
    Initialize the necessary data for the reconstruction process, including measurements, PSF, etc.

    Args:
        config (dict): Configuration dictionary with paths and parameters.

    Returns:
        Tuple: Initialized measurements, PSF, filter stack, and other required data.
    """
    # Load the measurement data
    bits = config["measurement_processing"]["bits"]
    crop_indices = config["measurement_processing"]["crop_indices"]
    datafolder = config["data"]["datafolder"]
    calibration_folder_location = config["calibration"]["folder_location"]
    calibration_wavelengths_file = config["calibration"]["calibration_wavelengths_file"]
    psf_name = config["calibration"]["psf_name"]
    filter_cube_file = config["calibration"]["filter_cube_file"]
    
    try:
        sample_meas = sdc.importTiff(datafolder, "meas.tiff") / (2**bits - 1)
    except:
        sample_meas = jnp.mean(sdc.tif_loader(os.path.join(datafolder, "measurements")) / (2**bits - 1), axis=0)

    # Load the background image
    try:
        background = sdc.importTiff(datafolder, "bg.tiff") / 2**bits
    except:
        print("No background image found, continuing without background subtraction")
        background = jnp.zeros_like(sample_meas)

    # Process the measurement data
    meas = sdc.cropci((sample_meas - background).clip(0, 1), crop_indices)
    meas = meas / meas.max()

    # Load wavelength calibration and downsample to spectral resolution of filter cube
    wv = jnp.load(os.path.join(calibration_folder_location, calibration_wavelengths_file))
    wavelengths = jnp.arange(config["measurement_processing"]["wvmin"],
                             config["measurement_processing"]["wvmax"] + config["measurement_processing"]["wvstep"],
                             config["measurement_processing"]["wvstep"])

    # Load and crop filter cube
    normalized_filter_cube = jnp.load(os.path.join(calibration_folder_location, filter_cube_file))
    filterstack = sdc.cropci(normalized_filter_cube, crop_indices)
    msum = sdc.sumFilterArray(filterstack, wv, config["measurement_processing"]["wvmin"],
                              config["measurement_processing"]["wvmax"],
                              config["measurement_processing"]["wvstep"])
    spectral_filter = msum / jnp.amax(msum)
    spectral_filter = spectral_filter - jnp.amin(spectral_filter, axis=0, keepdims=True)[0]

    # Load and process the PSF
    sensor_psf = jnp.load(os.path.join(calibration_folder_location, psf_name))
    psf = sdc.center_crop(sensor_psf, spectral_filter.shape[1:])
    psf = psf / jnp.sum(psf)
    psf = psf.clip(0)

    # Downsample measurements, PSF, and spectral filter
    downsample_factor = config["measurement_processing"]["downsample_factor"]
    m = avg_pool(jnp.array(spectral_filter)[..., None],
                 (downsample_factor, downsample_factor),
                 (downsample_factor, downsample_factor),
                 "VALID").squeeze()
    psf = avg_pool(jnp.array(psf)[None, ..., None],
                   (downsample_factor, downsample_factor),
                   (downsample_factor, downsample_factor),
                   "VALID").squeeze()
    meas = avg_pool(jnp.array(meas)[None, ..., None],
                    (downsample_factor, downsample_factor),
                    (downsample_factor, downsample_factor),
                    "VALID").squeeze()

    # Initialize Fourier space of PSF
    xk = jnp.zeros_like(m)
    padding = (
        (0, 0, 0),
        (np.ceil(xk.shape[1] / 2).astype(int), np.floor(xk.shape[1] / 2).astype(int), 0),
        (np.ceil(xk.shape[2] / 2).astype(int), np.floor(xk.shape[2] / 2).astype(int), 0),
    )
    hpad = jax.lax.pad(psf[None, ...], 0.0, padding).squeeze()
    hfftpad = jnp.fft.fft2(hpad)


    xk = sdc.jax_adjoint_model(meas, m, hfftpad, padding)

    return meas, psf, m, xk, hfftpad

def initialize_svd(xk, rank):
    """
    Perform SVD on reshaped xk to get U, S, and V matrices.

    Args:
        xk (jnp.array): The input data to be factorized.
        rank (int): The number of singular values to keep.

    Returns:
        Tuple: Matrices U, S, and V after performing SVD.
    """
    W = xk.shape[0]
    Y = xk.shape[1]
    X = xk.shape[2]
    xk_reshaped = xk.reshape(W, -1)  # Shape (Lambda, X*Y)

    # Perform SVD
    U, S, VT = jnp.linalg.svd(xk_reshaped, full_matrices=False)
    U = U[:, :rank]
    S = S[:rank]
    VT = VT[:rank, :]

    # Combine the right singular vectors with singular values
    V = jnp.diag(S) @ VT

    return U, V, W, Y, X

class Reconstruction:
    """
    Base class for reconstruction strategies.

    Args:
        optimizer (optax.GradientTransformation): Optimizer for updating parameters.
    """
    
    def __init__(self, optimizer, **kwargs):
        self.optimizer = optimizer
        self.opt_state = None  # to be initialized in subclasses

    def init_params(self):
        """
        Initialize optimizer states or parameters. 
        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def reconstruct(self, *args):
        """
        Perform the reconstruction process. 
        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def compute_loss_and_grad(self, *args):
        """
        Compute loss and gradients for the reconstruction.
        Must be implemented in subclasses.
        """
        raise NotImplementedError

    def apply_updates(self, grads):
        """
        Apply the gradients to update parameters using the optimizer.
        This needs to be overridden in subclasses to handle specific parameters.
        """
        raise NotImplementedError

    def get_save_dict(self):
        """
        Return a dictionary of the current reconstruction parameters for saving.
        Must be implemented in subclasses.
        """
        raise NotImplementedError

class LowRankReconstruction(Reconstruction):
    """
    Low-rank reconstruction strategy.

    Args:
        U (jnp.array): Left singular vectors.
        V (jnp.array): Right singular vectors.
    """
    
    def __init__(self, U, V, W, Y, X, **kwargs):
        super().__init__(**kwargs)
        self.U = U
        self.V = V
        self.W = W
        self.Y = Y
        self.X = X
        self.opt_state_U = None
        self.opt_state_V = None

    def init_params(self):
        """Initialize optimizer states for U and V."""
        self.opt_state_U = self.optimizer.init(self.U)
        self.opt_state_V = self.optimizer.init(self.V)

    def reconstruct(self):
        """
        Reconstruct using low-rank approximation.

        Returns:
            jnp.array: The reconstructed output.
        """
        return sdc.low_rank_reconstruction(self.U, self.V).reshape(self.W, self.Y, self.X)

    def loss_func(self, U, V, meas, padded_psf_fft, filter_array, thr, xytv, lamtv):
        """
        Calculate the loss for low-rank reconstruction.
        """
        xk = sdc.low_rank_reconstruction(U, V).reshape(filter_array.shape)
        sim_meas = sdc.jax_forward_model(xk, filter_array, padded_psf_fft)

        dlam = jnp.gradient(U, axis=0)
        ddlam = jnp.gradient(dlam, axis=0)
        dy, dx = jnp.gradient(xk, axis=(-1, -2))

        lamtv_loss = jnp.linalg.norm(ddlam, 1)
        data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
        tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
        sparsity_loss = jnp.linalg.norm(V.ravel(), 1)

        return data_loss + lamtv * lamtv_loss + xytv * tv_loss + thr * sparsity_loss

    def compute_loss_and_grad(self, meas, hfftpad, m, thr, xytv, lamtv):
        """
        Compute the loss and gradients for low-rank reconstruction.
        """
        return jax.value_and_grad(self.loss_func, argnums=(0, 1))(
            self.U, self.V, meas, hfftpad, m, thr, xytv, lamtv
        )

    def apply_updates(self, grads):
        """
        Apply gradient updates to U and V.
        """
        updates_U, self.opt_state_U = self.optimizer.update(grads[0], self.opt_state_U, self.U)
        updates_V, self.opt_state_V = self.optimizer.update(grads[1], self.opt_state_V, self.V)
        self.U = optax.apply_updates(self.U, updates_U)
        self.V = optax.apply_updates(self.V, updates_V)
        self.U = jnp.clip(self.U, 0, None)
        self.V = jnp.clip(self.V, 0, None)
        

    def get_save_dict(self):
        """
        Return a dictionary of the current reconstruction parameters for saving.
        """
        return {
            'U': self.U,
            'V': self.V,
            'xk': self.reconstruct()
        }
    
class OneHotReconstruction(LowRankReconstruction):
    """
    One-hot reconstruction strategy, extending low-rank strategy.
    """
    
    def __init__(self, U, V, weights=None, temperature=1.0, temperature_decay=0.994, **kwargs):
        super().__init__(U, V, **kwargs)
        self.weights = weights if weights is not None else V[:]
        self.temperature = temperature
        self.temperature_decay = temperature_decay
        self.opt_state_weights = None

    def init_params(self):
        """Initialize optimizer states for U, V, and weights."""
        super().init_params()
        self.opt_state_weights = self.optimizer.init(self.weights)

    def reconstruct(self):
        """
        Perform one-hot reconstruction using U, V, and weights.
        """
        return sdc.one_hot_reconstruction(self.U, self.V, self.weights, self.temperature).reshape(self.W, self.Y, self.X)

    def loss_func(self, U, V, weights, meas, padded_psf_fft, filter_array, thr, xytv, lamtv, temperature):
        """
        Calculate the loss for one-hot reconstruction.
        """
        xk = sdc.one_hot_reconstruction(U, V, weights, temperature).reshape(filter_array.shape)
        sim_meas = sdc.jax_forward_model(xk, filter_array, padded_psf_fft)

        dlam = jnp.gradient(U, axis=0)
        ddlam = jnp.gradient(dlam, axis=0)
        dy, dx = jnp.gradient(xk, axis=(-1, -2))

        lamtv_loss = jnp.linalg.norm(ddlam, 1)
        data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
        tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
        sparsity_loss = jnp.linalg.norm(V.ravel(), 1)

        return data_loss + lamtv * lamtv_loss + xytv * tv_loss + thr * sparsity_loss

    def compute_loss_and_grad(self, meas, hfftpad, m, thr, xytv, lamtv):
        """
        Compute the loss and gradients for one-hot reconstruction.
        """
        return jax.value_and_grad(self.loss_func, argnums=(0, 1, 2))(
            self.U, self.V, self.weights, meas, hfftpad, m, thr, xytv, lamtv, self.temperature
        )

    def apply_updates(self, grads):
        """
        Apply gradient updates to U, V, and weights.
        """
        updates_U, self.opt_state_U = self.optimizer.update(grads[0], self.opt_state_U, self.U)
        updates_V, self.opt_state_V = self.optimizer.update(grads[1], self.opt_state_V, self.V)
        updates_weights, self.opt_state_weights = self.optimizer.update(grads[2], self.opt_state_weights, self.weights)
        
        self.U = optax.apply_updates(self.U, updates_U)
        self.V = optax.apply_updates(self.V, updates_V)
        self.weights = optax.apply_updates(self.weights, updates_weights)
        self.U = jnp.clip(self.U, 0, None)
        self.V = jnp.clip(self.V, 0, None)
        self.weights = jnp.clip(self.weights, 0, None)

    def update_temperature(self):
        """
        Update the temperature parameter using the decay factor.
        """
        self.temperature *= self.temperature_decay

    def get_save_dict(self):
        """
        Return a dictionary of the current reconstruction parameters for saving.
        """
        return {
            'U': self.U,
            'V': self.V,
            'weights': self.weights,
            'temperature': self.temperature,
            'xk': self.reconstruct()
        }
  
class RegularReconstruction(Reconstruction):
    """
    Regular reconstruction strategy (no low-rank or one-hot encoding).
    """
    
    def __init__(self, xk, **kwargs):
        super().__init__(**kwargs)
        self.xk = xk
        self.opt_state = None

    def init_params(self):
        """Initialize optimizer state for xk."""
        self.opt_state = self.optimizer.init(self.xk)

    def reconstruct(self):
        """
        Perform regular reconstruction.

        Returns:
            jnp.array: The current state of xk.
        """
        return self.xk

    def loss_func(self, xk, meas, padded_psf_fft, filter_array, thr, xytv, lamtv):
        """
        Define the overall loss function for regular reconstruction.
        """
        sim_meas = sdc.jax_forward_model(xk, filter_array, padded_psf_fft)

        dlam, dy, dx = jnp.gradient(xk, axis=(0, 1, 2))
        ddlam = jnp.gradient(dlam, axis=0)

        data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
        tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
        sparsity_loss = jnp.linalg.norm(xk.ravel(), 1)
        lamtv_loss = jnp.linalg.norm(ddlam.ravel(), 2) ** 2

        return data_loss + xytv * tv_loss + lamtv * lamtv_loss + thr * sparsity_loss

    def compute_loss_and_grad(self, meas, hfftpad, m, thr, xytv, lamtv):
        """
        Compute the loss and gradients for regular reconstruction.
        """
        return jax.value_and_grad(self.loss_func, argnums=(0))(
            self.xk, meas, hfftpad, m, thr, xytv, lamtv
        )

    def apply_updates(self, grads):
        """
        Apply gradient updates to xk.
        """
        updates_xk, self.opt_state = self.optimizer.update(grads, self.opt_state, self.xk)
        self.xk = optax.apply_updates(self.xk, updates_xk)
        self.xk = jnp.clip(self.xk, 0, None)

    def get_save_dict(self):
        """
        Return a dictionary of the current reconstruction parameters for saving.
        """
        return {
            'xk': self.xk
        }


def get_reconstruction_strategy(use_low_rank, use_one_hot, **kwargs):
    """
    Factory function to return the appropriate reconstruction strategy.

    Args:
        use_low_rank (bool): Flag to indicate if low-rank reconstruction should be used.
        use_one_hot (bool): Flag to indicate if one-hot encoding should be used.

    Returns:
        Reconstruction: An instance of a subclass of Reconstruction.
    """
    if use_low_rank:
        if use_one_hot:
            return OneHotReconstruction(**kwargs)
        return LowRankReconstruction(**kwargs)
    return RegularReconstruction(**kwargs)

def run_reconstruction(
    strategy, kmax, save_freq, kprint, meas, hfftpad, m, thr, xytv, lamtv, 
    W, Y, X, save_location, wavelengths, run_name, run_id, overwrite
):
    """
    Run the reconstruction process and log progress.

    Args:
        strategy (Reconstruction): Reconstruction strategy object.
        kmax (int): Number of iterations for the reconstruction process.
        save_freq (int): Frequency of saving the reconstruction state.
        kprint (int): Frequency of printing/logging intermediate results.
        meas (jnp.array): Measurement data.
        hfftpad (jnp.array): Fourier space of PSF.
        m (jnp.array): Measurement processing data.
        thr (float): Threshold value.
        xytv (float): XY total variation.
        lamtv (float): Regularization parameter for total variation.
        W (int): Image width.
        Y (int): Image height.
        X (int): Image depth.
        save_location (str): Directory to save the results.
        wavelengths (jnp.array): Array of wavelengths.
        run_name (str): Name of the current run for logging.
        run_id (str): ID of the current run for logging.
        overwrite (bool): Flag to overwrite the same file or save iterations separately.
    """
    strategy.init_params()

    for k in tqdm(range(kmax)):
        wandb_log = {}

        if k % save_freq == 0:
            save_reconstruction(k, strategy, save_location,run_name, run_id)

        if k % kprint == 0:
            log_intermediate_results(wandb_log, strategy, k, wavelengths, run_name)

        # Compute loss and gradients
        loss, grads = strategy.compute_loss_and_grad(meas, hfftpad, m, thr, xytv, lamtv)

        # Apply updates
        strategy.apply_updates(grads)

        # Update temperature if using OneHotReconstruction
        if isinstance(strategy, OneHotReconstruction):
            strategy.update_temperature()

        # Log loss values
        wandb_log["loss"] = loss
        wandb.log(wandb_log)
    
    # Log final results
    log_intermediate_results(wandb_log, strategy, k, wavelengths, run_name)
    save_reconstruction(k, strategy, save_location,run_name, run_id, overwrite)

def save_reconstruction(k, strategy, save_location, run_name, run_id, overwrite=True):
    """
    Save the current reconstruction state to a pickle file.

    Args:
        k (int): Current iteration number.
        strategy (Reconstruction): Reconstruction strategy object.
        save_location (str): Directory to save the pickle file.
        run_name (str): Name of the current run for logging.
        run_id (str): ID of the current run for logging.
        overwrite (bool): Flag to overwrite the same file or save iterations separately. Default value = True
    """
    
    save_dict = strategy.get_save_dict()
    if overwrite: # overwrite the same file
        with open(os.path.join(save_location, f"{run_name}_{run_id}.pkl"), "wb") as f:
            pickle.dump(save_dict, f)
    else: # else save iterations separately
        with open(os.path.join(save_location, f"{run_name}_{run_id}_{k}.pkl"), "wb") as f:
            pickle.dump(save_dict, f)

def log_initial_data(wandb_log, meas, psf, gt):
    """
    Log the initial measurement, PSF, and ground truth data to Wandb.

    Args:
        wandb_log (dict): Dictionary to store log data.
        meas (jnp.array): Measurement data.
        psf (jnp.array): Point spread function (PSF).
        gt (jnp.array): Ground truth image.
    """
    wandb_log = sdc.wandb_log_meas(wandb_log, meas)
    wandb_log = sdc.wandb_log_psf(wandb_log, psf)
    wandb_log = sdc.wandb_log_ground_truth(wandb_log, gt)
    wandb.log(wandb_log)

def log_intermediate_results(wandb_log, strategy, k, wavelengths, run_name):
    """
    Log intermediate results during the reconstruction process.

    Args:
        wandb_log (dict): Dictionary to store log data.
        strategy (Reconstruction): Reconstruction strategy object.
        k (int): Current iteration number.
        wavelengths (jnp.array): Array of wavelengths.
        run_name (str): Name of the current run.
    """
    xk = strategy.reconstruct()
    wandb_log = sdc.wandb_log_sim_meas(wandb_log, sdc.jax_forward_model(xk, m, hfftpad))
    wandb_log = sdc.wandb_log_false_color_recon(wandb_log, xk / jnp.max(xk) * jnp.sum(xk, 0)[None, ...], wavelengths)

    if isinstance(strategy, LowRankReconstruction):
        wandb_log = sdc.wandb_log_low_rank_components(wandb_log, strategy.U, wavelengths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some configuration and GPU index.")
    parser.add_argument("--gpu_index", type=int, required=True, help="Index of the GPU to use")
    parser.add_argument("--config_file_path", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Set up GPU for JAX
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

    # Load the configuration
    config = load_config(args.config_file_path)

    # Initialize wandb
    wbrun = wandb.init(project=config["wandb"]["project_name"], name=config["wandb"]["run_name"], config=config)
    run_id = wbrun.id

    # Initialize data (meas, psf, filter stack, etc.)
    meas, psf, m, xk, hfftpad = initialize_data(config)
    

    # Load ground truth image
    try:
        gt = sdc.importTiff(config["data"]["datafolder"], "gt.tiff")
        if gt.shape[-1] != 3:
            gt = gt / 2 ** config["measurement_processing"]["bits"]
            gt = jnp.expand_dims(gt, 0)
            gt = sdc.rotate_90(gt)
            gt = jnp.squeeze(gt)
        gt = gt / jnp.max(gt)
    except:
        print("No ground truth image found, continuing without ground truth")
        gt = jnp.zeros_like(meas)

    # Log initial data
    log_initial_data({}, meas, psf, gt)

    # Initialize wavelengths from config
    wavelengths = jnp.arange(config["measurement_processing"]["wvmin"],
                             config["measurement_processing"]["wvmax"] + config["measurement_processing"]["wvstep"],
                             config["measurement_processing"]["wvstep"])

    # Initialize SVD if using low rank
    if config["reconstruction"]["use_low_rank"]:
        U, V, W, Y, X = initialize_svd(xk, config["reconstruction"]["rank"])
    else:
        U = V = W = Y = X = None

    # Initialize the optimizer
    optimizer = optax.adam(learning_rate=config["reconstruction"]["step_size"])

    # Get the appropriate reconstruction strategy
    strategy = get_reconstruction_strategy(
        config["reconstruction"]["use_low_rank"],
        config["reconstruction"]["use_one_hot"],
        xk=xk, U=U, V=V, W=W, Y=Y, X=X, weights=None, temperature=1,
        optimizer=optimizer
    )

    #Check if we want to overwrite the same file or save iterations separately
    if config["wandb"].get("overwrite") == False:
        overwrite = False
    else:
        overwrite = True

    # Run the reconstruction process
    run_reconstruction(
        strategy, config["reconstruction"]["kmax"], config["wandb"]["save_frequency"],
        config["wandb"]["kprint"], meas, hfftpad, m, config["reconstruction"]["thr"],
        config["reconstruction"]["xytv"], config["reconstruction"]["lamtv"],
        W, Y, X, config["wandb"]["save_location"], wavelengths, config["wandb"]["run_name"], run_id, overwrite
    )
