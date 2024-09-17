# configuration file

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import csv
from ipywidgets import IntProgress
from IPython.display import display
from skimage import io
import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
from scipy.ndimage import rotate

import jax.numpy as jnp
import jax.lax

dtype = float


def center_crop(array, crop_size):
    """
    Center crop the input array to the given crop size. 
    If the crop size is larger than the array size, the array will be zero-padded.

    Args:
        array (np.array): Input array.
        crop_size (tuple): Target crop size (height, width).

    Returns:
        np.array: Center-cropped or zero-padded array.
    """
    array_shape = array.shape
    crop_height, crop_width = crop_size
    
    # If the crop size is larger, pad with zeros
    if crop_height > array_shape[0] or crop_width > array_shape[1]:
        pad_height = max(crop_height - array_shape[0], 0)
        pad_width = max(crop_width - array_shape[1], 0)
        
        # Calculate padding for each side
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # Apply zero padding
        array = np.pad(array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)

    # Calculate crop start and end positions
    start_y = (array.shape[0] - crop_height) // 2
    start_x = (array.shape[1] - crop_width) // 2
    end_y = start_y + crop_height
    end_x = start_x + crop_width

    return array[start_y:end_y, start_x:end_x]




def rotate_90(image, angle=-90):
    """Rotate image by -90 degrees using scipy."""
    return rotate(image, angle, axes=(1, 2), reshape=False)

def rotate_90(image, angle=-90):
    """
    Rotate image by -90 degrees using scipy.

    Args:
        image (np.array): Input image.
        angle (int): Rotation angle, default is -90 degrees.

    Returns:
        np.array: Rotated image.
    """
    return rotate(image, angle, axes=(1, 2), reshape=False)

def find_max_pixel(x):
    """
    Find the index of the pixel where the sum across channels is maximized.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Index of the pixel with the maximum sum across channels.
    """
    summed = np.sum(x, axis=0)
    max_value = np.max(summed)
    return np.squeeze(np.nonzero(summed == max_value))


def get_split_idx(num_splits, split, target, dim):
    """
    Calculate the start and end indices for splitting along a dimension.

    Args:
        num_splits (int): Total number of splits.
        split (int): Current split index.
        target (np.array): Array to be split.
        dim (int): Dimension to split along.

    Returns:
        tuple: Start and end indices for the current split.
    """
    leng = target.shape[dim]
    split_size = int(np.floor(leng / num_splits))
    return split_size * split, split_size * (split + 1)

def interpolate(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Piecewise linear interpolation.

    Args:
        x (np.ndarray): Interpolation points.
        xp (np.ndarray): Known points.
        fp (np.ndarray): Function values at known points.

    Returns:
        np.ndarray: Interpolated values.
    """
    x = np.expand_dims(x, axis=1)
    xp = np.expand_dims(xp, axis=1)
    
    # Calculate slope (m) and intercept (b)
    m = (fp[1:, :] - fp[:-1, :]) / (xp[1:, :] - xp[:-1, :])
    b = fp[:-1, :] - (m * xp[:-1, :])
    
    # Find indices where x is greater than or equal to xp, clamping values to ensure valid indices
    indices = np.sum(x[:, None, :] >= xp[None, :, :], axis=-2) - 1
    indices = np.clip(indices, 0, xp.shape[0] - 1)

    # Return the interpolated values
    return m[indices.squeeze(), :] * x + b[indices.squeeze(), :]


def piecewise_linear(x, x0, y0, k1, k2):
    """
    Perform piecewise linear interpolation with different slopes on each side.

    Args:
        x (np.array): Input array.
        x0 (float): Transition point.
        y0 (float): Value at the transition point.
        k1 (float): Slope for the left side.
        k2 (float): Slope for the right side.

    Returns:
        np.array: Interpolated values.
    """
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )

def sumFilterArray(filterstack, wv, wvmin, wvmax, wvstep):
    """
    Resample and sum the filter stack for specific wavelength intervals.

    Args:
        filterstack (np.array): Input filter stack.
        wv (np.array): Wavelength array.
        wvmin (float): Minimum wavelength.
        wvmax (float): Maximum wavelength.
        wvstep (float): Wavelength step.

    Returns:
        np.array: Resampled and summed filter stack.
    """
    # Create new wavelength array
    wvnew = np.arange(wvmin, wvmax + wvstep, wvstep)
    
    # Find the first matching index in the wavelength array
    j0 = np.where(wvnew[0] == wv)[0][0]
    
    # Get dimensions of the filter stack
    dim0, dim1, dim2 = filterstack.shape
    
    # Initialize an empty array to hold the summed filter stack
    msum = np.zeros((len(wvnew), dim1, dim2))

    # Resample and sum filterstack based on the wavelength intervals
    for k in range(len(wvnew)):
        if k < len(wvnew) - 1:
            j1 = np.where(wvnew[k + 1] == wv)[0][0]
        else:
            j0 = np.where(wvmax == wv)[0][0]  # Handle the last index
            j1 = np.where(wvmax + wvstep == wv)[0][0]
        
        # Sum the filterstack over the specified wavelength range
        msum[k, :, :] = np.sum(filterstack[j0:j1, :, :], axis=0)
        j0 = j1

    return msum

def importTiff(datafolder, fname):
    """
    Import a specific tiff file from a directory.

    Args:
        datafolder (str): Directory path.
        fname (str): File name.

    Returns:
        np.array: Imported image as a numpy array.
    """
    im = io.imread(os.path.join(datafolder, fname)).astype(float)
    return np.asarray(im).astype(float)


def tif_loader(path):
    """
    Import all tiff files from a directory.

    Args:
        path (str): Directory path.

    Returns:
        np.array: Stack of tiff images along dimension 2.
    """
    fnames = sorted([fname for fname in os.listdir(path) if fname.endswith(".tiff")])
    prog = IntProgress(min=0, max=len(fnames))
    display(prog)
    
    for ii, file in enumerate(fnames):
        prog.value += 1
        im = io.imread(os.path.join(path, file)).astype(float)
        if ii == 0:
            imageStack = np.zeros((len(fnames), im.shape[0], im.shape[1]))
        imageStack[ii, :, :] = im

    return imageStack


def cropci(im, ci):
    """
    Crop an image based on the provided crop indices.

    Args:
        im (np.array): Input image array.
        ci (tuple): Crop indices (y1, y2, x1, x2).

    Returns:
        np.array: Cropped image.
    """
    if len(im.shape) == 3:
        return im[:, ci[0]:ci[1], ci[2]:ci[3]]
    elif len(im.shape) == 2:
        return im[ci[0]:ci[1], ci[2]:ci[3]]
    else:
        print("Not an image.")
        return

def resample(psf, oldpix=1.67, newpix=5.3):
    """
    Resample the PSF to a new pixel size.

    Args:
        psf (np.array): Input PSF.
        oldpix (float): Original pixel size.
        newpix (float): New pixel size.

    Returns:
        np.array: Resampled PSF.
    """
    zoom = oldpix / newpix
    newsize = (int(psf.shape[1] * zoom), int(psf.shape[0] * zoom))
    return np.array(Image.fromarray(psf).resize(newsize))


def loadspectrum(path):
    """
    Load a spectrum from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        tuple: Wavelengths and intensities.
    """
    with open(path) as file:
        csvreader = csv.reader(file)
        next(csvreader)  
        rows = [row[0].split(";") for row in csvreader]
        wavelengths = [float(row[0]) for row in rows[32:-1]]
        intensities = [float(row[1]) for row in rows[32:-1]]

    return wavelengths, intensities

    # If you use this method, please cite the following paper:
    #  M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson,
    #  H. Deborah and J. R. Sveinsson,
    #  "Creating RGB Images from Hyperspectral Images Using a Color Matching Function",
    #  IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020
    #
    #  @INPROCEEDINGS{hsi2rgb,
    #  author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson}
    #  and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}},
    #  booktitle={IEEE International Geoscience and Remote Sensing Symposium},
    #  title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function},
    #  year={2020}, volume={}, number={}, pages={}}
    #
    # Paper is available at
    # https://www.researchgate.net/profile/Jakob_Sigurdsson
def HSI2RGB_jax(wY, HSI, ydim, xdim, d, threshold):
    """
    Convert Hyperspectral Image (HSI) to RGB using a color matching function.

    Args:
        wY (np.array): Wavelengths in nm.
        HSI (np.array): Hyperspectral image matrix (#pixels x #bands).
        ydim (int): Image y-dimension.
        xdim (int): Image x-dimension.
        d (int): Determines the illuminant used (50, 55, 65, 75).
        threshold (float): Threshold for RGB contrast adjustment.

    Returns:
        np.array: RGB image.
    """
    # Load reference illuminant data
    D = spio.loadmat("/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope/D_illuminants.mat")
    w = D["wxyz"][:, 0]
    x = D["wxyz"][:, 1]
    y = D["wxyz"][:, 2]
    z = D["wxyz"][:, 3]
    D = D["D"]

    illuminant_indices = {50: 2, 55: 3, 65: 1, 75: 4}
    wI = D[:, 0]
    I = D[:, illuminant_indices[d]]
    I[:] = 100  # Adjust for fluorescence imaging

    # Interpolate wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(wY)
    x = PchipInterpolator(w, x, extrapolate=True)(wY)
    y = PchipInterpolator(w, y, extrapolate=True)(wY)
    z = PchipInterpolator(w, z, extrapolate=True)(wY)

    # Truncate at 780 nm
    i = bisect(wY, 800)
    HSI = HSI[:, :i] / HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute X, Y, Z for the image
    k = 1 / jnp.trapezoid(y * I, wY)
    X = k * jnp.trapezoid(HSI @ jnp.diag(I * x), wY, axis=1)
    Y = k * jnp.trapezoid(HSI @ jnp.diag(I * y), wY, axis=1)
    Z = k * jnp.trapezoid(HSI @ jnp.diag(I * z), wY, axis=1)

    XYZ = jnp.array([X, Y, Z])

    # Convert to RGB
    M = jnp.array([[3.2404542, -1.5371385, -0.4985314],
                   [-0.9692660, 1.8760108, 0.0415560],
                   [0.0556434, -0.2040259, 1.0572252]])

    sRGB = M @ XYZ

    # Apply gamma correction
    gamma_map = sRGB > 0.0031308
    sRGB = sRGB.at[gamma_map].set(1.055 * jnp.power(sRGB[gamma_map], (1.0 / 2.4)) - 0.055)
    sRGB = sRGB.at[jnp.invert(gamma_map)].set(12.92 * sRGB[jnp.invert(gamma_map)])
    sRGB = sRGB.at[sRGB > 1].set(1)
    sRGB = sRGB.at[sRGB < 0].set(0)

    # Apply thresholding if necessary
    if threshold:
        for idx in range(3):
            y = sRGB[idx, :]
            a, b = jnp.histogram(y, 100)
            b = b[:-1] + jnp.diff(b) / 2
            a = jnp.cumsum(a) / jnp.sum(a)
            th = b[0]
            i = a < threshold
            if i.any():
                th = b[i][-1]
            y = y - th
            y = y.at[y < 0].set(0)

            a, b = jnp.histogram(y, 100)
            b = b[:-1] + jnp.diff(b) / 2
            a = jnp.cumsum(a) / jnp.sum(a)
            i = a > 1 - threshold
            th = b[i][0]
            y = y.at[y > th].set(th)
            y = y / th
            sRGB = sRGB.at[idx, :].set(y)

    R = jnp.reshape(sRGB[0, :], [ydim, xdim])
    G = jnp.reshape(sRGB[1, :], [ydim, xdim])
    B = jnp.reshape(sRGB[2, :], [ydim, xdim])
    return jnp.transpose(jnp.array([R, G, B]), [1, 2, 0])

def jax_crop2D(target_shape, mat):
    """
    Center crop the 2D or 3D input matrix to the target shape.

    Args:
        target_shape (tuple): Target shape for cropping.
        mat (np.array): Input matrix.

    Returns:
        np.array: Cropped matrix.
    """
    y_margin = (mat.shape[-2] - target_shape[-2]) // 2
    x_margin = (mat.shape[-1] - target_shape[-1]) // 2
    if mat.ndim == 2:
        return mat[y_margin : -y_margin or None, x_margin : -xmargin or None]
    elif mat.ndim == 3:
        return mat[:, y_margin : -y_margin or None, x_margin : -x_margin or None]
    else:
        raise ValueError("jax_crop2D only supports 2D and 3D arrays.")


def jax_forward_model(object, spectral_filter, padded_fft_psf):
    """
    Forward model for hyperspectral imaging.

    Args:
        object (np.array): Input object.
        spectral_filter (np.array): Spectral filter.
        padded_fft_psf (np.array): Padded FFT of the PSF.

    Returns:
        np.array: Measurement data.
    """
    paddings = (
        (0, 0, 0),
        (np.ceil(object.shape[1] / 2).astype(int), np.floor(object.shape[1] / 2).astype(int), 0),
        (np.ceil(object.shape[2] / 2).astype(int), np.floor(object.shape[2] / 2).astype(int), 0),
    )
    padded_object = jax.lax.pad(object, 0.0, paddings)
    fft_object = jnp.fft.fft2(padded_object)
    fft_product = padded_fft_psf * fft_object

    ifft_product = jnp.fft.ifftshift(jnp.fft.ifft2(fft_product), axes=(1, 2))
    ifft_product = abs(jax_crop2D(object.shape, ifft_product))
    ifft_product = ifft_product * spectral_filter
    measurement = jnp.sum(ifft_product, axis=0)
    return measurement.clip(0)


def jax_adjoint_model(measurement, spectral_filter, padded_fft_psf, padding):
    """
    Adjoint model for hyperspectral imaging.

    Args:
        measurement (np.array): Input measurement data.
        spectral_filter (np.array): Spectral filter.
        padded_fft_psf (np.array): Padded FFT of the PSF.
        padding (tuple): Padding dimensions.

    Returns:
        np.array: Reconstructed image.
    """
    y = measurement[None, ...] * spectral_filter
    ypad = jax.lax.pad(y, 0.0, padding)
    yfftpad = jnp.fft.fft2(ypad)
    x = jnp.fft.ifftshift(jnp.fft.ifft2(yfftpad * jnp.conj(padded_fft_psf)), axes=(1, 2))
    x = abs(jax_crop2D(measurement.shape, x))
    return x.clip(0)


def low_rank_reconstruction(U, V):
    """
    Reconstruct image from low-rank matrices U and V.

    Args:
        U (np.array): Left singular matrix.
        V (np.array): Right singular matrix.

    Returns:
        np.array: Reconstructed image.
    """
    return U @ V




def one_hot_reconstruction(U, V, weights, temperature):
    """
    Reconstruct image using one-hot encoding.

    Args:
        U (np.array): Left singular matrix.
        V (np.array): Right singular matrix.
        weights (np.array): Weight matrix.
        temperature (float): Softmax temperature.

    Returns:
        np.array: Reconstructed image.
    """
    out = jax.nn.softmax(V / temperature, axis=0) * weights
    return U @ out

def bw_visualize(image, title="", cmap="gray", colorbar=False, figsize=(10, 10)):
    """
    Display a black and white image.

    Args:
        image (np.array): Input image.
        title (str): Title of the plot.
        cmap (str): Colormap to use.
        colorbar (bool): Whether to include a colorbar.
        figsize (tuple): Figure size.

    Returns:
        None
    """
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    if colorbar:
        plt.colorbar()
    plt.show()


def color_visualize(image, wavelengths, title="", figsize=(10, 10)):
    """
    Display a color image based on spectral ranges.

    Args:
        image (np.array): Input image.
        wavelengths (np.array): Wavelengths associated with the image bands.
        title (str): Title of the plot.
        figsize (tuple): Figure size.

    Returns:
        None
    """
    spectral_ranges = [(450, 495), (495, 570), (620, 750)]
    rgb_image = select_and_average_bands(image, wavelengths, spectral_ranges)
    rgb_image = (rgb_image - np.min(rgb_image)) / (np.max(rgb_image) - np.min(rgb_image))

    plt.figure(figsize=figsize)
    plt.imshow(rgb_image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()


def wandb_log_meas(wandb_log, meas):
    """
    Log the experimental measurement to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        meas (np.array): Measurement data.

    Returns:
        dict: Updated log dictionary.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(meas, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    wandb_log["experimental_measurement"] = fig
    plt.close(fig)
    return wandb_log


def wandb_log_sim_meas(wandb_log, meas):
    """
    Log the simulated measurement to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        meas (np.array): Simulated measurement data.

    Returns:
        dict: Updated log dictionary.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(meas, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    wandb_log["simulated_measurement"] = fig
    plt.close(fig)
    return wandb_log


def wandb_log_false_color_recon(wandb_log, recon, wavelengths):
    """
    Log the false color reconstruction to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        recon (np.array): Reconstructed image.
        wavelengths (np.array): Wavelengths associated with the image.

    Returns:
        dict: Updated log dictionary.
    """
    HSI_data = jnp.transpose(recon, (1, 2, 0)).reshape(-1, recon.shape[0])
    false_color_image = HSI2RGB_jax(wavelengths, HSI_data, recon.shape[1], recon.shape[2], 65, False)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(false_color_image ** 0.6)
    ax.axis("off")
    wandb_log["false_color_recon"] = fig
    plt.close(fig)
    return wandb_log


def wandb_log_psf(wandb_log, psf):
    """
    Log the PSF to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        psf (np.array): Point spread function (PSF).

    Returns:
        dict: Updated log dictionary.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(psf, cmap="gray")
    ax.axis("off")
    wandb_log["psf"] = fig
    plt.close(fig)
    return wandb_log


def wandb_log_ground_truth(wandb_log, gt):
    """
    Log the ground truth image to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        gt (np.array): Ground truth image.

    Returns:
        dict: Updated log dictionary.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(gt if gt.shape[-1] == 3 else gt, cmap="gray" if gt.shape[-1] != 3 else None)
    ax.axis("off")
    wandb_log["ground_truth"] = fig
    plt.close(fig)
    return wandb_log


def wandb_log_low_rank_components(wandb_log, U, wavelengths):
    """
    Log the low-rank components to wandb.

    Args:
        wandb_log (dict): Wandb log dictionary.
        U (np.array): Low-rank matrix U.
        wavelengths (np.array): Wavelengths associated with the image.

    Returns:
        dict: Updated log dictionary.
    """
    U = U / jnp.max(U)

    fig, ax = plt.subplots()
    for ii in range(U.shape[-1]):
        ax.plot(wavelengths, U[:, ii])
    ax.set_title("Low Rank Components")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.legend([f"Component {ii}" for ii in range(U.shape[-1])])
    wandb_log["low_rank_components"] = fig
    plt.close(fig)
    return wandb_log