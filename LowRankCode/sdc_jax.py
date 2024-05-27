# configuration file

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import csv
from ipywidgets import IntProgress
from IPython.display import display
from skimage import io
import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
import jax.numpy as jnp
import jax.lax

dtype = torch.float32


def find_max_pixel(x):
    return (torch.sum(x, 0) == torch.max(torch.sum(x, 0))).nonzero().squeeze()


def get_split_idx(num_splits, split, target, dim):
    leng = target.shape[dim]
    split_size = int(np.floor(leng / num_splits))
    return split_size * split, split_size * (split + 1)


def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)
    xp = xp.unsqueeze(1)
    m = (fp[1:, :] - fp[:-1, :]) / (xp[1:, :] - xp[:-1, :])
    b = fp[:-1, :] - (m.mul(xp[:-1, :]))
    indices = (torch.sum(torch.ge(x[:, None, :], xp[None, :, :]), -2) - 1).clamp(
        0, xp.shape[0] - 1
    )
    return m[indices.squeeze(), :] * x + b[indices.squeeze(), :]


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


def sumFilterArray(filterstack, wv, wvmin, wvmax, wvstep):
    # filterstack is ndarray.  wvmin, wvmax, wvstep are scalers
    # returns new ndarray msum with dimensions yx same as filterstack, and lambda depending on length of wvnew
    wvnew = torch.arange(wvmin, wvmax + wvstep, wvstep)
    # resample the filterarray
    # find where in wvindex meets wvnew
    j0 = torch.where(wvnew[0] == wv)[0][0]
    (dim0, dim1, dim2) = filterstack.shape
    dim0 = len(range(len(wvnew)))
    msum = torch.zeros((dim0, dim1, dim2))

    for k in range(dim0):
        # get index according to wv
        # sum and add
        if k < dim0 - 1:
            j1 = np.where(wvnew[k + 1] == wv)[0][0]
        else:
            j0 = np.where(wvmax == wv)[0][0]  # handle the last index
            j1 = np.where(wvmax + wvstep == wv)[0][0]
        msum[k, :, :] = torch.sum(filterstack[j0:j1, :, :], 0)
        j0 = j1
    return msum


# import specific tiff file from directory
# inputs - datafolder (string): directory, fname (string): specific file
# outputs - imarray (numpy array) containing tiff file as image
def importTiff(datafolder, fname):

    im = io.imread(os.path.join(datafolder, fname)).astype(float)
    imarray = torch.tensor(im, dtype=dtype)
    return imarray


# import all tiff files from a directory
# inputs - path (string): directory
# outputs - imageStack (torch array): stack of tiff images along dim 2
def tif_loader(path):
    fnames = [fname for fname in os.listdir(path) if fname.endswith(".tiff")]
    fnames.sort()
    prog = IntProgress(min=0, max=len(fnames))  # instantiate the bar
    display(prog)  # display the bar
    for ii in range(len(fnames)):
        prog.value += 1
        file = fnames[ii]
        im = io.imread(os.path.join(path, file)).astype(float)
        #         im = np.asarray(im).astype(float)
        if ii == 0:
            imageStack = torch.zeros((len(fnames), im.shape[0], im.shape[1]))
        imageStack[ii, :, :] = torch.tensor(im, dtype=dtype)
    return imageStack


def cropci(im, ci):
    if len(im.shape) == 3:
        return im[:, ci[0] : ci[1], ci[2] : ci[3]]
    elif len(im.shape) == 2:
        return im[ci[0] : ci[1], ci[2] : ci[3]]
    else:
        print("not an image")
        return


def resample(psf, oldpix=1.67, newpix=5.3):
    zoom = oldpix / newpix
    s = psf.shape
    newsize = (int(s[1] * zoom), int(s[0] * zoom))  # flip to x,y
    pilpsf = Image.fromarray(psf)
    pilpsfzoom = pilpsf.resize(newsize)
    psf0 = np.array(pilpsfzoom)
    return psf0


def loadspectrum(path):
    file = open(path)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
        #     print(row)
        rows.append(row)
    file.close()
    spec = rows[32:-1]
    wavelength = []
    intensity = []
    for ii in spec:
        vals = ii[0].split(";")
        wavelength.append(float(vals[0]))
        intensity.append(float(vals[1]))
    return (wavelength, intensity)


def HSI2RGB_jax(wY, HSI, ydim, xdim, d, threshold):
    # wY: wavelengths in nm
    # Y : HSI as a (#pixels x #bands) matrix,
    # dims: x & y dimension of image
    # d: 50, 55, 65, 75, determines the illuminant used, if in doubt use d65
    # thresholdRGB : True if thesholding should be done to increase contrast
    #
    #
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
    #
    #

    # Load reference illuminant
    D = spio.loadmat(
        "/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope/D_illuminants.mat"
    )
    w = D["wxyz"][:, 0]
    x = D["wxyz"][:, 1]
    y = D["wxyz"][:, 2]
    z = D["wxyz"][:, 3]
    D = D["D"]

    i = {50: 2, 55: 3, 65: 1, 75: 4}
    wI = D[:, 0]
    I = D[:, i[d]]
    # Changed I for flouresence imaging
    I[:] = 100

    # Interpolate to image wavelengths
    I = PchipInterpolator(wI, I, extrapolate=True)(
        wY
    )  # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w, x, extrapolate=True)(
        wY
    )  # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w, y, extrapolate=True)(
        wY
    )  # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w, z, extrapolate=True)(
        wY
    )  # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i = bisect(wY, 800)
    HSI = HSI[:, 0:i] / HSI.max()
    wY = wY[:i]
    I = I[:i]
    x = x[:i]
    y = y[:i]
    z = z[:i]

    # Compute k
    k = 1 / jnp.trapezoid(y * I, wY)

    # Compute X,Y & Z for image
    X = k * jnp.trapezoid(HSI @ jnp.diag(I * x), wY, axis=1)
    Z = k * jnp.trapezoid(HSI @ jnp.diag(I * z), wY, axis=1)
    Y = k * jnp.trapezoid(HSI @ jnp.diag(I * y), wY, axis=1)

    XYZ = jnp.array([X, Y, Z])

    # Convert to RGB
    M = jnp.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )

    sRGB = M @ XYZ

    # Gamma correction
    gamma_map = sRGB > 0.0031308
    sRGB = sRGB.at[gamma_map].set(
        1.055 * jnp.power(sRGB[gamma_map], (1.0 / 2.4)) - 0.055
    )
    sRGB = sRGB.at[jnp.invert(gamma_map)].set(12.92 * sRGB[jnp.invert(gamma_map)])
    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB = sRGB.at[sRGB > 1].set(1)
    sRGB = sRGB.at[sRGB < 0].set(0)

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
    y_margin = (mat.shape[-2] - target_shape[-2]) // 2
    x_margin = (mat.shape[-1] - target_shape[-1]) // 2
    if mat.ndim == 2:
        return mat[y_margin : -y_margin or None, x_margin : -x_margin or None]
    elif mat.ndim == 3:
        return mat[:, y_margin : -y_margin or None, x_margin : -x_margin or None]
    else:
        raise ValueError("crop2D only supports 2D and 3D arrays")


def jax_forward_model(object, spectral_filter, padded_fft_psf):
    paddings = (
        (0, 0, 0),
        (
            np.ceil(object.shape[1] / 2).astype(int),
            np.floor(object.shape[1] / 2).astype(int),
            0,
        ),
        (
            np.ceil(object.shape[2] / 2).astype(int),
            np.floor(object.shape[2] / 2).astype(int),
            0,
        ),
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
    y = measurement[None, ...] * spectral_filter
    ypad = jax.lax.pad(y, 0.0, padding)
    yfftpad = jnp.fft.fft2(ypad)
    x = jnp.fft.ifftshift(
        jnp.fft.ifft2(yfftpad * jnp.conj(padded_fft_psf)), axes=(1, 2)
    )
    x = abs(jax_crop2D(measurement.shape, x))
    return x.clip(0)


def low_rank_reconstruction(U, V):
    return U @ V


def low_rank_loss(U, V, meas, padded_psf_fft, filter_array, thr, xytv, lamtv):
    # Reconstruct xk from U and V
    xk = low_rank_reconstruction(U, V).reshape(
        filter_array.shape[0], filter_array.shape[1], filter_array.shape[2]
    )

    # calculate the forward model
    sim_meas = jax_forward_model(xk, filter_array, padded_psf_fft)

    # calculate the wavelength gradients
    dlam = jnp.gradient(U, axis=0)
    ddlam = jnp.gradient(dlam, axis=0)

    # calculate the x and y gradients
    dy, dx = jnp.gradient(xk, axis=(-1, -2))

    # calculate the loss
    lamtv_loss = jnp.linalg.norm(ddlam, 1)
    data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
    tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
    sparsity_loss = jnp.linalg.norm(V.ravel(), 1)
    loss = data_loss + lamtv * lamtv_loss + xytv * tv_loss + thr * sparsity_loss
    return loss


def one_hot_reconstruction(U, V, weights, temperature):
    # apply a softmax to V with temperatur
    out = jax.nn.softmax(V / temperature, axis=0)
    out = out * weights
    return U @ out


def one_hot_loss(
    U, V, weights, meas, padded_psf_fft, filter_array, thr, xytv, lamtv, temperature
):
    # Reconstruct xk from U and V
    xk = one_hot_reconstruction(U, V, weights, temperature).reshape(
        filter_array.shape[0], filter_array.shape[1], filter_array.shape[2]
    )

    # calculate the forward model
    sim_meas = jax_forward_model(xk, filter_array, padded_psf_fft)

    # calculate the wavelength gradients
    dlam = jnp.gradient(U, axis=0)
    ddlam = jnp.gradient(dlam, axis=0)

    # calculate the x and y gradients
    dy, dx = jnp.gradient(xk, axis=(-1, -2))

    # calculate the loss
    lamtv_loss = jnp.linalg.norm(ddlam, 1)
    data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
    tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
    sparsity_loss = jnp.linalg.norm(V.ravel(), 1)
    loss = data_loss + lamtv * lamtv_loss + xytv * tv_loss + thr * sparsity_loss
    return loss


# define a loss function
def loss(xk, meas, hfftpad, m, thr, xytv, lamtv):
    # calculate the forward model
    sim_meas = jax_forward_model(xk, m, hfftpad)

    # calculate the x, y, and lambda gradients
    dlam, dy, dx = jnp.gradient(xk, axis=(0, 1, 2))
    ddlam = jnp.gradient(dlam, axis=0)

    # calculate the data loss
    data_loss = jnp.linalg.norm((sim_meas - meas).ravel(), 2) ** 2
    # calculate the xy total variation loss
    tv_loss = jnp.linalg.norm(dx.ravel(), 1) + jnp.linalg.norm(dy.ravel(), 1)
    # calculate the sparsity loss
    sparsity_loss = jnp.linalg.norm(xk.ravel(), 1)
    # calculate the lambda total variation loss
    lamtv_loss = jnp.linalg.norm(ddlam.ravel(), 2) ** 2
    # calculate the total loss
    loss = data_loss + xytv * tv_loss + lamtv * lamtv_loss + thr * sparsity_loss
    return loss


def bw_visualize(image, title="", cmap="gray", colorbar=False, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    if title != "":
        plt.title(title)
    if colorbar:
        plt.colorbar()
    # plt.axis('off')
    plt.show()


def color_visualize(image, wavelengths, title="", figsize=(10, 10)):
    # Spectral ranges for averaging (in nm) for Blue, Green, Red channels
    spectral_ranges = [(450, 495), (495, 570), (620, 750)]

    # Select and average the bands
    rgb_image = select_and_average_bands(image, wavelengths, spectral_ranges)

    # Normalize the RGB image to enhance contrast if necessary
    rgb_image = (rgb_image - np.min(rgb_image)) / (
        np.max(rgb_image) - np.min(rgb_image)
    )

    # Display the image
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_image)
    plt.title("False Color RGB Visualization with Averaged Bands")
    plt.axis("off")  # Hide axes for better visualization
    plt.show()


def wandb_log_meas(wandb_log, meas):
    # plot the lenslet positions
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the measurement
    ax.imshow(meas, cmap="gray", vmin=0, vmax=1)
    # turn off the axis
    ax.axis("off")
    # add plot to log dictionary
    wandb_log["experimental_measurement"] = fig
    # close the figure
    plt.close()
    return wandb_log


def wandb_log_sim_meas(wandb_log, meas):
    # make the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the measurement
    ax.imshow(meas, cmap="gray", vmin=0, vmax=1)
    # turn off the axis
    ax.axis("off")
    # add plot to log dictionary
    wandb_log["simulated_measurement"] = fig
    # close the figure
    plt.close()
    return wandb_log


def wandb_log_false_color_recon(wandb_log, recon, wavelengths):
    # Create false color filter
    HSI_data = jnp.transpose(recon, (1, 2, 0))
    HSI_data = jnp.reshape(HSI_data, [-1, recon.shape[0]])
    false_color_image = HSI2RGB_jax(
        wavelengths, HSI_data, recon.shape[1], recon.shape[2], 65, False
    )

    # make the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the false color recon
    ax.imshow(false_color_image**0.6)
    # turn off the axis
    ax.axis("off")
    # add plot to log dictionary
    wandb_log["false_color_recon"] = fig
    # close the figure
    plt.close()
    return wandb_log


def wandb_log_psf(wandb_log, psf):
    # make the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the psf
    ax.imshow(psf, cmap="gray")
    # turn off the axis
    ax.axis("off")
    # add plot to log dictionary
    wandb_log["psf"] = fig
    # close the figure
    plt.close()
    return wandb_log


def wandb_log_ground_truth(wandb_log, gt):

    # make the graph
    fig, ax = plt.subplots(figsize=(10, 10))
    # plot the psf
    if gt.shape[-1] != 3:
        ax.imshow(gt, cmap="gray")
    else:
        ax.imshow(gt)
    # turn off the axis
    ax.axis("off")
    # add plot to log dictionary
    wandb_log["ground_truth"] = fig
    # close the figure
    plt.close()
    return wandb_log


def wandb_log_low_rank_components(wandb_log, U, wavelengths):
    U = U / jnp.max(U)
    # make the graph
    fig, ax = plt.subplots()
    # plot each component
    for ii in range(U.shape[-1]):
        ax.plot(wavelengths, U[:, ii])
    # add title
    ax.set_title("Low Rank Components")
    # add x label
    ax.set_xlabel("Wavelength (nm)")
    # add y label
    ax.set_ylabel("Intensity (a.u.)")
    # add legend
    ax.legend([f"Component {ii}" for ii in range(U.shape[-1])])
    # add plot to log dictionary
    wandb_log["low_rank_components"] = fig
    # close the figure
    plt.close()
    return wandb_log
