from matplotlib import pyplot as plt
import sdc_config as sdc
import jax.numpy as jnp

def wandb_log_meas(wandb_log, meas):
    # plot the lenslet positions
    fig, ax = plt.subplots(figsize=(10,10))
    # plot the measurement
    ax.imshow(meas, cmap='gray', vmin=0, vmax=1)
    # turn off the axis
    ax.axis('off')
    # add plot to log dictionary
    wandb_log['experimental_measurement'] = fig
    # close the figure
    plt.close()
    return wandb_log

def wandb_log_sim_meas(wandb_log, meas):
    # make the graph
    fig, ax = plt.subplots(figsize=(10,10))
    # plot the measurement
    ax.imshow(meas, cmap='gray', vmin=0, vmax=1)
    # turn off the axis
    ax.axis('off')
    # add plot to log dictionary
    wandb_log['simulated_measurement'] = fig
    # close the figure
    plt.close()
    return wandb_log

def wandb_log_false_color_recon(wandb_log, recon, wavelengths):
    # Create false color filter
    HSI_data = jnp.transpose(recon, (1, 2, 0))
    HSI_data = jnp.reshape(HSI_data, [-1, recon.shape[0]])
    false_color_image = sdc.HSI2RGB_jax(wavelengths, HSI_data , recon.shape[1], recon.shape[2], 65, False)

    # make the graph
    fig, ax = plt.subplots(figsize=(10,10))
    # plot the false color recon
    ax.imshow(false_color_image**.6)
    # turn off the axis
    ax.axis('off')
    # add plot to log dictionary
    wandb_log['false_color_recon'] = fig
    # close the figure
    plt.close()
    return wandb_log

def wandb_log_psf(wandb_log, psf):
    # make the graph
    fig, ax = plt.subplots(figsize=(10,10))
    # plot the psf
    ax.imshow(psf, cmap='gray')
    # turn off the axis
    ax.axis('off')
    # add plot to log dictionary
    wandb_log['psf'] = fig
    # close the figure
    plt.close()
    return wandb_log

def wandb_log_ground_truth(wandb_log, gt):
    # make the graph
    fig, ax = plt.subplots(figsize=(10,10))
    # plot the psf
    ax.imshow(gt, cmap='gray')
    # turn off the axis
    ax.axis('off')
    # add plot to log dictionary
    wandb_log['ground_truth'] = fig
    # close the figure
    plt.close()
    return wandb_log
