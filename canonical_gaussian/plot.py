import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import torch.distributions as D
import math
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import numpy as np 
# -- plotting -- 
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from os.path import exists
import torch.distributions as D
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import hydra
from copy import deepcopy
from hydra import compose, initialize
from omegaconf import DictConfig
import random
import pandas as pd
from setup import setup
from modules.dense import DenseEncoder1Layer, LinearEncoder
from utils import log_pos_dens_at, MyGetJacobian, fKL2, rKL2
plt.style.use('science')
matplotlib.rcParams.update({'font.size': 30}) 


def fixed_sigma_width_plot_lin(cfg, this_sigma, widths, theta_grid, bin_width, test_theta, test_x, **kwargs):
    all_losses = []
    all_test_losses = []
    all_angle_estimates = []
    all_true_angles = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, 1e-4, w)
        try:
            losses = np.load('{}/canonical_gaussian/{}/losses.npy'.format(cfg.log_dir, access_string))
            all_losses.append(losses)

            test_losses = np.load('{}/canonical_gaussian/{}/test_losses.npy'.format(cfg.log_dir, access_string))
            all_test_losses.append(test_losses)
        except:
            pass

        try:
            angle_estimates = np.load('{}/canonical_gaussian/{}/angle_estimates.npy'.format(cfg.log_dir, access_string))
            all_angle_estimates.append(angle_estimates)
        except:
            pass

        true_angles = np.load('{}/canonical_gaussian/{}/true_angles.npy'.format(cfg.log_dir, access_string))

        all_true_angles.append(true_angles)

    all_losses_lin = []
    all_test_losses_lin = []
    all_angle_estimates_lin = []
    all_true_angles_lin = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, 1e-4, w)
        try:
            losses = np.load('{}/canonical_gaussian/{}/losses_lin.npy'.format(cfg.log_dir, access_string))
            all_losses_lin.append(losses)

            test_losses = np.load('{}/canonical_gaussian/{}/test_losses_lin.npy'.format(cfg.log_dir, access_string))
            all_test_losses_lin.append(test_losses)
        except:
            pass

        try:
            angle_estimates = np.load('{}/canonical_gaussian/{}/angle_estimates_lin.npy'.format(cfg.log_dir, access_string))
            all_angle_estimates_lin.append(angle_estimates)
        except:
            pass

        true_angles = np.load('{}/canonical_gaussian/{}/true_angles_lin.npy'.format(cfg.log_dir, access_string))

        all_true_angles_lin.append(true_angles)

    # Get vline for true posterior
    log_posterior_ground_truths = log_pos_dens_at(test_theta, theta_grid, test_x, bin_width, **kwargs)
    mean_posterior_nll = -1*log_posterior_ground_truths.mean()

    # Plot training losses
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        ax.plot(np.arange(len(all_losses[j])), gaussian_filter1d(all_losses[j], 10.), label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(np.arange(len(all_losses_lin[j])), gaussian_filter1d(all_losses_lin[j], 10.), label='{}-lin'.format(widths[j]), linewidth=2.0)
    
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.ylim(0.70, 0.75)
    plt.savefig('./canonical_gaussian/figures/sigma={},width={},losses_lin.png'.format(this_sigma, widths[0]))

    # Plot testing losses
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        if j == len(all_losses)-1:
            ax.plot(5000*np.arange(len(all_test_losses[j])), all_test_losses[j], label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(5000*np.arange(len(all_test_losses_lin[j])), all_test_losses_lin[j], label='{}-lin'.format(widths[j]), linewidth=2.0)
    ax.axhline(mean_posterior_nll.item(), c='r', linestyle='dashed')
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.savefig('./canonical_gaussian/figures/sigma={},width={},testlosses_lin.png'.format(this_sigma, widths[0]))

    # Plot testing losses zoomed
    fig, ax = plt.subplots(figsize=(10,10))
    for j in range(len(all_losses)):
        if j == len(all_losses)-1:
            ax.plot(5000*np.arange(len(all_test_losses[j])), all_test_losses[j], label='{}'.format(widths[j]), linewidth=2.0)
        ax.plot(5000*np.arange(len(all_test_losses_lin[j])), all_test_losses_lin[j], label='{}-lin'.format(widths[j]), linewidth=2.0)
    ax.axhline(mean_posterior_nll.item(), c='r', linestyle='dashed')
    plt.legend(title='Width')
    plt.xlabel('Gradient Step')
    plt.ylabel('NLL')
    plt.ylim(0.705, 0.735)
    plt.savefig('./canonical_gaussian/figures/sigma={},width={},testlosses_lin_zoom.png'.format(this_sigma, widths[0]))

def matrix_norm_plot(cfg, p, this_sigma, widths, **kwargs):
    # Plot Frobeniuis Norm Jacobian
    to_plot = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, cfg.training.lr, w)
        jacs = np.load('{}/canonical_gaussian/{}/jacobians.npy'.format(cfg.log_dir, access_string))
        to_plot.append(jacs)

    # Plot losses
    fig, ax = plt.subplots(figsize=(30,10))
    
    for j in range(len(widths)):
        nrecorded = len(to_plot[j])
        ax.plot(250*torch.arange(len(to_plot[j]))[:nrecorded//2], to_plot[j][:nrecorded//2], label=widths[j], linewidth=4.0)
    leg = plt.legend(title='Width', loc='lower right', ncol=len(widths))
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(8.0)
    plt.xlabel('Gradient Step')
    plt.title('Frobenius Norm: $||Jf(x;\phi)-Jf(x;\phi_0)||_F$')
    plt.savefig('./canonical_gaussian/figures/sigma={},p={},jacobian.png'.format(this_sigma, p), dpi=600)

    # Plot Frobenius Norm NTK
    to_plot = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, cfg.training.lr, w)
        jacs = np.load('{}/canonical_gaussian/{}/ntks.npy'.format(cfg.log_dir, access_string))
        to_plot.append(jacs)

    # Plot losses
    fig, ax = plt.subplots(figsize=(30,10))
    
    for j in range(len(widths)):
        nrecorded = len(to_plot[j])
        ax.plot(250*torch.arange(len(to_plot[j]))[:nrecorded//2], to_plot[j][:nrecorded//2], label=widths[j], linewidth=4.0)
    leg = plt.legend(title='Width', loc='lower right', ncol=len(widths))
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(8.0)
    plt.xlabel('Gradient Step')
    plt.title('Frobenius Norm: $||K_\phi(x,x)-K_{\phi_0}(x,x)||_F$')
    plt.savefig('./canonical_gaussian/figures/sigma={},p={},ntk.png'.format(this_sigma, p), dpi=600)

    # Plot trajectory of eigenvalues
    to_plot = []
    for w in widths:
        access_string = 'loss={},sigma={},lr={},width={},nlayer=1'.format(cfg.training.loss, this_sigma, cfg.training.lr, w)
        evals = np.load('{}/canonical_gaussian/{}/evals.npy'.format(cfg.log_dir, access_string))
        to_plot.append(evals)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(30,10))
    
    for j in range(len(widths)):
        nrecorded = len(to_plot[j])
        ax[0].plot(250*torch.arange(len(to_plot[j]))[:nrecorded//2], to_plot[j][:nrecorded//2, 0], label=widths[j], linewidth=4.0)
        ax[1].plot(250*torch.arange(len(to_plot[j]))[:nrecorded//2], to_plot[j][:nrecorded//2, 1], label=widths[j], linewidth=4.0)
    
    ax[0].set_ylabel('$|\lambda_1|$')
    ax[1].set_ylabel('$|\lambda_2|$')
    # # change the line width for the legend
    # for line in leg.get_lines():
    #     line.set_linewidth(8.0)
    ax[0].set_xlabel('Gradient Step')
    ax[1].set_xlabel('Gradient Step')
    fig.suptitle('Eigenvalues of $K_\phi(x,x)-K_{\phi_0}(x,x)$')
    #fig.legend(title='Width', ncol=3)
    fig.legend(ax[0].get_legend_handles_labels()[0], ax[0].get_legend_handles_labels()[1], loc='lower right')

    plt.savefig('./canonical_gaussian/figures/sigma={},p={},evals.png'.format(this_sigma, p), dpi=600)


@hydra.main(version_base=None, config_path="../config", config_name="canonical")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config")
    # cfg = compose(config_name="canonical")
    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    # Access experimental settings to plot
    cfg.training.lr=1e-4
    sigmas = [5e-1]
    widths = [64,128,256,512,1024,2048,4096]
    log_dir = cfg.log_dir

    (true_theta, 
    true_x,
    true_noise,
    test_theta,
    test_x,
    test_noise, 
    logger_string,
    encoder,
    optimizer,
    scheduler,
    lin_encoder,
    lin_optimizer,
    lin_scheduler,
    kwargs) = setup(cfg)


    bin_width=1e-3
    theta_grid = torch.arange(1e-6, 2*math.pi, bin_width)

    for p in ['fro']:
        for sigma in sigmas:
            try:
                matrix_norm_plot(cfg, p, sigma, widths, **kwargs)
            except:
                print('Exception on {},{}'.format(sigma, p))
                continue

    for sigma in sigmas:
        for width in widths:
            try:
                fixed_sigma_width_plot_lin(cfg, sigma, [width], theta_grid, bin_width, test_theta, test_x, **kwargs)
            except:
                print('Exception on {}'.format(sigma))
                continue

if __name__ == "__main__":
   main()



