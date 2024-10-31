import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import torch
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
import numpy as np 
# -- plotting -- 
import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
import hydra
from hydra import compose, initialize
from omegaconf import DictConfig
import random
import matplotlib as mpl
from setup import setup_nonamortized
import scienceplots
plt.style.use('science')
mpl.rc('lines', linewidth=4.0)
matplotlib.rcParams.update({'font.size': 45}) 

def each_loss(cfg, losses, inits, lr_favi, lr_iwbo, **kwargs):
    
    for loss in losses:
        all_angles = []
        this_lr = lr_favi if loss == 'favi' else lr_iwbo
        for init in inits:
            try:
                logger_string = '{},{},{},{},init={},true={}'.format(loss, this_lr, 64, 32, init, cfg.data.true_angle)
                angles = np.load('{}/rotated_mnist_full/{}/angle_estimates.npy'.format(cfg.log_dir, logger_string))
                if loss == 'favi':
                    # Explicitly correct for network learned offset
                    angles = ((angles*180/math.pi) - init) % 360 # ((angles*180/math.pi) % 360) - init
                elif loss == 'iwbo':
                    angles = ((angles*180/math.pi)) % 360 
                all_angles.append(angles)
            except Exception as e:
                print(e)
                pass
        
        # Full trajectories of each objective
        if loss == 'favi':
            fig, ax = plt.subplots(figsize=(60,30))
            for j in range(len(all_angles)):
                ax.plot(all_angles[j], linewidth=1.0)
            plt.xticks(fontsize=100)
            plt.yticks(fontsize=100)
            #plt.title('Angles Estimates - {}'.format(loss.upper()), fontsize=95)
            plt.ylim(240, 280)
            plt.xlabel('Gradient Step', fontsize=130)
            plt.ylabel('Angle (Degrees)', fontsize=130)
            plt.savefig('./rotated_mnist_full/figs/angle_estimates_{}.png'.format(loss), dpi=100)
        if loss == 'iwbo':
            fig, ax = plt.subplots(figsize=(60,30))
            for j in range(len(all_angles)):
                ax.plot(all_angles[j], linewidth=5.0)
            plt.xticks(fontsize=100)
            plt.yticks(fontsize=100)
            #plt.title('Angles Estimates - {}'.format(loss.upper()), fontsize=75)
            plt.xlabel('Gradient Step', fontsize=130)
            plt.ylabel('Angle (Degrees)', fontsize=130)
            plt.savefig('./rotated_mnist_full/figs/angle_estimates_{}.png'.format(loss), dpi=100)

        # Truncated trajectories, unique truncation for each loss.
        if loss == 'favi':
            fig, ax = plt.subplots(figsize=(60,30))
            for j in range(len(all_angles)):
                ax.plot(all_angles[j][:2000], linewidth=3.0)
            plt.xticks(fontsize=100)
            plt.yticks(fontsize=100)
            #plt.title('Angles Estimates'.format(loss.upper()), fontsize=95)
            plt.xlabel('Gradient Step', fontsize=130)
            plt.ylabel('Angle (Degrees)', fontsize=130)
            plt.savefig('./rotated_mnist_full/figs/angle_estimates_favi_trunc.png', dpi=100)
        if loss == 'iwbo':
            fig, ax = plt.subplots(figsize=(60,30))
            for j in range(len(all_angles)):
                ax.plot(all_angles[j][:1250], linewidth=10.0)
            plt.xticks(fontsize=100)
            plt.yticks(fontsize=100)
            #plt.title('Angles Estimates'.format(loss.upper()), fontsize=95)
            plt.xlabel('Gradient Step', fontsize=130)
            plt.ylabel('Angle (Degrees)', fontsize=130)
            plt.savefig('./rotated_mnist_full/figs/angle_estimates_iwbo_trunc.png', dpi=100)

    return

def example_grid(cfg, true_x, height=20, width=10, **kwargs):
    n_digit = height*width
    digits = true_x[:n_digit].reshape((n_digit, 28, 28)).cpu().numpy()
    fig, ax = plt.subplots(nrows=height, ncols=width, figsize=(50,50))
    for i in range(height):
        for j in range(width):
            this_digit = height*i+j
            ax[i,j].imshow(digits[this_digit])
            ax[i,j].set_xticklabels([])
            ax[i,j].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig('./rotated_mnist_full/figs/example_grid.png')

@hydra.main(version_base=None, config_path="../config", config_name="mnist_full")
def main(cfg : DictConfig) -> None:
    # initialize(config_path="../config", job_name="test_app")
    # cfg = compose(config_name="mnist_full")

    seed = cfg.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    dir = cfg.dir
    os.chdir(dir)

    (
        z,
        xs,
        prior,
        epochs,
        device,
        mb_size,
        encoder,
        angle_encoder,
        optimizer_encoder,
        optimizer_encoder2,
        scheduler_encoder,
        scheduler_encoder2,
        logger_string,
        gan,
        kwargs
    ) = setup_nonamortized(cfg)

    losses = ['favi', 'iwbo']
    lr_favi = 1e-4
    lr_iwbo = 1e-2
    inits = list(np.arange(0,360,30))

    each_loss(cfg, losses, inits, lr_favi, lr_iwbo, **kwargs)
    example_grid(cfg, xs, 10, 10, **kwargs)

if __name__ == "__main__":
    main()  