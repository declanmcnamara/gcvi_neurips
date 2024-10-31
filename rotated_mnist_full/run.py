import os
os.environ["OMP_NUM_THREADS"] = "4" 
os.environ["OPENBLAS_NUM_THREADS"] = "4" 
os.environ["MKL_NUM_THREADS"] = "4"  
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  
os.environ["NUMEXPR_NUM_THREADS"] = "4"
import sys
sys.path.append('../')
import torch
import numpy as np
import hydra
import sys
sys.path.append("../")
import numpy as np 
# -- plotting -- 
from os.path import exists
from hydra import compose, initialize
from omegaconf import DictConfig
import random
from losses import iwbo_loss_nonamortized, favi_loss
from setup import setup_nonamortized
from rotated_mnist_toy.setup import setup

def loss_choice(loss_name, z, true_x, **kwargs):
    if loss_name == 'favi':
        return favi_loss(true_x, **kwargs)
    elif loss_name == 'iwbo':
        return iwbo_loss_nonamortized(true_x, **kwargs)
    else:
        raise ValueError('Specify an appropriate loss name string.')


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
    log_dir = cfg.log_dir

    (
        z,
        xs,
        prior,
        steps,
        device,
        mb_size,
        encoder,
        angle_encoder,
        optimizer_encoder,
        optimizer_angle_encoder,
        scheduler_encoder,
        scheduler_angle_encoder,
        logger_string,
        gan,
        kwargs
    ) = setup_nonamortized(cfg) if cfg.training.loss == 'iwbo' else setup(cfg)

    if not exists('{}'.format(cfg.log_dir)):
        os.mkdir('{}'.format(cfg.log_dir))
    if not exists('{}/rotated_mnist_full'.format(cfg.log_dir)):
        os.mkdir('{}/rotated_mnist_full'.format(cfg.log_dir))
    if not exists('{}/rotated_mnist_full/{}'.format(cfg.log_dir, logger_string)):
        os.mkdir('{}/rotated_mnist_full/{}'.format(cfg.log_dir, logger_string))

    loss_name = kwargs['loss_name']
    kwargs.pop('loss_name')

    angle_estimates = []

    for j in range(steps):
        if optimizer_encoder is not None:
            optimizer_encoder.zero_grad()
        optimizer_angle_encoder.zero_grad()
        loss = loss_choice(loss_name, z, xs, **kwargs)
        if torch.isnan(loss).any():
            continue
        loss.backward()
        if scheduler_encoder is not None:
            scheduler_encoder.step_and_update_lr()
        scheduler_angle_encoder.step_and_update_lr()

        print('Loss iteration {} is {}'.format(j, loss.item()))
        del loss
        torch.cuda.empty_cache()

        # Angle logging
        if cfg.training.loss == 'favi':
            angle_estimate = angle_encoder.get_q(xs).loc.item()
        else:
            angle_estimate = gan.rotation.item() # update model parameter is nonamortized setting
        angle_estimates.append(angle_estimate)
            
        if (j+1) % 1000 == 0:
            torch.save(gan.state_dict(), '{}/rotated_mnist_full/{}/gan.pth'.format(log_dir, logger_string))
            if encoder:
                torch.save(encoder.state_dict(), '{}/rotated_mnist_full/{}/encoder.pth'.format(log_dir, logger_string))
            if angle_encoder:
                torch.save(angle_encoder.state_dict(), '{}/rotated_mnist_full/{}/angle_encoder.pth'.format(log_dir, logger_string))
            np.save('{}/rotated_mnist_full/{}/angle_estimates.npy'.format(log_dir, logger_string), np.array(angle_estimates))

if __name__ == "__main__":
    main()  
