import torch
import torch.distributions as D
import torch.nn as nn
from einops import rearrange,reduce,repeat
import torch.nn.functional as F

def get_imp_weights_nonamortized(particles, pts, log=False, **kwargs):
    '''
    Given a set of particles, and data poins pts, uses encoder and model to construct
    importance weights.
    '''
    device = kwargs['device']
    encoder = kwargs['encoder']
    mb_size = kwargs['mb_size']
    prior = kwargs['prior']
    gan = kwargs['gan']
    K = kwargs['K']

    log_prior = reduce(gan.prior.log_prob(particles), 'K b d -> K', 'sum')#.sum(-1)
    log_q = reduce(encoder.get_q(pts).log_prob(particles), 'K b d -> K', 'sum')#.sum(-1)

    image_means = gan.decoder(particles)
    rot_means = gan.rotate(image_means, gan.rotation)
    gen_imgs = gan.denorm(rot_means.reshape((K, mb_size, 28,28)))
    final_means = gen_imgs.reshape(K, mb_size, -1)
    distr = D.Normal(final_means, gan.noise)
    log_p = reduce(distr.log_prob(pts), 'K b 784 -> K', 'sum')#.sum(-1)

    log_weights = log_prior + log_p - log_q
    weights = nn.Softmax(0)(log_weights)

    if log:
        return weights.to(device), log_weights.to(device)
    else:
        return weights.to(device)
    

def iwbo_loss_nonamortized(xs, **kwargs):
    # Choose data points
    K = kwargs['K']
    device = kwargs['device']
    mb_size = kwargs['mb_size']
    indices = torch.randint(low=0, high=len(xs), size=(mb_size,))
    pts = xs[indices].to(device)

    encoder = kwargs['encoder']

    # Get samples, weights
    particles = encoder.get_q(pts).rsample((K,)) #K x b x d
    weights, log_weights = get_imp_weights_nonamortized(particles, pts, log=True, **kwargs)
    weights = weights.detach()

    # Return loss
    return -1*torch.dot(weights, log_weights)

def favi_loss(xs, **kwargs):
    # Generate synethetic data batch
    mb_size = kwargs['favi_mb_size']
    angle_prior = kwargs['angle_prior']
    gan = kwargs['gan']
    device = kwargs['device']
    angle_encoder = kwargs['angle_encoder']

    sim_angle = angle_prior.sample().to(device)
    z = gan.prior.sample((mb_size,64)).to(device)
    x = gan.decoder(z)
    x = gan.rotate(x, sim_angle)
    gen_imgs = gan.denorm(x.reshape((-1, 28,28)).detach())
    distr = D.Normal(gen_imgs.reshape(mb_size,-1), gan.noise)
    sim_x = distr.sample()

    return -1*angle_encoder.get_log_prob(sim_angle, sim_x)
