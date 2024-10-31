import torch

def processor(outputs):
    '''Given outputs of set transformer, cast
    to parameters for encoder model.'''
    means = outputs[...,0].view(-1,1) 
    log_sds = outputs[...,1].view(-1,1).clamp(-10., 10.)
    return means, torch.exp(log_sds)

def processor_batch(outputs):
    means = outputs[..., 0].unsqueeze(-1)
    log_sds = outputs[..., 1].clamp(-10., 10.).unsqueeze(-1)
    return means, torch.exp(log_sds)