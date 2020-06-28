import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import numpy as np
import math

def loss_function(recon_x,x,mu,logstd,rec_log_std=0,sum_samplewise=True):
    """
    recon_x : reconstructed sample  
    x : sample
    mu: mean of sample
    logstd: log of std deviation
    rec_log_std: reconstruction log standard deviation
    Gives the loss function for CeVAE network"""

    rec_std = math.exp(rec_log_std)
    rec_var = rec_std**2

    x_dist = dist.Normal(recon_x,rec_std)
    log_p_x_z = x_dist.log_prob(x)
    if sum_samplewise:
        log_p_x_z = torch.sum(log_p_x_z, dim=(1, 2, 3))

    z_prior = dist.Normal(0, 1.)
    z_post = dist.Normal(mu, torch.exp(logstd))

    kl_div = dist.kl_divergence(z_post, z_prior)
    if sum_samplewise:
        kl_div = torch.sum(kl_div, dim=(1, 2, 3))

    if sum_samplewise:
        loss = torch.mean(kl_div - log_p_x_z)
    else:
        loss = torch.mean(torch.sum(kl_div, dim=(1, 2, 3)) - torch.sum(log_p_x_z, dim=(1, 2, 3)))

    return loss, kl_div, -log_p_x_z


def kl_loss_fn(recon_x,x,mu,logstd,rec_log_std=0,sum_samplewise=True):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
    # loss = nn.KLDivLoss()
    # losses = loss(mu,logstd**2)
    loss = BCE*KLD
    return loss, BCE, KLD

def kl_loss_fn_train(recon_x,x,mu,logstd,rec_log_std=0,sum_samplewise=True):
    BCE = F.mse_loss(recon_x, x)
    KLD = -0.5 * torch.sum(1 + logstd - mu.pow(2) - logstd.exp())
    # loss = nn.KLDivLoss()
    # losses = loss(mu,logstd**2)
    loss = BCE*KLD
    return loss


def rec_loss_fn (recon_x,x):

    """
    The function checks the reconstruction loss of image in VAE
    """

    loss_fn = nn.MSELoss()
    loss = loss_fn(x,recon_x)

    return loss