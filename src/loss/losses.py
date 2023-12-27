import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd

from src.helpers.utils import get_scheduled_params

def weighted_rate_loss(config, total_nbpp, total_qbpp, step_counter, ignore_schedule=False):
    """
    Heavily penalize the rate with weight lambda_A >> lambda_B if it exceeds 
    some target r_t, otherwise penalize with lambda_B
    """
    lambda_A = get_scheduled_params(config.lambda_A, config.lambda_schedule, step_counter, ignore_schedule)
    lambda_B = get_scheduled_params(config.lambda_B, config.lambda_schedule, step_counter, ignore_schedule)

    assert lambda_A > lambda_B, "Expected lambda_A > lambda_B, got (A) {} <= (B) {}".format(
        lambda_A, lambda_B)

    target_bpp = get_scheduled_params(config.target_rate, config.target_schedule, step_counter, ignore_schedule)

    total_qbpp = total_qbpp.item()
    if total_qbpp > target_bpp:
        rate_penalty = lambda_A
    else:
        rate_penalty = lambda_B
    weighted_rate = rate_penalty * total_nbpp

    return weighted_rate, float(rate_penalty)

def _non_saturating_loss(D_real_logits, D_gen_logits, D_real=None, D_gen=None):

    D_loss_real = F.binary_cross_entropy_with_logits(input=D_real_logits,
        target=torch.ones_like(D_real_logits))
    D_loss_gen = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.zeros_like(D_gen_logits))
    D_loss = D_loss_real + D_loss_gen

    G_loss = F.binary_cross_entropy_with_logits(input=D_gen_logits,
        target=torch.ones_like(D_gen_logits))

    return D_loss, G_loss

def _least_squares_loss(D_real, D_gen, D_real_logits=None, D_gen_logits=None):
    D_loss_real = torch.mean(torch.square(D_real - 1.0))
    D_loss_gen = torch.mean(torch.square(D_gen))
    D_loss = 0.5 * (D_loss_real + D_loss_gen)

    G_loss = 0.5 * torch.mean(torch.square(D_gen - 1.0))
    
    return D_loss, G_loss

def wgan_gp_loss(disc_out,intermediates, D, l_gp=10):
    def compute_gradient_penalty(D, real_samples, gen_samples, latents):
        alpha = torch.rand((real_samples.size(0), 1, 1, 1),device=real_samples.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * gen_samples)).requires_grad_(True)
        d_interpolates = D(interpolates, latents)[0]
        gen = torch.ones(d_interpolates.shape[0], 1, requires_grad=False, device=real_samples.device)

        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=gen,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    D_real = disc_out.D_real
    D_gen = disc_out.D_gen
    real_imgs = intermediates.input_image.detach()
    gen_imgs = intermediates.reconstruction.detach()
    latents = intermediates.latents_quantized.detach()
    
    with torch.enable_grad():
        gradient_penalty = compute_gradient_penalty(D, real_imgs, gen_imgs, latents)
    D_loss = -torch.mean(D_real) + torch.mean(D_gen) + l_gp * gradient_penalty
    G_loss = -torch.mean(D_gen)

    return D_loss, G_loss

def wgan_div_loss(disc_out,intermediates, D, p=6,k=2):
    def compute_gradient_penalty(D,real_imgs,fake_imgs,latents):
        real_imgs = real_imgs.clone().detach().requires_grad_(True)
        fake_imgs = fake_imgs.clone().detach().requires_grad_(True)
        D_real = D(real_imgs,latents)[0]
        D_gen = D(fake_imgs,latents)[0]

        real_grad_out = torch.ones(D_real.size(0), 1, requires_grad=False, device=D_real.device)
        fake_grad_out = torch.ones(D_gen.size(0), 1, requires_grad=False, device=D_gen.device)
        with torch.enable_grad():
            real_grad = autograd.grad(
                D_real, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            fake_grad = autograd.grad(
                D_gen, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)
        div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
        return div_gp

    D_real = disc_out.D_real
    D_gen = disc_out.D_gen
    real_imgs = intermediates.input_image
    fake_imgs = intermediates.reconstruction.detach()
    latents = intermediates.latents_quantized.detach()

    with torch.enable_grad():
        div_gp = compute_gradient_penalty(D,real_imgs,fake_imgs,latents)
    D_loss = -torch.mean(D_real) + torch.mean(D_gen) + div_gp
    G_loss = -torch.mean(D_gen)

    return D_loss, G_loss
