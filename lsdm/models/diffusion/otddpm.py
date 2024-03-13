import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from functools import partial

from lsdm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from lsdm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from lsdm.models.diffusion.ddpm import DDPM


class OTDDPM(DDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(conditioning_key=None, *args, **kwargs)
        self.clip_denoised = True

        betas = make_beta_schedule(self.beta_schedule, self.timesteps, linear_start=self.linear_start, linear_end=self.linear_end,
                                       cosine_s=self.cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('sqrt_alphas_cumprod_prev', to_torch(np.sqrt(alphas_cumprod_prev)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod_prev', to_torch(np.sqrt(1. - alphas_cumprod_prev)))
        self.register_buffer('coefficient_xt', to_torch(np.sqrt(1.-alphas_cumprod_prev)/np.sqrt(1.-alphas_cumprod)))
        self.register_buffer('coefficient_x0', to_torch(np.sqrt(alphas_cumprod)-np.sqrt(alphas_cumprod*(1.-alphas_cumprod_prev)/(1.-alphas_cumprod))))

    def normalization(self, x):
        x = (x - x.min()) / (x.max() - x.min())
        return x*2. - 1.

    def normalization_0_1(self, x):
        return (x - x.min()) / (x.max() - x.min())

    # training steps:
    def get_input(self, batch, k):
        x = super().get_input(batch, k).to(self.device)
        x_T = super().get_input(batch, "source").to(self.device)
        return x, x_T

    def q_project(self, x_0, x_T, t):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * x_T)

    def p_losses(self, x_0, x_T, t):
        x_intermediate = self.q_project(x_0, x_T, t)
        model_out = self.model(x_intermediate, t)

        loss_dict = {}
        if self.parameterization == "eps":
            # directly compared with the added noise:
            target = x_T
        elif self.parameterization == "x0":
            target = x_0
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x_0, x_T, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x_0.shape[0],), device=self.device).long()
        return self.p_losses(x_0, x_T, t, *args, **kwargs)

    def shared_step(self, batch):
        x_0, x_T = self.get_input(batch, self.first_stage_key)
        # just the forward step:
        loss, loss_dict = self(x_0, x_T)
        return loss, loss_dict

    # inference steps:
    def predict_start_from_noise(self, x_t, x_capital_t, t):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * x_capital_t
        )

    def q_posterior_from_x0(self, x_0, x_T, t):
        posterior_mean = (
                extract_into_tensor(self.sqrt_alphas_cumprod_prev, t, x_T.shape) * x_0 +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod_prev, t, x_T.shape) * x_T
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_T.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_T.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_from_xt(self, x_0, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.coefficient_x0, t, x_t.shape) * x_0 +
                extract_into_tensor(self.coefficient_xt, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            # x0:
            x_recon = self.predict_start_from_noise(x_t=x, x_capital_t=model_out, t=t)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise TypeError("no parameterization name called '{}'".format(self.parameterization))
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if self.parameterization == "eps":
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior_from_x0(x_0=x_recon, x_T=model_out, t=t)
        else:
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior_from_xt(x_0=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        # clip: False
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = self.normalization(noise_like(x.shape, device, repeat_noise))
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise * 1e-2

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = self.normalization(torch.randn(shape, device=device))
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = super().get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        # diffusion_row = list()
        # x_start = x[:n_row]

        # for t in range(self.num_timesteps):
        #     if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
        #         t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
        #         t = t.to(self.device).long()
        #         noise = torch.randn_like(x_start)
        #         x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        #         diffusion_row.append(x_noisy)

        # log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            # for i in range(samples.shape[0]):
            #     samples[i] = self.normalization_0_1(samples[i])
            #     for j in range(len(denoise_row)):
            #         denoise_row[j][i] = self.normalization_0_1(denoise_row[j][i])
            
            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log
