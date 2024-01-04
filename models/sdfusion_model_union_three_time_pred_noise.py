# Reference: diffusion is borrowed from the LDM repo: https://github.com/CompVis/latent-diffusion
# Specifically, functions from: https://github.com/CompVis/latent-diffusion/blob/main/ldm/models/diffusion/ddpm.py

import os
import sys
from collections import OrderedDict
from functools import partial
import copy

import numpy as np
from omegaconf import OmegaConf
from termcolor import colored, cprint
from einops import rearrange, repeat
from tqdm import tqdm
from random import random
import ocnn
from ocnn.nn import octree2voxel, octree_pad
from ocnn.octree import Octree, Points
from models.networks.dualoctree_networks import dual_octree
from models.networks.diffusion_networks.modules import octree_align

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.special import expm1

import torchvision.utils as vutils
import torchvision.transforms as transforms

from models.base_model import BaseModel
from models.networks.diffusion_networks.network_three_times_8_channels import DiffusionUNet
from models.model_utils import load_dualoctree
from models.networks.diffusion_networks.ldm_diffusion_util import *

from models.networks.diffusion_networks.samplers.ddim_new import DDIMSampler

# distributed
from utils.distributed import reduce_loss_dict

# rendering
from utils.util_3d import init_mesh_renderer, render_sdf, render_sdf_dualoctree
from utils.util_dualoctree import calc_sdf

TRUNCATED_TIME = 0.7

class SDFusionModel(BaseModel):
    def name(self):
        return 'SDFusion-Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.model_name = self.name()
        self.device = opt.device
        self.gradient_clip_val = 1.
        self.start_iter = opt.start_iter


        ######## START: Define Networks ########
        assert opt.df_cfg is not None
        assert opt.vq_cfg is not None

        # init df
        df_conf = OmegaConf.load(opt.df_cfg)
        vq_conf = OmegaConf.load(opt.vq_cfg)

        self.vq_conf = vq_conf
        self.solver = self.vq_conf.solver

        self.input_depth = self.vq_conf.model.depth
        self.large_depth = self.vq_conf.model.depth_stop
        self.small_depth = self.vq_conf.model.small_depth
        self.full_depth = self.vq_conf.model.full_depth

        # init diffusion networks
        df_model_params = df_conf.model.params
        unet_params = df_conf.unet.params
        self.conditioning_key = df_model_params.conditioning_key
        self.thres = 0.5
        if self.conditioning_key == 'adm':
            self.num_classes = unet_params.num_classes
        elif self.conditioning_key == 'None':
            self.num_classes = 1
        self.df = DiffusionUNet(unet_params, conditioning_key=self.conditioning_key)
        self.df.to(self.device)

        # record z_shape
        self.split_channel = 8
        self.code_channel = self.vq_conf.model.embed_dim
        z_sp_dim = 2 ** self.full_depth
        self.z_shape = (self.split_channel, z_sp_dim, z_sp_dim, z_sp_dim)

        self.ema_df = copy.deepcopy(self.df)
        self.ema_df.to(self.device)
        if opt.isTrain:
            self.ema_rate = opt.ema_rate
            self.ema_updater = EMA(self.ema_rate)
            self.reset_parameters()
            set_requires_grad(self.ema_df, False)

        self.init_diffusion_params(scale=1, opt=opt)

        # init vqvae

        self.autoencoder = load_dualoctree(conf = vq_conf, ckpt = opt.vq_ckpt, opt = opt)

        ######## END: Define Networks ########

        if self.isTrain:

            # initialize optimizers
            self.optimizer = optim.AdamW([p for p in self.df.parameters() if p.requires_grad == True], lr=opt.lr)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1000, 0.9)

            self.optimizers = [self.optimizer]
            self.schedulers = [self.scheduler]

            self.print_networks(verbose=False)

        if opt.ckpt is not None:
            self.load_ckpt(opt.ckpt, load_opt=self.isTrain)
            if self.isTrain:
                self.optimizers = [self.optimizer]
            # self.schedulers = [self.scheduler]


        # setup renderer
        if 'snet' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif 'pix3d' in opt.dataset_mode:
            dist, elev, azim = 1.7, 20, 20
        elif opt.dataset_mode == 'buildingnet':
            dist, elev, azim = 1.0, 20, 20

        self.renderer = init_mesh_renderer(image_size=256, dist=dist, elev=elev, azim=azim, device=self.device)

        # for distributed training
        if self.opt.distributed:
            self.make_distributed(opt)
            self.df_module = self.df.module
            self.autoencoder_module = self.autoencoder.module

        else:
            self.df_module = self.df
            self.autoencoder_module = self.autoencoder

        self.ddim_steps = 200
        if self.opt.debug == "1":
            # NOTE: for debugging purpose
            self.ddim_steps = 7
        cprint(f'[*] setting ddim_steps={self.ddim_steps}', 'blue')

    def reset_parameters(self):
        self.ema_df.load_state_dict(self.df.state_dict())

    def make_distributed(self, opt):
        self.df = nn.parallel.DistributedDataParallel(
            self.df,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters = True,
        )
        if opt.sync_bn:
            self.autoencoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.autoencoder)
        self.autoencoder = nn.parallel.DistributedDataParallel(
            self.autoencoder,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            broadcast_buffers=False,
            find_unused_parameters = False,
        )

    ############################ START: init diffusion params ############################
    def init_diffusion_params(self, scale=3., opt=None):

        df_conf = OmegaConf.load(opt.df_cfg)

        # ref: ddpm.py, line 44 in __init__()
        self.parameterization = "eps"
        self.learn_logvar = False

        self.v_posterior = 0.
        self.original_elbo_weight = 0.
        self.l_simple_weight = 1.
        # ref: ddpm.py, register_schedule
        self.register_schedule()
        logvar_init = 0.
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,)).to(self.device)
        self.scale = scale # default for uncond

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                        linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.betas = to_torch(betas).to(self.device)
        self.alphas_cumprod = to_torch(alphas_cumprod).to(self.device)
        self.alphas_cumprod_prev = to_torch(alphas_cumprod_prev).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod)).to(self.device)
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod)).to(self.device)
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod)).to(self.device)
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod)).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod - 1)).to(self.device)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_torch(posterior_variance).to(self.device)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_torch(np.log(np.maximum(posterior_variance, 1e-20))).to(self.device)
        self.posterior_mean_coef1 = to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(self.device)
        self.posterior_mean_coef2 = to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)).to(self.device)

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas).to(self.device) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")

        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
    ############################ END: init diffusion params ############################


    def batch_to_cuda(self, batch):
        def points2octree(points):
            octree = ocnn.octree.Octree(depth = self.input_depth, full_depth = self.full_depth)
            octree.build_octree(points)
            return octree

        points = [pts.cuda(non_blocking=True) for pts in batch['points']]
        octrees = [points2octree(pts) for pts in points]
        octree = ocnn.octree.merge_octrees(octrees)
        octree.construct_all_neigh()
        batch['octree_in'] = octree

        batch['split_small'] = self.octree2split_small(batch['octree_in'])
        batch['split_large'] = self.octree2split_large(batch['octree_in'])

    def set_input(self, input=None):
        self.batch_to_cuda(input)
        self.split_small = input['split_small']
        self.split_large = input['split_large']
        self.octree_in = input['octree_in']
        self.batch_size = self.octree_in.batch_size
        self.label = input['label']

    def switch_train(self):
        self.df.train()

    def switch_eval(self):
        self.df.eval()

    # check: ddpm.py, line 871 forward
    # check: p_losses
    # check: q_sample, apply_mode

    def forward(self):

        self.df.train()

        c = None

        with torch.no_grad():
            self.input_feature, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)

        batch_size = self.batch_size

        self.stage_flag = ''

        random_flag = random()

        stage1 = 1/3
        stage2 = 1/3
        stage3 = 1 - stage1 - stage2

        noised_split_small = None
        noised_split_large = None
        noised_input_feature = None

        if random_flag < stage1:

            self.stage_flag = 'small'

            times1 = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            times2 = torch.zeros((batch_size,), device = self.device).long() + self.num_timesteps
            times3 = torch.zeros((batch_size,), device = self.device).long() + self.num_timesteps

            alpha = self.sqrt_alphas_cumprod[times1]
            sigma = self.sqrt_one_minus_alphas_cumprod[times1]

            alpha = right_pad_dims_to(self.split_small, alpha)
            sigma = right_pad_dims_to(self.split_small, sigma)

            noise_small = torch.randn_like(self.split_small, device = self.device)
            noised_split_small = alpha * self.split_small + sigma * noise_small

        elif random_flag < stage1 + stage2:

            self.stage_flag = 'large'

            times1 = torch.zeros((batch_size,), device = self.device).long()
            times2 = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()
            times3 = torch.zeros((batch_size,), device = self.device).long() + self.num_timesteps

            alpha = self.sqrt_alphas_cumprod[times2]
            sigma = self.sqrt_one_minus_alphas_cumprod[times2]

            leaf_num = self.doctree_in.lnum[self.full_depth:self.small_depth].sum()
            minus = torch.zeros((leaf_num, self.split_channel), device = self.device) - 1
            self.split_large = torch.cat([minus, self.split_large], dim = 0)

            # print(self.split_large)

            noised_split_large = self.split_large.clone()

            batch_id = self.doctree_in.batch_id(depth = self.small_depth)
            noise_large = torch.randn_like(self.split_large)

            for i in range(batch_size):
                noised_split_large[batch_id == i] *= alpha[i]
                noise_i = noise_large[batch_id == i]
                sigma_i = sigma[i] * noise_i
                noised_split_large[batch_id == i] += sigma_i
        
        else:

            self.stage_flag = 'feature'

            times1 = torch.zeros((batch_size,), device = self.device).long()
            times2 = torch.zeros((batch_size,), device = self.device).long()
            times3 = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device).long()

            alpha = self.sqrt_alphas_cumprod[times3]
            sigma = self.sqrt_one_minus_alphas_cumprod[times3]

            noised_input_feature = self.input_feature.clone()

            batch_id = self.doctree_in.batch_id(depth = self.large_depth)
            noise_feature = torch.randn_like(self.input_feature)

            for i in range(batch_size):
                noised_input_feature[batch_id == i] *= alpha[i]
                noise_i = noise_feature[batch_id == i]
                sigma_i = sigma[i] * noise_i
                noised_input_feature[batch_id == i] += sigma_i

        output = self.df(x_small = noised_split_small, x_large = noised_split_large, x_feature = noised_input_feature, doctree_in = self.doctree_in, t1 = times1, t2 = times2, t3 = times3)
        # print(output.shape)

        if self.stage_flag == 'small':
            self.loss = F.mse_loss(output, noise_small)

        elif self.stage_flag == 'large':
            self.loss = F.mse_loss(output, noise_large)

        elif self.stage_flag == 'feature':
            self.loss = F.mse_loss(output, noise_feature)

    def get_sampling_timesteps(self, batch, device, steps):
        times = torch.linspace(1., 0., steps + 1, device=device)
        times = repeat(times, 't -> b t', b=batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        times = times.unbind(dim=-1)
        return times


    @torch.no_grad()
    def uncond_withdata_small(self, data, split_path, category = 'airplane', ema = True, ddim_steps = 200, ddim_eta = 0., save_index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        if data != None:
            self.set_input(data)

            octree_small = self.split2octree_small(self.split_small)

        elif split_path != None:
            split_small = torch.load(split_path)
            split_small = split_small.to(self.device)
            octree_small = self.split2octree_small(split_small)

        self.export_octree(octree_small, depth = self.small_depth, save_dir = f'{category}_lr', index = save_index)

        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        batch_size = doctree_small.batch_size

        doctree_num = doctree_small.total_num

        noised_split_large = torch.randn((doctree_num, self.split_channel), device = self.device)

        ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=ddim_steps,
                                                  num_ddpm_timesteps=self.num_timesteps,verbose=False)
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=False)

        ddim_sqrt_one_minus_alphas =np.sqrt(1. - ddim_alphas)
        timesteps = ddim_timesteps
        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        leaf_num = doctree_small.lnum[self.full_depth:self.small_depth].sum()

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            time1 = torch.zeros(batch_size, device = self.device)
            time2 = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            if ema:
                e_t = self.ema_df(x_small = None, x_large = noised_split_large, doctree_in = doctree_small, t1 = time1, t2 = time2)
            else:
                e_t = self.df(x_small = None, x_large = noised_split_large, doctree_in = doctree_small, t1 = time1, t2 = time2)

            # print(e_t.max(), e_t.min())
            a_t = ddim_alphas[index]
            a_prev = torch.tensor(ddim_alphas_prev[index])
            sigma_t = ddim_sigmas[index]
            sqrt_one_minus_at = ddim_sqrt_one_minus_alphas[index]

            pred_x0 = (noised_split_large - sqrt_one_minus_at * e_t) / a_t.sqrt()

            if step < 300:
                pred_x0.sign_()

            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * torch.randn(noised_split_large.shape, device = self.device)
            noised_split_large = a_prev.sqrt() * pred_x0 + dir_xt + noise

        print(noised_split_large)
        noised_split_large = noised_split_large[leaf_num:]
        print(noised_split_large.max(), noised_split_large.min())

        octree_large = self.split2octree_large(octree_small, noised_split_large)

        self.export_octree(octree_large, depth = self.input_depth, save_dir = f'{category}_hr', index = save_index)

        return

        doctree_large = dual_octree.DualOctree(octree_large)
        doctree_large.post_processing_for_docnn()

        doctree_num_large = doctree_large.total_num

        noised_feature = torch.randn((doctree_num_large, self.code_channel), device = self.device)

        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        feature_iter = tqdm(time_pairs, desc='latent feature sampling loop time step')
        feature_start = None

        for time3, time_next3 in feature_iter:

            log_snr = self.log_snr(time3)
            log_snr_next = self.log_snr(time_next3)

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            time1 = torch.zeros(batch_size, device = self.device)
            time2 = torch.zeros(batch_size, device = self.device)
            noise_cond1 = self.log_snr(time1)
            noise_cond2 = self.log_snr(time2)
            noise_cond3 = self.log_snr(time3)

            if ema:
                feature_start  = self.ema_df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = doctree_large, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            else:
                feature_start  = self.df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = doctree_large, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noised_feature * (1 - c) / alpha + c * feature_start)
            variance = (sigma_next ** 2) * c
            noised_feature = mean + torch.sqrt(variance) * torch.randn_like(noised_feature)

        samples = noised_feature
        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        self.output = self.autoencoder_module.decode_code(samples, doctree_large)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(f'{category}_mesh', index)


    @torch.no_grad()
    def uncond_withdata_large(self, data, steps = 200, category = 'airplane', ema = True, index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        self.set_input(data)
        self.export_octree(self.octree_in, depth = self.large_depth, save_dir = f'{category}_hr', index = index)
        self.input_data, self.doctree_in = self.autoencoder_module.extract_code(self.octree_in)
        batch_size = self.doctree_in.batch_size

        noised_feature = torch.randn_like(self.input_data)

        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        feature_iter = tqdm(time_pairs, desc='latent feature sampling loop time step')
        feature_start = None

        for time3, time_next3 in feature_iter:

            log_snr = self.log_snr(time3)
            log_snr_next = self.log_snr(time_next3)

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            time1 = torch.zeros(batch_size, device = self.device)
            time2 = torch.zeros(batch_size, device = self.device)
            noise_cond1 = self.log_snr(time1)
            noise_cond2 = self.log_snr(time2)
            noise_cond3 = self.log_snr(time3)

            if ema:
                feature_start  = self.ema_df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = self.doctree_in, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            else:
                feature_start  = self.df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = self.doctree_in, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noised_feature * (1 - c) / alpha + c * feature_start)
            variance = (sigma_next ** 2) * c
            noised_feature = mean + torch.sqrt(variance) * torch.randn_like(noised_feature)

        samples = noised_feature
        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        self.output = self.autoencoder_module.decode_code(samples, self.doctree_in)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(f'{category}_mesh', index)


    @torch.no_grad()
    def uncond(self, batch_size=16, steps=200, category = 'airplane', ema = True, truncated_index: float = 0.0, index = 0):

        if ema:
            self.ema_df.eval()
        else:
            self.df.eval()

        shape = (batch_size, *self.z_shape)

        small_time_pairs = self.get_sampling_timesteps(
            batch_size, device=self.device, steps=steps)

        noised_split_small = torch.randn(shape, device = self.device)

        x_start_small = None

        small_iter = tqdm(small_time_pairs, desc='small sampling loop time step')

        for time1, time_next1 in small_iter:

            log_snr = self.log_snr(time1)
            log_snr_next = self.log_snr(time_next1)
            log_snr, log_snr_next = map(
                partial(right_pad_dims_to, noised_split_small), (log_snr, log_snr_next))

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            noise_cond1 = self.log_snr(time1)
            time2 = torch.ones(batch_size, device = self.device)
            time3 = torch.ones(batch_size, device = self.device)
            noise_cond2 = self.log_snr(time2)
            noise_cond3 = self.log_snr(time3)

            if ema:
                x_start_small = self.ema_df(x_small = noised_split_small, x_large = None, x_feature = None, doctree_in = None, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            else:
                x_start_small = self.df(x_small = noised_split_small, x_large = None, x_feature = None, doctree_in = None, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            if time1[0] < TRUNCATED_TIME:
                x_start_small.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split_small * (1 - c) / alpha + c * x_start_small)
            variance = (sigma_next ** 2) * c
            noise = torch.where(
                rearrange(time_next1 > truncated_index, 'b -> b 1 1 1 1'),
                torch.randn_like(noised_split_small),
                torch.zeros_like(noised_split_small)
            )
            noised_split_small = mean + torch.sqrt(variance) * noise

        print(noised_split_small.max(), noised_split_small.min())
        # torch.save(noised_split_small, f'noised_split_small_{index}.pth')

        octree_small = self.split2octree_small(noised_split_small)

        self.export_octree(octree_small, self.small_depth, save_dir = f'{category}_lr', index = index)

        # return

        doctree_small = dual_octree.DualOctree(octree_small)
        doctree_small.post_processing_for_docnn()

        doctree_num_small = doctree_small.total_num
        noised_split_large = torch.randn((doctree_num_small, self.split_channel), device = self.device)

        print(noised_split_large.shape)

        large_time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        large_iter = tqdm(large_time_pairs, desc='large sampling loop time step')
        x_start_large = None

        leaf_num = doctree_small.lnum[self.full_depth:self.small_depth].sum()

        for time2, time_next2 in large_iter:

            log_snr = self.log_snr(time2)
            log_snr_next = self.log_snr(time_next2)

            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            time1 = torch.zeros(batch_size, device = self.device)
            noise_cond1 = self.log_snr(time1)
            noise_cond2 = self.log_snr(time2)

            time3 = torch.ones(batch_size, device = self.device)
            noise_cond3 = self.log_snr(time3)

            # if ema:
            #     pred_noise  = self.ema_df(x_small = None, x_large = noised_split_large, x_feature = None, doctree_in = doctree_small, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            # else:
            #     pred_noise  = self.df(x_small = None, x_large = noised_split_large, x_feature = None, doctree_in = doctree_small, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            # x_start_large = (noised_split_large - sigma * pred_noise) / alpha.clamp(min=1e-8)

            # if time2[0] < TRUNCATED_TIME:
            #     x_start_large.sign_()

            # noised_split_large = x_start_large * alpha_next + pred_noise * sigma_next


            if ema:
                x_start_large  = self.ema_df(x_small = None, x_large = noised_split_large, x_feature = None, doctree_in = doctree_small, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            else:
                x_start_large  = self.df(x_small = None, x_large = noised_split_large, x_feature = None, doctree_in = doctree_small, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            if time2[0] < TRUNCATED_TIME:
                x_start_large.sign_()

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (noised_split_large * (1 - c) / alpha + c * x_start_large)
            variance = (sigma_next ** 2) * c
            noised_split_large = mean + torch.sqrt(variance) * torch.randn_like(noised_split_large)

        noised_split_large = noised_split_large[leaf_num:]
        print(noised_split_large.max(), noised_split_large.min())

        octree_large = self.split2octree_large(octree_small, noised_split_large)

        self.export_octree(octree_large, depth = self.large_depth, save_dir = f'{category}_hr', index = index)

        # return

        doctree_large = dual_octree.DualOctree(octree_large)
        doctree_large.post_processing_for_docnn()

        doctree_num_large = doctree_large.total_num

        noised_feature = torch.randn((doctree_num_large, self.code_channel), device = self.device)

        time_pairs = self.get_sampling_timesteps(
            1, device=self.device, steps=steps)

        feature_iter = tqdm(time_pairs, desc='latent feature sampling loop time step')
        feature_start = None

        for time3, time_next3 in feature_iter:

            log_snr = self.log_snr(time3)
            log_snr_next = self.log_snr(time_next3)

            alpha, _ = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            time1 = torch.zeros(batch_size, device = self.device)
            time2 = torch.zeros(batch_size, device = self.device)
            noise_cond1 = self.log_snr(time1)
            noise_cond2 = self.log_snr(time2)
            noise_cond3 = self.log_snr(time3)

            if ema:
                feature_start  = self.ema_df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = doctree_large, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)
            else:
                feature_start  = self.df(x_small = None, x_large = None, x_feature = noised_feature, doctree_in = doctree_large, t1 = noise_cond1, t2 = noise_cond2, t3 = noise_cond3)

            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (noised_feature * (1 - c) / alpha + c * feature_start)
            variance = (sigma_next ** 2) * c
            noised_feature = mean + torch.sqrt(variance) * torch.randn_like(noised_feature)

        samples = noised_feature
        print(samples.max())
        print(samples.min())
        print(samples.mean())
        print(samples.std())

        self.output = self.autoencoder_module.decode_code(samples, doctree_large)
        self.get_sdfs(self.output['neural_mpu'], batch_size, bbox = None)
        self.export_mesh(f'{category}_mesh', index)


    def logits2voxel(self, logits, octree):

        logit_full = logits[self.full_depth]
        logit_full_p1 = logits[self.full_depth + 1]
        logit_full_p1 = logit_full_p1.reshape(-1, 8)
        total_logits = torch.zeros((len(logit_full), 8), device = self.device) - 1
        total_logits[logit_full > 0] = logit_full_p1
        x_start = octree2voxel(total_logits, octree = octree, depth = self.full_depth)
        x_start = x_start.permute(0,4,1,2,3).contiguous()

        return x_start

    def octree2split_small(self, octree):

        child_full_p1 = octree.children[self.full_depth + 1]
        split_full_p1 = (child_full_p1 >= 0)
        split_full_p1 = split_full_p1.reshape(-1, 8)
        split_full = octree_pad(data = split_full_p1, octree = octree, depth = self.full_depth)
        split_full = octree2voxel(data=split_full, octree=octree, depth = self.full_depth)
        split_full = split_full.permute(0,4,1,2,3).contiguous()

        split_full = split_full.float()
        split_full = 2 * split_full - 1  # scale to [-1, 1]

        return split_full

    def octree2split_large(self, octree):

        child_small_p1 = octree.children[self.small_depth + 1]
        split_small_p1 = (child_small_p1 >= 0)
        split_small_p1 = split_small_p1.reshape(-1, 8)
        split_small = octree_pad(data = split_small_p1, octree = octree, depth = self.small_depth)

        split_small = split_small.float()
        split_small = 2 * split_small - 1    # scale to [-1, 1]

        return split_small

    def split2octree_small(self, split):

        discrete_split = copy.deepcopy(split)
        discrete_split[discrete_split > 0] = 1
        discrete_split[discrete_split < 0] = 0

        batch_size = discrete_split.shape[0]
        octree_out = create_full_octree(depth = self.input_depth, full_depth = self.full_depth, batch_size = batch_size, device = self.device)
        split_sum = torch.sum(discrete_split, dim = 1)
        nempty_mask_voxel = (split_sum > 0)
        x, y, z, b = octree_out.xyzb(self.full_depth)
        nempty_mask = nempty_mask_voxel[b,x,y,z]
        label = nempty_mask.long()
        octree_out.octree_split(label, self.full_depth)
        octree_out.octree_grow(self.full_depth + 1)
        octree_out.depth += 1

        x, y, z, b = octree_out.xyzb(depth = self.full_depth, nempty = True)
        nempty_mask_p1 = discrete_split[b,:,x,y,z]
        nempty_mask_p1 = nempty_mask_p1.reshape(-1)
        label_p1 = nempty_mask_p1.long()

        # for i in range(len(label_p1)):
        #     if random() < 0.01:
        #         if label_p1[i] == 0:
        #             label_p1[i] = 1
        #         elif label_p1[i] == 1:
        #             label_p1[i] = 0

        octree_out.octree_split(label_p1, self.full_depth + 1)
        octree_out.octree_grow(self.full_depth + 2)
        octree_out.depth += 1

        return octree_out

    def split2octree_large(self, octree, split):

        discrete_split = copy.deepcopy(split)
        discrete_split[discrete_split > 0] = 1
        discrete_split[discrete_split < 0] = 0

        octree_out = copy.deepcopy(octree)
        split_sum = torch.sum(discrete_split, dim = 1)
        nempty_mask_small = (split_sum > 0)
        label = nempty_mask_small.long()
        octree_out.octree_split(label, depth = self.small_depth)
        octree_out.octree_grow(self.small_depth + 1)
        octree_out.depth += 1

        nempty_mask_small_p1 = discrete_split[split_sum > 0]
        nempty_mask_small_p1 = nempty_mask_small_p1.reshape(-1)
        label_p1 = nempty_mask_small_p1.long()
        octree_out.octree_split(label_p1, depth = self.small_depth + 1)
        octree_out.octree_grow(self.small_depth + 2)
        octree_out.depth += 1

        return octree_out

    def export_octree(self, octree, depth, save_dir = None, index = 0):

        if not os.path.exists(save_dir): os.makedirs(save_dir)

        batch_id = octree.batch_id(depth = depth, nempty = False)
        data = torch.ones((len(batch_id), 1), device = self.device)
        data = octree2voxel(data = data, octree = octree, depth = depth, nempty = False)
        data = data.permute(0,4,1,2,3).contiguous()

        batch_size = octree.batch_size

        for i in tqdm(range(batch_size)):
            voxel = data[i].squeeze().cpu().numpy()
            mesh = voxel2mesh(voxel)
            if batch_size == 1:
                mesh.export(os.path.join(save_dir, f'{index}.obj'))
            else:
                mesh.export(os.path.join(save_dir, f'{i}.obj'))


    def get_sdfs(self, neural_mpu, batch_size, bbox):
        # bbox used for marching cubes
        if bbox is not None:
            self.bbmin, self.bbmax = bbox[:3], bbox[3:]
        else:
            sdf_scale = self.solver.sdf_scale
            self.bbmin, self.bbmax = -sdf_scale, sdf_scale    # sdf_scale = 0.9

        self.sdfs = calc_sdf(neural_mpu, batch_size, size = self.solver.resolution, bbmin = self.bbmin, bbmax = self.bbmax)

    def export_mesh(self, save_dir, index, level = 0):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        ngen = self.sdfs.shape[0]
        size = self.solver.resolution
        bbmin = self.bbmin
        bbmax = self.bbmax
        mesh_scale=self.vq_conf.data.test.point_scale
        for i in range(ngen):
            filename = os.path.join(save_dir, f'{i}.obj')
            if ngen == 1:
                filename = os.path.join(save_dir, f'{index}.obj')
            sdf_value = self.sdfs[i].cpu().numpy()
            vtx, faces = np.zeros((0, 3)), np.zeros((0, 3))
            try:
                vtx, faces, _, _ = skimage.measure.marching_cubes(sdf_value, level)
            except:
                pass
            if vtx.size == 0 or faces.size == 0:
                print('Warning from marching cubes: Empty mesh!')
                return
            vtx = vtx * ((bbmax - bbmin) / size) + bbmin   # [0,sz]->[bbmin,bbmax]  把vertex放缩到[bbmin, bbmax]之间
            vtx = vtx * mesh_scale
            mesh = trimesh.Trimesh(vtx, faces)  # 利用Trimesh创建mesh并存储为obj文件。
            mesh.export(filename)

    @torch.no_grad()
    def eval_metrics(self, dataloader, thres=0.0, global_step=0):
        self.eval()

        ret = OrderedDict([
            ('dummy_metrics', 0.0),
        ])
        self.train()
        return ret

    def backward(self):

        self.loss.backward()

    def update_EMA(self):
        update_moving_average(self.ema_df, self.df, self.ema_updater)

    def optimize_parameters(self):

        self.set_requires_grad([self.df], requires_grad=True)

        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.update_EMA()

    def get_current_errors(self):

        ret = OrderedDict([
            ('diffusion loss', self.loss.data),
        ])

        if hasattr(self, 'loss_gamma'):
            ret['gamma'] = self.loss_gamma.data

        return ret, self.stage_flag

    def get_current_visuals(self):

        with torch.no_grad():
            self.img_gen_df = render_sdf_dualoctree(self.renderer, self.sdfs, level=0,
                                                bbmin = self.bbmin, bbmax = self.bbmax,
                                                mesh_scale = self.vq_conf.data.test.point_scale, render_all = True)
            # self.img_gen_df = render_sdf(self.renderer, self.gen_df)

        vis_tensor_names = [
            'img_gen_df',
        ]

        vis_ims = self.tnsrs2ims(vis_tensor_names)
        visuals = zip(vis_tensor_names, vis_ims)

        return OrderedDict(visuals)

    def save(self, label, global_iter, save_opt=True):

        state_dict = {
            'df': self.df_module.state_dict(),
            'ema_df': self.ema_df.state_dict(),
            'opt': self.optimizer.state_dict(),
            'global_step': global_iter,
        }

        # if save_opt:
        #     state_dict['opt'] = self.optimizer.state_dict()

        save_filename = 'df_%s.pth' % (label)
        save_path = os.path.join(self.opt.ckpt_dir, save_filename)

        ckpts = os.listdir(self.opt.ckpt_dir)
        ckpts = [ck for ck in ckpts if ck!='df_steps-latest.pth']
        ckpts.sort(key=lambda x: int(x[9:-4]))
        if len(ckpts) > self.opt.ckpt_num:
            for ckpt in ckpts[:-self.opt.ckpt_num]:
                os.remove(os.path.join(self.opt.ckpt_dir, ckpt))

        torch.save(state_dict, save_path)

    def load_ckpt(self, ckpt, load_opt=True):
        map_fn = lambda storage, loc: storage
        if type(ckpt) == str:
            state_dict = torch.load(ckpt, map_location=map_fn)
        else:
            state_dict = ckpt

        # self.vqvae.load_state_dict(state_dict['vqvae'])
        self.df.load_state_dict(state_dict['df'])
        self.ema_df.load_state_dict(state_dict['ema_df'])
        self.start_iter = state_dict['global_step']
        print(colored('[*] weight successfully load from: %s' % ckpt, 'blue'))

        if load_opt:
            self.optimizer.load_state_dict(state_dict['opt'])
            print(colored('[*] optimizer successfully restored from: %s' % ckpt, 'blue'))
