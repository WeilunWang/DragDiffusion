# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from socket import has_dualstack_ipv6
import sys
import copy
import traceback
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.cm
import dnnlib
from diffusers.pipelines import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler
#----------------------------------------------------------------------------

class CapturedException(Exception):
    def __init__(self, msg=None):
        if msg is None:
            _type, value, _traceback = sys.exc_info()
            assert value is not None
            if isinstance(value, CapturedException):
                msg = str(value)
            else:
                msg = traceback.format_exc()
        assert isinstance(msg, str)
        super().__init__(msg)

#----------------------------------------------------------------------------

class CaptureSuccess(Exception):
    def __init__(self, out):
        super().__init__()
        self.out = out

#----------------------------------------------------------------------------

def add_watermark_np(input_image_array, watermark_text="AI Generated"):
    image = Image.fromarray(np.uint8(input_image_array)).convert("RGBA")

    # Initialize text image
    txt = Image.new('RGBA', image.size, (255, 255, 255, 0))
    font = ImageFont.truetype('arial.ttf', round(25/512*image.size[0]))
    d = ImageDraw.Draw(txt)

    text_width, text_height = font.getsize(watermark_text)
    text_position = (image.size[0] - text_width - 10, image.size[1] - text_height - 10)
    text_color = (255, 255, 255, 128)  # white color with the alpha channel set to semi-transparent

    # Draw the text onto the text canvas
    d.text(text_position, watermark_text, font=font, fill=text_color)

    # Combine the image with the watermark
    watermarked = Image.alpha_composite(image, txt)
    watermarked_array = np.array(watermarked)
    return watermarked_array

#----------------------------------------------------------------------------

class Renderer:
    def __init__(self, disable_timing=False):
        self._device        = torch.device('cuda')
        self._pkl_data      = dict()    # {pkl: dict | CapturedException, ...}
        self._networks      = dict()    # {cache_key: torch.nn.Module, ...}
        self._pinned_bufs   = dict()    # {(shape, dtype): torch.Tensor, ...}
        self._cmaps         = dict()    # {name: torch.Tensor, ...}
        self._is_timing     = False
        if not disable_timing:
            self._start_event   = torch.cuda.Event(enable_timing=True)
            self._end_event     = torch.cuda.Event(enable_timing=True)
        self._disable_timing = disable_timing
        self._net_layers    = dict()    # {cache_key: [dnnlib.EasyDict, ...], ...}

    def render(self, **args):
        if self._disable_timing:
            self._is_timing = False
        else:
            self._start_event.record(torch.cuda.current_stream(self._device))
            self._is_timing = True
        res = dnnlib.EasyDict()
        try:
            init_net = False
            if not hasattr(self, 'G'):
                init_net = True
            if hasattr(self, 'pkl'):
                if self.pkl != args['pkl']:
                    init_net = True
            if hasattr(self, 'w_load'):
                if self.w_load is not args['w_load']:
                    init_net = True
            if hasattr(self, 'w0_seed'):
                if self.w0_seed != args['w0_seed']:
                    init_net = True
            if hasattr(self, 'w_plus'):
                if self.w_plus != args['w_plus']:
                    init_net = True
            if args['reset_w']:
                init_net = True
            res.init_net = init_net
            if init_net:
                self.init_network(res, **args)
            self._render_drag_impl(res, **args)
        except:
            res.error = CapturedException()
        if not self._disable_timing:
            self._end_event.record(torch.cuda.current_stream(self._device))
        if 'image' in res:
            res.image = self.to_cpu(res.image).detach().numpy()
            res.image = add_watermark_np(res.image, 'AI Generated')
        if 'stats' in res:
            res.stats = self.to_cpu(res.stats).detach().numpy()
        if 'error' in res:
            res.error = str(res.error)
        # if 'stop' in res and res.stop:

        if self._is_timing and not self._disable_timing:
            self._end_event.synchronize()
            res.render_time = self._start_event.elapsed_time(self._end_event) * 1e-3
            self._is_timing = False
        return res

    def get_network(self, pkl, device, **kwargs):
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                  clip_sample=False, set_alpha_to_one=False)
        ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path="/home/wangwl/.cache/huggingface/diffusers/models--runwayml--stable-diffusion-v1-5/snapshots/aa9ba505e1973ae5cd05f5aedd345178f52f8e6a/",
                                                             scheduler=scheduler, torch_dtype=torch.float16, **kwargs).to(device)
        return ldm_stable

    def _get_pinned_buf(self, ref):
        key = (tuple(ref.shape), ref.dtype)
        buf = self._pinned_bufs.get(key, None)
        if buf is None:
            buf = torch.empty(ref.shape, dtype=ref.dtype).pin_memory()
            self._pinned_bufs[key] = buf
        return buf

    def to_device(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).to(self._device)

    def to_cpu(self, buf):
        return self._get_pinned_buf(buf).copy_(buf).clone()

    def _ignore_timing(self):
        self._is_timing = False

    def _apply_cmap(self, x, name='viridis'):
        cmap = self._cmaps.get(name, None)
        if cmap is None:
            cmap = matplotlib.cm.get_cmap(name)
            cmap = cmap(np.linspace(0, 1, num=1024), bytes=True)[:, :3]
            cmap = self.to_device(torch.from_numpy(cmap))
            self._cmaps[name] = cmap
        hi = cmap.shape[0] - 1
        x = (x * hi + 0.5).clamp(0, hi).to(torch.int64)
        x = torch.nn.functional.embedding(x, cmap)
        return x

    def init_network(self, res,
        pkl             = None,
        w0_seed         = 0,
        w_load          = None,
        w_plus          = True,
        noise_mode      = 'const',
        trunc_psi       = 0.7,
        trunc_cutoff    = None,
        input_transform = None,
        lr              = 0.001,
        **kwargs
        ):
        # Dig up network details.
        self.pkl = pkl
        G = self.get_network(pkl, 'cuda')
        self.G = G

        self.feat_refs = None
        self.points0_pt = None

    def set_z_and_z_optim(self, z, lr):
        self.z = z.detach().clone()
        self.z.requires_grad = True
        self.z_optim = torch.optim.Adam([self.z], lr=lr, eps=1e-4)
        print(f'Rebuild optimizer with lr: {lr}')

    def update_lr(self, lr):
        del self.z_optim
        self.z_optim = torch.optim.Adam([self.z], lr=lr, eps=1e-4)
        print(f'Rebuild optimizer with lr: {lr}')
        print('    Remain feat_refs and points0_pt')

# -------------------------------DDIM inversion------------------------------------

    @torch.no_grad()
    def _ddim_inversion(self, image, prompt, num_ddim_steps=50):
        latent = self.image2latent(image)
        text_input = self.G.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.G.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        cond_embeddings = self.G.text_encoder(text_input.input_ids.cuda())[0]
        self.G.scheduler.set_timesteps(num_ddim_steps)
        for ddim_latent in self.ddim_loop(latent, cond_embeddings, num_inverse_steps=40):
            ddim_image = self.latent2image(ddim_latent)
            yield ddim_latent, ddim_image

    @torch.no_grad()
    def _ddim_inference(self, latent, prompt, num_ddim_steps=50, num_inverse_steps=40):
        device = self.G._execution_device

        text_input = self.G.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.G.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self.G.text_encoder(text_input.input_ids.to(device))[0]

        self.G.scheduler.set_timesteps(num_ddim_steps, device=device)
        timesteps = self.G.scheduler.timesteps[-num_inverse_steps:]
        for i, t in enumerate(timesteps):
            noise_pred = self.G.unet(latent, t, encoder_hidden_states=prompt_embeds).sample
            latent = self.G.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
            sample = self.G.decode_latents(latent)[0]
            yield latent, sample

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.G.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        image = np.array(image)
        image = torch.from_numpy(image).half().cuda() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0)
        latents = self.G.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def ddim_loop(self, latent, cond_embeddings, num_inverse_steps=50):
        latent = latent.clone().detach()
        for i in range(num_inverse_steps):
            t = self.G.scheduler.timesteps[len(self.G.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            yield latent

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.G.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
                  sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(
            timestep - self.G.scheduler.config.num_train_timesteps // self.G.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.G.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.G.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.G.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

# ----------------------------------------------------------------------------

# ------------------------------ DragDiffusion -------------------------------

    def _render_drag_impl(self, res, timestep, prompt_embeds,
        points          = [],
        targets         = [],
        mask            = None,
        lambda_mask     = 0.1,
        r1              = 1,
        r2              = 3,
        img_scale_db    = 0,
        img_normalize   = False,
        is_drag         = False,
        reset           = False,
        to_pil          = False,
        img_resolution  = 512,
        **kwargs
    ):
        G = self.G
        z = self.z
        if hasattr(self, 'points'):
            if len(points) != len(self.points):
                reset = True
        if reset:
            self.feat_refs = None
            self.points0_pt = None
        self.points = points

        # Run synthesis network.
        noise_pred, feat = G.unet(z, timestep, encoder_hidden_states=prompt_embeds, return_dict=False, return_feat=True)
        with torch.no_grad():
            extra_step_kwargs = G.prepare_extra_step_kwargs(generator=None, eta=0.0)
            z_prev = G.scheduler.step(noise_pred.detach(), timestep, z.detach(), **extra_step_kwargs).prev_sample
            img = G.decode_latents(z_prev)[0]
            img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)

        h, w = img_resolution, img_resolution

        if is_drag:
            X = torch.linspace(0, h, h)
            Y = torch.linspace(0, w, w)
            xx, yy = torch.meshgrid(X, Y)
            feat_resize = F.interpolate(feat, [h, w], mode='bilinear')
            if self.feat_refs is None:
                self.feat0_resize = F.interpolate(feat.detach(), [h, w], mode='bilinear')
                self.z0_resize = F.interpolate(z.detach(), [h, w], mode='bilinear')
                self.feat_refs = []
                for point in points:
                    py, px = round(point[0]), round(point[1])
                    self.feat_refs.append(self.feat0_resize[:,:,py,px])
                self.points0_pt = torch.Tensor(points).unsqueeze(0).to(self._device) # 1, N, 2

            # Point tracking with feature matching
            with torch.no_grad():
                for j, point in enumerate(points):
                    r = round(r2 / 512 * h)
                    up = max(point[0] - r, 0)
                    down = min(point[0] + r + 1, h)
                    left = max(point[1] - r, 0)
                    right = min(point[1] + r + 1, w)
                    feat_patch = feat_resize[:,:,up:down,left:right]
                    L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
                    _, idx = torch.min(L2.view(1,-1), -1)
                    width = right - left
                    point = [idx.item() // width + up, idx.item() % width + left]
                    points[j] = point

            res.points = [[point[0], point[1]] for point in points]

            # Motion supervision
            loss_motion = 0
            res.stop = True
            for j, point in enumerate(points):
                direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
                if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
                    res.stop = False
                if torch.linalg.norm(direction) > 1:
                    distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
                    relis, reljs = torch.where(distance < round(r1 / 512 * h))
                    direction = direction / (torch.linalg.norm(direction) + 1e-7)
                    gridh = (relis-direction[1]) / (h-1) * 2 - 1
                    gridw = (reljs-direction[0]) / (w-1) * 2 - 1
                    grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
                    target = F.grid_sample(feat_resize.float(), grid, align_corners=True).squeeze(2)
                    loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs], target.detach())

            loss = loss_motion
            if mask is not None:
                if mask.min() == 0 and mask.max() == 1:
                    mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
                    z_resize = F.interpolate(z, [h, w], mode='bilinear')
                    loss_fix = F.l1_loss(z_resize * mask_usq, self.z0_resize.detach() * mask_usq)
                    loss += lambda_mask * loss_fix

            if not res.stop:
                self.z_optim.zero_grad()
                loss.backward(retain_graph=True)
                self.z_optim.step()

        # Scale and convert to uint8.
        img = img[0]
        if img_normalize:
            img = img / img.norm(float('inf'), dim=[1,2], keepdim=True).clip(1e-8, 1e8)
        img = img * (10 ** (img_scale_db / 20))
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).permute(1, 2, 0)
        if to_pil:
            from PIL import Image
            img = img.cpu().numpy()
            img = Image.fromarray(img)
        res.image = img

#----------------------------------------------------------------------------
