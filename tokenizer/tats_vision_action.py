# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import math
import argparse
import numpy as np
import pickle as pkl

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .utils import shift_dim, adopt_weight, comp_getattr
from .modules import LPIPS, Codebook

def silu(x):
    return x*torch.sigmoid(x)

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)

def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss

class VQGANVisionAction(pl.LightningModule):
    '''
    Add an action encoder to the Video VQGAN model
    the action space is encoded by a separate encoder, and decoded by a separate decoder
    action input is a 7-dim vector, representing xyz, rpy, gripper
    but both action and visual encodings are quantized by the same codebook
    after both encoder, we add attention layers to fuse the action and visual encodings
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes
        self.automatic_optimization = False

        if not hasattr(args, 'padding_type'):
            args.padding_type = 'replicate'
        self.encoder = Encoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type, args.padding_type)
        self.decoder = Decoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, args.embedding_dim, 1, padding_type=args.padding_type)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, self.enc_out_ch, 1)

        self.action_encoder = ActionEncoderStack(args.action_dim, args.action_hidden_dim, args.embedding_dim)
        activations = [getattr(torch, args.action_activation[i]) if args.action_activation[i] != 'none' else torch.nn.Identity() for i in range(len(args.action_activation))]
        self.action_decoder = ActionDecoderStack(args.embedding_dim, args.action_hidden_dim, args.action_dim, activations)

        attn_layer = nn.TransformerDecoderLayer(d_model=args.embedding_dim, nhead=8)
        self.video_action_attn = nn.TransformerDecoder(attn_layer, num_layers=args.video_action_layers)
        
        self.codebook = Codebook(args.n_codes, args.embedding_dim, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)

        self.gan_feat_weight = args.gan_feat_weight
        self.image_discriminator = NLayerDiscriminator(args.image_channels, args.disc_channels, args.disc_layers)
        self.video_discriminator = NLayerDiscriminator3D(args.image_channels, args.disc_channels, args.disc_layers)
        
        if args.disc_loss_type == 'vanilla':
            self.disc_loss = vanilla_d_loss
        elif args.disc_loss_type == 'hinge':
            self.disc_loss = hinge_d_loss

        self.perceptual_model = LPIPS().eval()

        self.image_gan_weight = args.image_gan_weight
        self.video_gan_weight = args.video_gan_weight

        self.perceptual_weight = args.perceptual_weight

        self.l1_weight = args.l1_weight
        self.save_hyperparameters()

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length//self.args.sample_every_n_frames, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, x_action, include_embeddings=False):
        z_vision = self.pre_vq_conv(self.encoder(x)) # B, embed_dim, t, h, w  *t, h, w is downsampled T, H, W*
        z_action = self.action_encoder(x_action).permute(0, 2, 1, 3) # B, embed_dim, T, 7

        v_shape = z_vision.shape
        a_shape = z_action.shape

        # cat the action embeddings to the visual embeddings, and do self-attention
        z_vision_action = torch.cat([z_vision.flatten(2), z_action.flatten(2)], dim=-1).permute(0, 2, 1) # B, (t*h*w+T*7), embed_dim
        z_vision_action = self.video_action_attn(z_vision_action, z_vision_action) # B, (t*h*w+T*7, embed_dim

        if self.args.wo_transformer_residual:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) # B, embed_dim, t, h, w
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) # B, embed_dim, T, 7
        else:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) + z_vision
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) + z_action

        vq_output = self.codebook(z_vision)
        vq_output_action = self.codebook(z_action.unsqueeze(-1))

        if include_embeddings:
            return (vq_output['embeddings'], vq_output['encodings']), (vq_output_action['embeddings'], vq_output_action['encodings'])
        else:
            return vq_output['encodings'], vq_output_action['encodings']

    def decode(self, encodings, encodings_action):
        h = F.embedding(encodings, self.codebook.embeddings) # B, t, h, w, embed_dim
        h = self.post_vq_conv(shift_dim(h, -1, 1)) # B, embed_dim, t, h, w
        visual_decoded = self.decoder(h) # B, T, C, H, W

        h_action = F.embedding(encodings_action, self.codebook.embeddings) # B, T, 7, embed_dim
        h_action = h_action.permute(0, 1, 3, 2) # B, T, embed_dim, 7
        action_decoded = self.action_decoder(h_action) # B, T, embed_dim, 7
        
        return visual_decoded, action_decoded

    def decode_video(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings) # B, t, h, w, embed_dim
        h = self.post_vq_conv(shift_dim(h, -1, 1)) # B, embed_dim, t, h, w
        return self.decoder(h) # B, T, C, H, W

    def decode_action(self, encodings): # encodings: B, T, 7
        h = F.embedding(encodings, self.codebook.embeddings) # B, T, 7, embed_dim
        h = h.permute(0, 1, 3, 2)
        return self.action_decoder(h)

    def forward(self, x, x_action, x_action_masked=None, opt_stage=None, log_image=False):
        B, C, T, H, W = x.shape 
        # x_action is in shape B, T, action_dim
        
        z_vision = self.pre_vq_conv(self.encoder(x)) # B, embed_dim, t, h, w  *t, h, w is downsampled T, H, W*
        z_action = self.action_encoder(x_action if x_action_masked is None else x_action_masked).permute(0, 2, 1, 3) # B, embed_dim, T, 7

        v_shape = z_vision.shape
        a_shape = z_action.shape

        # cat the action embeddings to the visual embeddings, and do self-attention
        z_vision_action = torch.cat([z_vision.flatten(2), z_action.flatten(2)], dim=-1).permute(0, 2, 1) # B, (t*h*w+T*7), embed_dim
        z_vision_action = self.video_action_attn(z_vision_action, z_vision_action) # B, (t*h*w+T*7, embed_dim

        if self.args.wo_transformer_residual:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) # B, embed_dim, t, h, w
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) # B, embed_dim, T, 7
        else:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) + z_vision
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) + z_action

        vq_output = self.codebook(z_vision)
        vq_output_action = self.codebook(z_action.unsqueeze(-1))

        vq_embeddings = vq_output['embeddings'] # B, embed_dim, t, h, w
        vq_embeddings_action = vq_output_action['embeddings'] # B, embed_dim, T, 7, 1
        
        x_recon = self.decoder(self.post_vq_conv(vq_embeddings))
        x_recon_action = self.action_decoder(vq_embeddings_action.squeeze(-1).permute(0, 2, 1, 3)) # B, T, embed_dim, 7

        recon_loss = F.l1_loss(x_recon, x) * self.l1_weight
        recon_loss_action = F.l1_loss(x_recon_action, x_action) * self.l1_weight

        frame_idx = torch.randint(0, T, [B]).to(x.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        if log_image:
            return frames, frames_recon, x, x_recon

        if opt_stage is not None:
            if opt_stage == 0: # autoencoder
                perceptual_loss = 0
                if self.perceptual_weight > 0:
                    perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight

                logits_image_fake, pred_image_fake = self.image_discriminator(frames_recon)
                logits_video_fake, pred_video_fake = self.video_discriminator(x_recon)
                g_image_loss = -torch.mean(logits_image_fake)
                g_video_loss = -torch.mean(logits_video_fake)
                g_loss = self.image_gan_weight*g_image_loss + self.video_gan_weight*g_video_loss
                
                disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
                aeloss = disc_factor * g_loss

                # gan feature matching loss
                image_gan_feat_loss = 0
                video_gan_feat_loss = 0
                feat_weights = 4.0 / (3 + 1)
                if self.image_gan_weight > 0:
                    logits_image_real, pred_image_real = self.image_discriminator(frames)
                    for i in range(len(pred_image_fake)-1):
                        image_gan_feat_loss += feat_weights * F.l1_loss(pred_image_fake[i], pred_image_real[i].detach()) * (self.image_gan_weight>0)
                if self.video_gan_weight > 0:
                    logits_video_real, pred_video_real = self.video_discriminator(x)
                    for i in range(len(pred_video_fake)-1):
                        video_gan_feat_loss += feat_weights * F.l1_loss(pred_video_fake[i], pred_video_real[i].detach()) * (self.video_gan_weight>0)

                gan_feat_loss = disc_factor * self.gan_feat_weight * (image_gan_feat_loss + video_gan_feat_loss)

                self.log_dict({'train/logits_image_fake': logits_image_fake.mean().detach(),
                                'train/logits_video_fake': logits_video_fake.mean().detach(),
                                'train/g_image_loss': g_image_loss,
                                'train/g_video_loss': g_video_loss,
                                'train/image_gan_feat_loss': image_gan_feat_loss,
                                'train/video_gan_feat_loss': video_gan_feat_loss,
                                'train/perceptual_loss': perceptual_loss,
                                'train/recon_loss': recon_loss,
                                'train/recon_loss_action': recon_loss_action,
                                'train/aeloss': aeloss,
                                'train/commitment_loss': vq_output['commitment_loss'],
                                'train/commitment_loss_action': vq_output_action['commitment_loss'],
                                'train/perplexity': vq_output['perplexity'], 
                                'train/perplexity_action': vq_output_action['perplexity']},
                                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                                batch_size=self.args.batch_size)

                return recon_loss, recon_loss_action, x_recon, x_recon_action, vq_output, vq_output_action, aeloss, perceptual_loss, gan_feat_loss

            if opt_stage == 1: # discriminator
                logits_image_real, _ = self.image_discriminator(frames.detach())
                logits_video_real, _ = self.video_discriminator(x.detach())

                logits_image_fake, _ = self.image_discriminator(frames_recon.detach())
                logits_video_fake, _ = self.video_discriminator(x_recon.detach())

                d_image_loss = self.disc_loss(logits_image_real, logits_image_fake)
                d_video_loss = self.disc_loss(logits_video_real, logits_video_fake)
                disc_factor = adopt_weight(self.global_step, threshold=self.args.discriminator_iter_start)
                discloss = disc_factor * (self.image_gan_weight*d_image_loss + self.video_gan_weight*d_video_loss)

                self.log_dict({'train/logits_image_real': logits_image_real.mean().detach(),
                                'train/logits_image_fake': logits_image_fake.mean().detach(),
                                'train/logits_video_real': logits_video_real.mean().detach(),
                                'train/logits_video_fake': logits_video_fake.mean().detach(),
                                'train/d_image_loss': d_image_loss,
                                'train/d_video_loss': d_video_loss,
                                'train/discloss': discloss},
                                prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True,
                                batch_size=self.args.batch_size)

                return discloss

        else: # opt_stage is None, i.e., validation
            perceptual_loss = self.perceptual_model(frames, frames_recon).mean() * self.perceptual_weight
            return recon_loss, recon_loss_action, x_recon, x_recon_action, vq_output, vq_output_action, perceptual_loss

    def training_step(self, batch, batch_idx):
        opt_ae, opt_disc = self.optimizers()
        sch_ae, sch_disc = self.lr_schedulers()
        
        x = batch['video']
        x_action = batch['actions']
        x_action_masked = batch['actions_masked'] if 'actions_masked' in batch else None

        recon_loss, recon_loss_action, _, _, vq_output, vq_output_action, aeloss, perceptual_loss, gan_feat_loss = self.forward(x, x_action, x_action_masked, opt_stage=0)
        commitment_loss = vq_output['commitment_loss']
        commitment_loss_action = vq_output_action['commitment_loss']
        loss_ae = recon_loss + recon_loss_action + commitment_loss + commitment_loss_action + \
                 aeloss + perceptual_loss + gan_feat_loss
        opt_ae.zero_grad()
        self.manual_backward(loss_ae)
        opt_ae.step()
        sch_ae.step()

        loss_disc = self.forward(x, x_action, x_action_masked, opt_stage=1)
        opt_disc.zero_grad()
        self.manual_backward(loss_disc)
        opt_disc.step()
        sch_disc.step()


    def validation_step(self, batch, batch_idx):
        x = batch['video']
        x_action = batch['actions']
        recon_loss, recon_loss_action, _, _, vq_output, vq_output_action, perceptual_loss = self.forward(x, x_action)
        self.log_dict({'val/recon_loss': recon_loss,
                       'val/recon_loss_action': recon_loss_action,
                       'val/perceptual_loss': perceptual_loss,
                       'val/perplexity': vq_output['perplexity'],
                       'val/perplexity_action': vq_output_action['perplexity'],
                       'val/commitment_loss': vq_output['commitment_loss'],
                       'val/commitment_loss_action': vq_output_action['commitment_loss']},
                       prog_bar=True, sync_dist=True,
                       batch_size=self.args.batch_size)
        
    def configure_optimizers(self):

        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.action_encoder.parameters())+
                                  list(self.action_decoder.parameters())+
                                  list(self.video_action_attn.parameters())+
                                  list(self.pre_vq_conv.parameters())+
                                  list(self.post_vq_conv.parameters())+
                                  list(self.codebook.parameters()),
                                  lr=self.args.lr, betas=(0.5, 0.9))
        scheduler_ae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ae, T_max=self.args.max_steps // 2, eta_min=0)
        opt_disc = torch.optim.Adam(list(self.image_discriminator.parameters())+
                                    list(self.video_discriminator.parameters()),
                                    lr=self.args.lr, betas=(0.5, 0.9))
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_max=self.args.max_steps // 2, eta_min=0)

        return [opt_ae, opt_disc], [scheduler_ae, scheduler_disc]

    def log_images(self, batch, **kwargs):
        log = dict()
        x = batch['video']
        x_action = batch['actions']
        frames, frames_rec, _, _ = self(x, x_action, log_image=True)
        log["inputs"] = frames
        log["reconstructions"] = frames_rec
        return log

    def log_videos(self, batch, **kwargs):
        log = dict()
        x = batch['video']
        x_action = batch['actions']
        _, _, x, x_rec = self(x, x_action, log_image=True)
        log["inputs"] = x
        log["reconstructions"] = x_rec
        return log

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return torch.nn.SyncBatchNorm(in_channels)


class Encoder(nn.Module):
    def __init__(self, n_hiddens, downsample, image_channel=3, norm_type='group', padding_type='replicate'):
        super().__init__()
        n_times_downsample = np.array([int(math.log2(d)) for d in downsample])
        self.conv_blocks = nn.ModuleList()
        max_ds = n_times_downsample.max()

        self.conv_first = SamePadConv3d(image_channel, n_hiddens, kernel_size=3, padding_type=padding_type)

        for i in range(max_ds):
            block = nn.Module()
            in_channels = n_hiddens * 2**i
            out_channels = n_hiddens * 2**(i+1)
            stride = tuple([2 if d > 0 else 1 for d in n_times_downsample])
            block.down = SamePadConv3d(in_channels, out_channels, 4, stride=stride, padding_type=padding_type)
            block.res = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_downsample -= 1

        self.final_block = nn.Sequential(
            Normalize(out_channels, norm_type), 
            SiLU()
        )

        self.out_channels = out_channels

    def forward(self, x):
        h = self.conv_first(x)
        for block in self.conv_blocks:
            h = block.down(h)
            h = block.res(h)
        h = self.final_block(h)
        return h


class Decoder(nn.Module):
    def __init__(self, n_hiddens, upsample, image_channel, norm_type='group'):
        super().__init__()

        n_times_upsample = np.array([int(math.log2(d)) for d in upsample])
        max_us = n_times_upsample.max()
        
        in_channels = n_hiddens*2**max_us
        self.final_block = nn.Sequential(
            Normalize(in_channels, norm_type),
            SiLU()
        )

        self.conv_blocks = nn.ModuleList()
        for i in range(max_us):
            block = nn.Module()
            in_channels = in_channels if i ==0 else n_hiddens*2**(max_us-i+1)
            out_channels = n_hiddens*2**(max_us-i)
            us = tuple([2 if d > 0 else 1 for d in n_times_upsample])
            block.up = SamePadConvTranspose3d(in_channels, out_channels, 4, stride=us)
            block.res1 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            block.res2 = ResBlock(out_channels, out_channels, norm_type=norm_type)
            self.conv_blocks.append(block)
            n_times_upsample -= 1

        self.conv_last = SamePadConv3d(out_channels, image_channel, kernel_size=3)
    def forward(self, x):
        h = self.final_block(x)
        for i, block in enumerate(self.conv_blocks):
            h = block.up(h)
            h = block.res1(h)
            h = block.res2(h)
        h = self.conv_last(h)
        return h




class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0, norm_type='group', padding_type='replicate'):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm2 = Normalize(in_channels, norm_type)
        self.conv2 = SamePadConv3d(out_channels, out_channels, kernel_size=3, padding_type=padding_type)
        if self.in_channels != self.out_channels:
            self.conv_shortcut = SamePadConv3d(in_channels, out_channels, kernel_size=3, padding_type=padding_type)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x)

        return x+h


# Does not support dilation
class SamePadConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        # assumes that the input shape is divisible by stride
        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x):
        return self.conv(F.pad(x, self.pad_input, mode=self.padding_type))


class SamePadConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, padding_type='replicate'):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3
        if isinstance(stride, int):
            stride = (stride,) * 3

        total_pad = tuple([k - s for k, s in zip(kernel_size, stride)])
        pad_input = []
        for p in total_pad[::-1]: # reverse since F.pad starts from last dim
            pad_input.append((p // 2 + p % 2, p // 2))
        pad_input = sum(pad_input, tuple())
        self.pad_input = pad_input
        self.padding_type = padding_type

        self.convt = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                        stride=stride, bias=bias,
                                        padding=tuple([k - 1 for k in kernel_size]))

    def forward(self, x):
        return self.convt(F.pad(x, self.pad_input, mode=self.padding_type))

        
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
    # def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None

class NLayerDiscriminator3D(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.SyncBatchNorm, use_sigmoid=False, getIntermFeat=True):
        super(NLayerDiscriminator3D, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv3d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv3d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[-1], res[1:]
        else:
            return self.model(input), None

class PositionalEncoding(nn.Module):
	def __init__(self, min_deg=0, max_deg=10):
		super(PositionalEncoding, self).__init__()
		self.min_deg = min_deg
		self.max_deg = max_deg
		self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])

	def forward(self, x):
		# x: B*3
		x_ = x
		shape = list(x.shape[:-1]) + [-1]
		x_enc = (x[..., None, :] * self.scales[:, None].to(x.device)).reshape(shape)
		x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)

		# PE
		x_ret = torch.sin(x_enc)
		x_ret = torch.cat([x_ret, x_], dim=-1) # B*(6*(max_deg-min_deg)+3)
		return x_ret

class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, min_deg=0, max_deg=10):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.pos_enc = PositionalEncoding(min_deg, max_deg)
        self.fc1 = nn.Linear(input_dim * (2 * (self.pos_enc.max_deg - self.pos_enc.min_deg) + 1), hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, embed_dim * input_dim)
        self.final_block = nn.Sequential(
            SiLU(),
            nn.LayerNorm(embed_dim * input_dim)
        )

    def forward(self, x):
        x = self.pos_enc(x)
        h = self.fc1(x)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.final_block(h)
        h = h.reshape(*x.shape[:-1], self.embed_dim, self.input_dim)
        return h

class ActionEncoderStack(nn.Module):
    '''
    stack multiple action encoders with the same hidden dim and embed dim
    but different input dim
    '''
    def __init__(self, input_dims, hidden_dim, embed_dim, min_deg=0, max_deg=10):
        super().__init__()
        self.input_dims = input_dims # a tuple
        self.encoders = nn.ModuleList()
        for input_dim in input_dims:
            self.encoders.append(ActionEncoder(input_dim, hidden_dim, embed_dim, min_deg, max_deg))

    def forward(self, x):
        # split x into B, T, input_dim
        x_split = torch.split(x, self.input_dims, dim=-1) # (B, T, input_dim) 
        return torch.cat([encoder(x_) for encoder, x_ in zip(self.encoders, x_split)], dim=-1) # B, T, embed_dim, 7 (sum of input_dims)

class ActionDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, output_dim, activation=torch.tanh):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim * output_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.start_block = nn.Sequential(
            SiLU(),
            nn.LayerNorm(embed_dim * output_dim)
        )
        self.activation = activation

    def forward(self, x):
        # x is in shape B, T, embed_dim
        # for the output, the last entry falls into [0, 1] while other entries are in [-1, 1]
        h = self.start_block(x)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.activation(h)
        # h = torch.cat([torch.tanh(h[:, :-1]), torch.sigmoid(h[:, -1:])], dim=1)
        return h

class ActionDecoderStack(nn.Module):
    '''
    stack multiple action decoders with the same hidden dim and embed dim
    but different output dim
    '''
    def __init__(self, embed_dim, hidden_dim, output_dims, activations):
        super().__init__()
        self.output_dims = output_dims # a tuple
        self.decoders = nn.ModuleList()
        for (output_dim, activation) in zip(output_dims, activations):
            self.decoders.append(ActionDecoder(embed_dim, hidden_dim, output_dim, activation))

    def forward(self, x):
        # x is in shape B, T, embed_dim, 7
        x_split = torch.split(x, self.output_dims, dim=-1)
        return torch.cat([decoder(x_.flatten(-2)) for decoder, x_ in zip(self.decoders, x_split)], dim=-1)

class VQGANVisionActionEval(nn.Module):
    '''
    Add an action encoder to the Video VQGAN model
    the action space is encoded by a separate encoder, and decoded by a separate decoder
    action input is a 7-dim vector, representing xyz, rpy, gripper
    but both action and visual encodings are quantized by the same codebook
    after both encoder, we add attention layers to fuse the action and visual encodings
    '''
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = args.n_codes

        if not hasattr(args, 'padding_type'):
            args.padding_type = 'replicate'
        self.encoder = Encoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type, args.padding_type)
        self.decoder = Decoder(args.n_hiddens, args.downsample, args.image_channels, args.norm_type)
        self.enc_out_ch = self.encoder.out_channels
        self.pre_vq_conv = SamePadConv3d(self.enc_out_ch, args.embedding_dim, 1, padding_type=args.padding_type)
        self.post_vq_conv = SamePadConv3d(args.embedding_dim, self.enc_out_ch, 1)

        self.action_encoder = ActionEncoderStack(args.action_dim, args.action_hidden_dim, args.embedding_dim)
        activations = [getattr(torch, args.action_activation[i]) if args.action_activation[i] != 'none' else torch.nn.Identity() for i in range(len(args.action_activation))]
        self.action_decoder = ActionDecoderStack(args.embedding_dim, args.action_hidden_dim, args.action_dim, activations)

        attn_layer = nn.TransformerDecoderLayer(d_model=args.embedding_dim, nhead=8)
        self.video_action_attn = nn.TransformerDecoder(attn_layer, num_layers=args.video_action_layers)
        
        self.codebook = Codebook(args.n_codes, args.embedding_dim, no_random_restart=args.no_random_restart, restart_thres=args.restart_thres)

    @property
    def latent_shape(self):
        input_shape = (self.args.sequence_length//self.args.sample_every_n_frames, self.args.resolution,
                       self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape,
                                             self.args.downsample)])

    def encode(self, x, x_action, include_embeddings=False):
        z_vision = self.pre_vq_conv(self.encoder(x)) # B, embed_dim, t, h, w  *t, h, w is downsampled T, H, W*
        z_action = self.action_encoder(x_action).permute(0, 2, 1, 3) # B, embed_dim, T, 7

        v_shape = z_vision.shape
        a_shape = z_action.shape

        # cat the action embeddings to the visual embeddings, and do self-attention
        z_vision_action = torch.cat([z_vision.flatten(2), z_action.flatten(2)], dim=-1).permute(0, 2, 1) # B, (t*h*w+T*7), embed_dim
        z_vision_action = self.video_action_attn(z_vision_action, z_vision_action) # B, (t*h*w+T*7, embed_dim

        if self.args.wo_transformer_residual:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) # B, embed_dim, t, h, w
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) # B, embed_dim, T, 7
        else:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) + z_vision
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) + z_action

        vq_output = self.codebook(z_vision)
        vq_output_action = self.codebook(z_action.unsqueeze(-1))

        if include_embeddings:
            return (vq_output['embeddings'], vq_output['encodings']), (vq_output_action['embeddings'], vq_output_action['encodings'])
        else:
            return vq_output['encodings'], vq_output_action['encodings']

    def decode(self, encodings, encodings_action):
        h = F.embedding(encodings, self.codebook.embeddings) # B, t, h, w, embed_dim
        h = self.post_vq_conv(shift_dim(h, -1, 1)) # B, embed_dim, t, h, w
        visual_decoded = self.decoder(h) # B, T, C, H, W

        h_action = F.embedding(encodings_action, self.codebook.embeddings) # B, T, 7, embed_dim
        h_action = h_action.permute(0, 1, 3, 2) # B, T, embed_dim, 7
        action_decoded = self.action_decoder(h_action) # B, T, embed_dim, 7
        
        return visual_decoded, action_decoded

    def decode_video(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings) # B, t, h, w, embed_dim
        h = self.post_vq_conv(shift_dim(h, -1, 1)) # B, embed_dim, t, h, w
        return self.decoder(h) # B, T, C, H, W

    def decode_action(self, encodings): # encodings: B, T, 7
        h = F.embedding(encodings, self.codebook.embeddings) # B, T, 7, embed_dim
        h = h.permute(0, 1, 3, 2)
        return self.action_decoder(h)

    def forward(self, x, x_action, x_action_masked=None, opt_stage=None, log_image=False):
        B, C, T, H, W = x.shape 
        # x_action is in shape B, T, action_dim
        
        z_vision = self.pre_vq_conv(self.encoder(x)) # B, embed_dim, t, h, w  *t, h, w is downsampled T, H, W*
        z_action = self.action_encoder(x_action if x_action_masked is None else x_action_masked).permute(0, 2, 1, 3) # B, embed_dim, T, 7

        v_shape = z_vision.shape
        a_shape = z_action.shape

        # cat the action embeddings to the visual embeddings, and do self-attention
        z_vision_action = torch.cat([z_vision.flatten(2), z_action.flatten(2)], dim=-1).permute(0, 2, 1) # B, (t*h*w+T*7), embed_dim
        z_vision_action = self.video_action_attn(z_vision_action, z_vision_action) # B, (t*h*w+T*7, embed_dim

        if self.args.wo_transformer_residual:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) # B, embed_dim, t, h, w
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) # B, embed_dim, T, 7
        else:
            z_vision = z_vision_action[:, :v_shape[2]*v_shape[3]*v_shape[4]].permute(0, 2, 1).reshape(v_shape) + z_vision
            z_action = z_vision_action[:, v_shape[2]*v_shape[3]*v_shape[4]:].permute(0, 2, 1).reshape(a_shape) + z_action

        vq_output = self.codebook(z_vision)
        vq_output_action = self.codebook(z_action.unsqueeze(-1))

        vq_embeddings = vq_output['embeddings'] # B, embed_dim, t, h, w
        vq_embeddings_action = vq_output_action['embeddings'] # B, embed_dim, T, 7, 1
        
        x_recon = self.decoder(self.post_vq_conv(vq_embeddings))
        x_recon_action = self.action_decoder(vq_embeddings_action.squeeze(-1).permute(0, 2, 1, 3)) # B, T, embed_dim, 7

        frame_idx = torch.randint(0, T, [B]).to(x.device)
        frame_idx_selected = frame_idx.reshape(-1, 1, 1, 1, 1).repeat(1, C, 1, H, W)
        frames = torch.gather(x, 2, frame_idx_selected).squeeze(2)
        frames_recon = torch.gather(x_recon, 2, frame_idx_selected).squeeze(2)

        return x_recon, x_recon_action, vq_output, vq_output_action
    