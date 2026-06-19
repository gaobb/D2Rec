import torch
import torch.nn as nn
import math
import numpy as np
import random
import os
import cv2


from models import vit_encoder
from .dinov1.utils import trunc_normal_
from models.vision_transformer import Block as VitBlock, Attention, LinearAttention, LinearAttention2
from utils.utils import l2_normalize
from functools import partial
import time


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, feature_size, num_pos_feats=128):
        super().__init__()
        self.feature_size = feature_size  # H, W
        self.row_embed = nn.Embedding(feature_size[0], num_pos_feats)
        self.col_embed = nn.Embedding(feature_size[1], num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor):
        i = torch.arange(self.feature_size[1], device=tensor.device)  # W
        j = torch.arange(self.feature_size[0], device=tensor.device)  # H
        x_emb = self.col_embed(i)  # W x C // 2
        y_emb = self.row_embed(j)  # H x C // 2

        pos = torch.cat(
            [
                torch.cat(
                    [x_emb.unsqueeze(0)] * self.feature_size[0], dim=0
                ),  # H x W x C // 2
                torch.cat(
                    [y_emb.unsqueeze(1)] * self.feature_size[1], dim=1
                ),  # H x W x C // 2
            ],
            dim=-1,
        ).flatten(
            0, 1
        )  # (H X W) X C
        
        return pos

def build_position_embedding(pos_embed_type, feature_size, hidden_dim):
    '''
    if pos_embed_type in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        pos_embed = PositionEmbeddingSine(feature_size, hidden_dim // 2, normalize=True)
    '''
    if pos_embed_type in ("v3", "learned"):
        pos_embed = PositionEmbeddingLearned(feature_size, hidden_dim // 2)
    else:
        raise ValueError(f"not supported {pos_embed_type}")
    return pos_embed

class MaskHead(nn.Module):
    def __init__(self, inplanes, conv_dims=256, num_upconv=2):
        super(MaskHead, self).__init__()
        self.upconvs = []
        for k in range(num_upconv):
            upconv = nn.Sequential(
                nn.Conv2d(
                    inplanes if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                nn.BatchNorm2d(conv_dims),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    conv_dims if num_upconv > 0 else inplanes,
                    conv_dims,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
            )
            self.add_module("upconv{}".format(k + 1), upconv)
            self.upconvs.append(upconv)

        self.predictor = nn.Conv2d(
                conv_dims,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
            )


    def forward(self, x):
        for layer in self.upconvs:
            x = layer(x)

        return self.predictor(x).sigmoid()

class D2Rec(nn.Module):
    def __init__(
            self,
            backbone,
            dual_mask, 
            dual_mask_type,
            mask_head,
            embed_dim, 
            num_heads,
            image_size,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            backbone_require_grad_layer=[],
    ) -> None:
        super(D2Rec, self).__init__()
        self.backbone = backbone
        self.dual_mask = dual_mask
        self.dual_mask_type = dual_mask_type
        self.mask_head = mask_head
        self.decoder = self.build_decoder(embed_dim, num_heads)
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        

        self.remove_class_token = remove_class_token
        self.backbone_require_grad_layer = backbone_require_grad_layer
       
        if self.dual_mask:
            feat_size = image_size // 14
            pos_embed_type = 'learned'
            if self.dual_mask_type == 'spatial':
                # spatial
                self.pos_embed = build_position_embedding(pos_embed_type, [feat_size, feat_size], embed_dim)
                self.mask_token = build_position_embedding(pos_embed_type, [1,1], embed_dim)
            elif self.dual_mask_type == 'channel':
                # channel
                self.pos_embed = build_position_embedding(pos_embed_type, [1, 1], embed_dim)
                self.mask_token = build_position_embedding(pos_embed_type, [feat_size, feat_size], embed_dim)
            elif self.dual_mask_type == 'spatialchannel':
                # spatial and chnnel
                self.pos_embed = build_position_embedding(pos_embed_type, [feat_size, feat_size], embed_dim)
                self.mask_token = build_position_embedding(pos_embed_type, [1,1], embed_dim)

        if self.mask_head:
            self.refiner = MaskHead(embed_dim*2, 128, 2) # 2024-01-1
      
        if not hasattr(self.backbone, 'num_register_tokens'):
            self.backbone.num_register_tokens = 0
    

    def build_decoder(self, embed_dim, num_heads):
        decoder = []
        for i in range(8):
            blk = VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8),
                        attn=LinearAttention2)
            decoder.append(blk)

        decoder = nn.ModuleList(decoder)
        for m in decoder.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        return decoder
        
    
    def generate_spatial_mask(self, x, p):
        B, HW, C = x.shape
        # Step 1: Generate a random noise matrix
        noise = torch.rand(HW, B, device=x.device)

        # Step 2: Sort the noise matrix along the first dimension
        sorted_noise, indices = torch.sort(noise, dim=0)

        # Step 3: Create a mask with the same shape as the noise matrix
        mask = torch.zeros_like(noise, device=x.device)

        # Step 4: Set the first H*W*p elements to 1 for each batch
        num_elements = int(HW * p)
        mask[:num_elements, :] = 1

        # Step 5: Reorder the mask to match the original noise order
        mask = mask.scatter(0, indices, mask)

        # Step 6: Reshape the mask to the desired shape (B, 1, H, W)
        mask = mask.t().reshape(B, HW, 1)

        return mask
    
    def generate_channel_mask(self, x, p):
        B, HW, C = x.shape
        # Step 1: Generate a random noise matrix
        noise = torch.rand(C, B, device=x.device)

        # Step 2: Sort the noise matrix along the first dimension
        sorted_noise, indices = torch.sort(noise, dim=0)

        # Step 3: Create a mask with the same shape as the noise matrix
        mask = torch.zeros_like(noise, device=x.device)

        # Step 4: Set the first H*W*p elements to 1 for each batch
        num_elements = int(C * p)
        mask[:num_elements, :] = 1

        # Step 5: Reorder the mask to match the original noise order
        mask = mask.scatter(0, indices, mask)

        # Step 6: Reshape the mask to the desired shape (B, 1, H, W)
        mask = mask.t().reshape(B, 1, C)

        return mask
    
    def generate_spatialchannel_mask(self, x, p):
        B, HW, C = x.shape
        # Step 1: Generate a random noise matrix
        noise = torch.rand(HW * C, B, device=x.device)

        # Step 2: Sort the noise matrix along the first dimension
        sorted_noise, indices = torch.sort(noise, dim=0)

        # Step 3: Create a mask with the same shape as the noise matrix
        mask = torch.zeros_like(noise, device=x.device)

        # Step 4: Set the first H*W*p elements to 1 for each batch
        num_elements = int(HW * C * p)
        mask[:num_elements, :] = 1

        # Step 5: Reorder the mask to match the original noise order
        mask = mask.scatter(0, indices, mask)

        # Step 6: Reshape the mask to the desired shape (B, 1, H, W)
        mask = mask.t().reshape(B, HW, C)

        return mask
      
    def forward(self, x, training=True):
        batch_size = x.shape[0]
        x = self.backbone.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.backbone.blocks):
            if i <= self.target_layers[-1]:
                if i in self.backbone_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.backbone.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.backbone.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        
        if self.dual_mask:
            pos_embed = self.pos_embed(x)  # (H x W) x C
            mask_token = self.mask_token(x) # (1 x 1) x C

            mask_token = mask_token + pos_embed # (H x W) x C = (H x W) x C + (1 x 1) x C 
            mask_token = mask_token.unsqueeze(0).repeat(x.shape[0], 1, 1)

            if self.dual_mask_type == 'spatial':
                mask = self.generate_spatial_mask(x[:,1 + self.backbone.num_register_tokens:,:], p=0.5)
                
            elif self.dual_mask_type == 'channel':
                mask = self.generate_channel_mask(x[:,1 + self.backbone.num_register_tokens:,:], p=0.5)

            elif self.dual_mask_type == 'spatialchannel':
                mask = self.generate_spatialchannel_mask(x[:,1 + self.backbone.num_register_tokens:,:], p=0.5)
            

            x1 =  x[:,1 + self.backbone.num_register_tokens:,:] * mask + mask_token * (1-mask)
            x2 =  x[:,1 + self.backbone.num_register_tokens:,:] * (1 - mask) + mask_token * mask

            x1 = torch.cat([x[:,:1 + self.backbone.num_register_tokens,:], x1], 1)
            x2 = torch.cat([x[:,:1 + self.backbone.num_register_tokens,:], x2], 1)

            x = torch.cat([x1, x2],0)


        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=None)
            de_list.append(x)
        de_list = de_list[::-1]
        
        
        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]
        

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.backbone.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.backbone.num_register_tokens:, :] for d in de]

        if self.dual_mask:
            de = [(1-mask) * d[:d.shape[0]//2] + mask * d[d.shape[0]//2:] for d in de] # 互补
            
        en = [e.permute(0, 2, 1).reshape([batch_size, -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([batch_size, -1, side, side]).contiguous() for d in de]
        
        if self.mask_head:
            if training:
                # record original and reconstructed tokens of pseudo images
                pen = [e[batch_size//2:] for e in en]  
                pde = [d[batch_size//2:] for d in de]

                # return original and reconstructed tokens of normal image
                en = [e[:batch_size//2] for e in en]  
                de = [d[:batch_size//2] for d in de]

                consine_diffs = [
                    - l2_normalize(feat_t) * l2_normalize(feat_s)
                    for feat_t, feat_s in zip(pen, pde)
                    ]
            else:
                consine_diffs = [
                    - l2_normalize(feat_t) * l2_normalize(feat_s)
                    for feat_t, feat_s in zip(en, de)
                    ]

            consine_diffs = torch.cat(consine_diffs, dim=1)
            pred_masks  = self.refiner(consine_diffs.detach())
        else:
            pred_masks = None
           

        return en, de, pred_masks

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)
  

def build_d2rec(encoder_name: str = 'dinov2reg_vit_base_14', 
                dual_mask: bool=False, 
                dual_mask_type: str='channel',
                mask_head: bool=False,
                image_size: int = 224):

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    backbone = vit_encoder.load(encoder_name)
    for p in backbone.parameters():
        p.requires_grad = False

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise "Architecture not in small, base, large."
    #ViTRD
    model =  D2Rec(backbone=backbone,
                  dual_mask=dual_mask, 
                  dual_mask_type=dual_mask_type,
                  mask_head=mask_head,
                  embed_dim=embed_dim, 
                  num_heads=num_heads,
                  image_size=image_size, 
                  target_layers=target_layers,
                  fuse_layer_encoder=fuse_layer_encoder,
                  fuse_layer_decoder=fuse_layer_decoder
                )
    
    return model