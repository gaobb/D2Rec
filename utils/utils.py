import torch
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from torch.nn import functional as F
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import math
from tabulate import tabulate
import logging
import torch.nn as nn
import pickle
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


def cal_anomaly_map(fs_list, ft_list, out_size=224):
    batch_size = fs_list[0].shape[0]
   
    a_map_list = []
    for item in range(len(ft_list)):
        a_map = 1 - torch.sum(fs_list[item] * ft_list[item], dim=1, keepdim=True)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)

    anomaly_map = torch.cat(a_map_list, dim=1).mean(dim=1, keepdim=True)

    return anomaly_map, a_map_list


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def l2_normalize(input, dim=1, eps=1e-12):
    denom = torch.sqrt(torch.sum(input**2, dim=dim, keepdim=True))
    return input / (denom + eps)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=np.float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cosine_loss_function(a, b):
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        loss += torch.mean(1 - torch.sum(a_ * b_, dim=1, keepdim=True))
      
    loss = loss / len(a)
    return loss

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def visualize(results_eval, save_path, obj_list, dataset):

    save_dir = os.path.join(save_path, 'visualizations-1218')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for cls_name in obj_list:
        idxes = results_eval['cls_names'] == cls_name
        pr_masks = results_eval['pr_masks'][idxes]
        img_paths = results_eval['img_paths'][idxes]
        gt_masks = results_eval['gt_masks'][idxes]
        
        img_paths = img_paths.tolist()

        for i in range(len(img_paths)):

            mask = gt_masks[i][0].cpu().numpy()

            if mask.sum() > 0:
                file_name = img_paths[i].split(dataset)[-1]

                filedir, file_name = os.path.split(file_name)
                _, defename = os.path.split(filedir)
                save_img_dir = os.path.join(save_dir, cls_name, defename)
                if not os.path.exists(save_img_dir):
                    os.makedirs(save_img_dir)

                img = cv2.imread(img_paths[i], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                #img = cv2.cvtColor(img, self.cvt_color)
                #img = cv2.imread(img_paths[i])
                
                pred = pr_masks[i, 0].cpu().numpy()
                img = cv2.resize(img, pred.shape)

                pred = pred[:, :, None].repeat(3, 2)

                # self normalize just for analysis
                score_map = np.uint8(normalize(pred)*255)
                score_map = cv2.applyColorMap(score_map, cv2.COLORMAP_JET)
            
                imgscore_map = apply_ad_scoremap(img, normalize(pred))
                imgscore_map = cv2.cvtColor(imgscore_map, cv2.COLOR_RGB2BGR)

 
                imgscore_path = os.path.join(save_img_dir, file_name)
                score_path = os.path.join(save_img_dir, "score-" + file_name)

                cv2.imwrite(imgscore_path, imgscore_map)
                cv2.imwrite(score_path, score_map)


def get_logger(save_path, log_file):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    txt_path = os.path.join(save_path, log_file)
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                    datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
