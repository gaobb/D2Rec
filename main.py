import torch
import torch.nn as nn
import math
from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
import cv2
from torch.utils.data import DataLoader, ConcatDataset

from dataset import Dataset
from models import build_d2rec
import torch.backends.cudnn as cudnn
import argparse
from utils.utils import cosine_loss_function, l2_normalize, cal_anomaly_map, get_gaussian_kernel, get_logger, WarmCosineScheduler, visualize
from torch.nn import functional as F
from utils import StableAdamW, Evaluator
import warnings
import copy
import logging
import itertools
from tqdm import tqdm
from tabulate import tabulate
import time
from torch.utils.data import Subset

warnings.filterwarnings("ignore")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DiceLoss(nn.Module):
   def __init__(self, smooth=1, p=2, reduction='mean'):
        super().__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

   def forward(self, pred_masks, gt_masks):
        predict = pred_masks.contiguous().view(pred_masks.shape[0], -1)
        target = gt_masks.contiguous().view(gt_masks.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

def run_inference(data_loader, cache_on_device):
    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, img_paths = [], [], [], [], [], []
    i = 0
    nums = 0
    total_time = 0 
    for items in tqdm(data_loader):
        img = items['img'].to(device)
        img_path = items['img_path']
        batch_size = img.shape[0] 
        
        cls_name = items['cls_name']
        cls_id = items['cls_id']

        gt_anomaly = items['anomaly'].to(device)
        gt_mask = items['img_mask'].to(device)
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        '''
        torch.cuda.synchronize()
        start_time = time.time()
        '''
        with torch.no_grad():
            ens, des, pred_masks = model(img, training=False)
        '''
        torch.cuda.synchronize()
        end_time = time.time()

        run_time = end_time - start_time
        
        if i > 0:
            total_time += run_time
            nums += img.shape[0]
        i += 1
        '''
        ens = [l2_normalize(en) for en in ens]
        des = [l2_normalize(de) for de in des]

        if resize_mask is not None:
            anomaly_map = cal_anomaly_map(ens, des, resize_mask)[0]
            anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
            if args.mask_head: 
                pred_masks = F.interpolate(pred_masks, size=resize_mask, mode='bilinear', align_corners=False)
            gt_mask = F.interpolate(gt_mask, size=resize_mask, mode='nearest')
        else:
            anomaly_map = cal_anomaly_map(ens, des, crop_size)[0]
            if args.mask_head: 
                pred_masks = F.interpolate(pred_masks, size=image_size, mode='bilinear', align_corners=False)

        if args.mask_head:
            anomaly_map = (anomaly_map + pred_masks)/2.0

        anomaly_map = gaussian_kernel(anomaly_map)
        topk = int(resize_mask * resize_mask * max_ratio) if resize_mask is not None else int(crop_size*crop_size * max_ratio)
        if max_ratio <= 0:
            anomaly_map_max, _ = torch.max(anomaly_map.view(batch_size, -1), dim=1)
        else:
            anomaly_map_topk, _ = torch.topk(anomaly_map.view(batch_size, -1), k=topk,  dim=1)
            anomaly_map_max = anomaly_map_topk.mean(dim=1)
        
        if cache_on_device:
            gt_masks.append(gt_mask.int())
            pr_masks.append(anomaly_map.to(device))

            cls_names.append(np.array(cls_name))
            img_paths.append(np.array(img_path))

            gt_anomalys.append(gt_anomaly.int())
            pr_anomalys.append(anomaly_map_max.to(device))
        else:
            gt_masks.append(gt_mask.int().cpu())
            pr_masks.append(anomaly_map.cpu())
            
            cls_names.append(np.array(cls_name))
            img_paths.append(np.array(img_path))

            gt_anomalys.append(gt_anomaly.int().cpu())
            pr_anomalys.append(anomaly_map_max.cpu())
    
    '''
    print("total_times", total_time)
    print("total_nums", nums)
    print("average times", total_time/nums)
    print("fps", nums/total_time) 
    '''
    results_eval = dict(gt_masks=gt_masks, pr_masks=pr_masks, cls_names=cls_names, gt_anomalys=gt_anomalys, pr_anomalys=pr_anomalys, img_paths=img_paths)
    results_eval = {k: np.concatenate(v, axis=0) if k == 'cls_names' or k == 'img_paths' else torch.cat(v, dim=0) for k, v in results_eval.items()}
    return results_eval

def update_msg(msg, metric_results, cls_name, idx):
    msg['Name'] = msg.get('Name', [])
    msg['Name'].append(cls_name)
    avg_act = True if len(obj_list) > 1 and idx == len(obj_list) - 1 else False
    msg['Name'].append('Avg') if avg_act else None

    cls_metric_msg = {}
    for metric in args.eval_metrics:
        metric_result = metric_results[metric] * 100
        cls_metric_msg[metric] = metric_result

        msg[metric] = msg.get(metric, [])
        msg[metric].append(metric_result)

        if avg_act:
            metric_result_avg = sum(msg[metric]) / len(msg[metric])
            msg[metric].append(metric_result_avg)

    print('{}: {}'.format(cls_name, ', '.join(['{}: {:.1f}'.format(metric, cls_metric_msg[metric]) for metric in args.eval_metrics])))


def test(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dual_mask_type = 'channel' #default
    max_ratio = 0.01 #default

    batch_size = args.batch_size
    image_size = args.image_size
    crop_size = image_size
    dice_loss_function = DiceLoss()

    save_path = args.save_path 

    if not os.path.os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = f'{args.dataset}_{args.seed}seed_test_log_50e.txt'
    logger = get_logger(save_path, log_file)
    logger.info(args)
    
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    test_data = Dataset(root=args.data_path, transform=data_transform, target_transform=gt_transform, image_size=image_size, dataset_name = args.dataset, mode='test')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, num_workers=8,  shuffle=False)
    obj_list = test_data.obj_list

    print('test images:{}, total classes:{}'.format(len(test_data), len(obj_list)))
    logger.info('test images:{}, total classes:{}'.format(len(test_data), len(obj_list)))
    
    # build and load pre-trained model
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    model = build_d2rec(encoder_name = 'dinov2reg_vit_base_14', dual_mask= args.dual_mask, dual_mask_type=dual_mask_type, mask_head = args.mask_head, image_size = crop_size)
    
    model_path = save_path + '/dinov2_d2rec_50.pth' #TODO: change to your model path
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

  
    # training
    if crop_size > 256:
        resize_mask = 256
    else:
        resize_mask = None
  
    gt_masks, pr_masks, cls_names, gt_anomalys, pr_anomalys, img_paths = [], [], [], [], [], []
    model.eval()
    use_classwise_test = args.dataset == 'Real-IAD-Variety'
    if use_classwise_test:
        evaluator = Evaluator(device, metrics=args.eval_metrics)
    elif len(test_data) < 5000:
        evaluator = Evaluator(device, metrics=args.eval_metrics)
    else:
        evaluator = Evaluator('cpu', metrics=args.eval_metrics)
    
 
    
    # save results
    msg = {}
    if use_classwise_test:
        class_indices = {}
        for data_idx, data_info in enumerate(test_data.data_all):
            class_indices.setdefault(data_info['cls_name'], []).append(data_idx)

        for idx, cls_name in enumerate(tqdm(obj_list)):
            cls_dataset = Subset(test_data, class_indices[cls_name])
            cls_loader = torch.utils.data.DataLoader(cls_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
            results_eval = run_inference(cls_loader, cache_on_device=True)
            metric_results = evaluator.run(results_eval, logger=logger)
            update_msg(msg, metric_results, cls_name, idx)

            del results_eval, cls_loader, cls_dataset
            torch.cuda.empty_cache()
    else:
        results_eval = run_inference(test_loader, cache_on_device=len(test_data) < 5000)
        for idx, cls_name in enumerate(tqdm(obj_list)):
            metric_results = evaluator.run(results_eval, cls_name, logger)
            update_msg(msg, metric_results, cls_name, idx)

    tab = tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.1f', numalign="center", stralign="center", )
    
    logger.info('\n' + tab)
    
    #visualize(results_eval, save_path, obj_list, args.dataset)

    del evaluator
   
def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dual_mask_type = 'channel'
    batch_size = args.batch_size
    image_size = args.image_size
    crop_size = image_size
    dice_loss_function = DiceLoss()

    save_path = args.save_path + args.dataset + '-' + str(args.image_size) + '-' + str(args.batch_size) + '-dualmask-' + str(args.dual_mask) + '-maskhead-' + str(args.mask_head)
    
    if not os.path.os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = f'{args.dataset}_{args.seed}seed_train_log.txt'
    logger = get_logger(save_path, log_file)
    logger.info(args)
    
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)
    train_data = Dataset(root=args.data_path, transform=data_transform, target_transform=gt_transform, image_size=image_size, dataset_name = args.dataset, mode='train')
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, num_workers=4,  shuffle=True)
    obj_list = train_data.obj_list

    total_iters = args.epoch * (len(train_data)//args.batch_size + 1)
    print('train images:{}, total classes:{}'.format(len(train_data), len(obj_list)))
    logger.info('train images:{}, total classes:{}'.format(len(train_data), len(obj_list)))
    
    # load model
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    model = build_d2rec(encoder_name = 'dinov2reg_vit_base_14', dual_mask= args.dual_mask, dual_mask_type=dual_mask_type, mask_head = args.mask_head, image_size = crop_size)
    model = model.to(device)
    
    fixed_num_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    learned_num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(args)

    print('fixed params:{:.1f}M, learned_params:{:.1f}M, all params:{:.1f}M'.format(fixed_num_params/1e+6,learned_num_params/1e+6,(fixed_num_params+learned_num_params)/1e+6))
    logger.info('fixed params:{:.1f}M, learned_params:{:.1f}M, all params:{:.1f}M'.format(fixed_num_params/1e+6,learned_num_params/1e+6,(fixed_num_params+learned_num_params)/1e+6))
    

    learned_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = StableAdamW([{'params': learned_params}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)
    # training
    for epoch in range(args.epoch):
        model.train()
        loss_list = []
        
        for items in tqdm(train_loader):
            img = items['img'].to(device)
            label =  items['anomaly'].to(device)
        
            gt_masks = items['img_mask'].to(device)
            gt_masks[gt_masks > 0.5], gt_masks[gt_masks <= 0.5] = 1, 0
            
            if args.mask_head:
                ps_img = items['ps_img'].to(device)
                ps_masks = items['ps_mask'].to(device)
                ps_masks[ps_masks > 0.5], ps_masks[ps_masks <= 0.5] = 1, 0

                img = torch.cat((img, ps_img), dim=0)
                gt_masks = torch.cat((gt_masks, ps_masks), dim=0)

            ens, des, pred_masks = model(img)

            ens = [l2_normalize(en) for en in ens]
            des = [l2_normalize(de) for de in des]

            loss = cosine_loss_function(ens, des)
            if args.mask_head:
                pred_masks = F.interpolate(pred_masks, size=crop_size, mode='bilinear', align_corners=False)
                loss += 0.5 * dice_loss_function(pred_masks, ps_masks)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(learned_params, max_norm=0.1)

            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list)))
        logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.epoch, np.mean(loss_list)))

        if (epoch + 1) % 5 == 0:
            # save model
            torch.save(model.state_dict(), save_path + f'/dinov2_d2rec_{epoch+1}.pth')              


if __name__ == '__main__':
    parser = argparse.ArgumentParser("D2Rec", add_help=True)
    parser.add_argument("--dual_mask", action='store_true', help="dual mask or not")
    parser.add_argument("--mask_head", action='store_true', help='mask head or not')
    parser.add_argument("--data_path", type=str, default="./datasets/mvtec/", help="dataset path")
    parser.add_argument("--save_path", type=str, default='./checkpoints/', help='path to save results')
    parser.add_argument("--dataset", type=str, default='mvtec', help="dataset name")
    parser.add_argument("-e", "--evaluate", action="store_true")
    parser.add_argument("--eval_metrics", type=str, nargs="+", default=['I-AUROC', 'I-AP', 'P-AUROC', 'P-AP'], help='evaluation metrics')
    parser.add_argument("--epoch", type=int, default=50, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--save_freq", type=int, default=10, help="save frequency")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()

    setup_seed(args.seed)

    if args.evaluate:
        test(args)
    else:
        train(args)
