import torch.utils.data as data
import json
import random
from PIL import Image
import numpy as np
import torch
import os
import cv2
import tifffile as tiff
from torchvision import transforms
from utils.cutpaste import CutPaste, PerlinPaste

def get_data_transforms(size, isize):
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    
    data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])

    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
  
    return data_transforms, gt_transforms

def generate_class_info(dataset_name):
    class_name_map_class_id = {}
    if dataset_name == 'mvtec':
        obj_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == 'goods':
        obj_list = ['cigarette_box', 'drink_bottle', 'drink_can', 'food_bottle', 'food_box', 'food_package'] 
    elif dataset_name == 'mvtec-3d':
        obj_list = [
        'bagel', 'cable_gland', 'carrot', 'cookie', 'dowel', 
        'foam', 'peach', 'potato', 'rope', 'tire'
        ]
    elif dataset_name == 'visa':
        obj_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == 'mpdd':
        obj_list = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == 'btad':
        obj_list = ['01', '02', '03']
    elif dataset_name == 'DAGM_KaggleUpload':
        obj_list = ['Class1','Class2','Class3','Class4','Class5','Class6','Class7','Class8','Class9','Class10']
    elif dataset_name == 'SDD':
        obj_list = ['electrical commutators']
    elif dataset_name == 'DTD':
        obj_list = ['Woven_001', 'Woven_127', 'Woven_104', 'Stratified_154', 'Blotchy_099', 'Woven_068', 'Woven_125', 'Marbled_078', 'Perforated_037', 'Mesh_114', 'Fibrous_183', 'Matted_069']
    elif dataset_name =='medical':
        obj_list = ['brain',  'liver', 'retinal']
    elif dataset_name == 'colon':
        obj_list = ['colon']
    elif dataset_name == 'ISBI':
        obj_list = ['skin']
    elif dataset_name == 'Chest':
        obj_list = ['chest']
    elif dataset_name == 'thyroid':
        obj_list = ['thyroid']
    
    for k, index in zip(obj_list, range(len(obj_list))):
        class_name_map_class_id[k] = index

    return obj_list, class_name_map_class_id

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, image_size, dataset_name, mode='test'):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.data_all = []
        meta_info = json.load(open(f'{self.root}/meta.json', 'r'))
        name = self.root.split('/')[-1]
        meta_info = meta_info[mode]

        self.cls_names = list(meta_info.keys())
        for cls_name in self.cls_names:
            self.data_all.extend(meta_info[cls_name])
        self.length = len(self.data_all)

        self.cutpaste_transform = CutPaste()
        self.perlinpaste_transform = PerlinPaste()

        self.mode = mode
        if dataset_name == 'Real-IAD-Variety':
            self.obj_list = self.cls_names
            self.class_name_map_class_id = {}
            for k, index in zip(self.obj_list, range(len(self.obj_list))):
                self.class_name_map_class_id[k] = index
            
        else:
            self.obj_list, self.class_name_map_class_id = generate_class_info(dataset_name)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = self.data_all[index]
        img_path, mask_path, cls_name, specie_name, anomaly = data['img_path'], data['mask_path'], data['cls_name'], \
                                                              data['specie_name'], data['anomaly']
                                        
        _, ext = os.path.splitext(os.path.join(self.root, img_path))
        if ext == '.tiff':
            img = tiff.imread(os.path.join(self.root, img_path))
            depth_map = img[:,:,-1]
            img = np.repeat(depth_map[:, :, np.newaxis], 3, axis=2) #depth_map_3channel
            img = np.round((img - np.min(img)) * 255 / (np.max(img) - np.min(img)))
            img = Image.fromarray(np.uint8(img))
        else:
            img = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if anomaly == 0:
            img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            if os.path.isdir(os.path.join(self.root, mask_path)):
                # just for classification not report error
                img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
            else:
                img_mask = np.array(Image.open(os.path.join(self.root, mask_path)).convert('L')) > 0
                img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')
        
        if self.mode == 'train':
           # hybrid cutpaste and dream
            if self.image_size == 224:
                img = img.resize((256, 256))
                img_mask = img_mask.resize((256, 256))
            else:
                img = img.resize((512, 512))
                img_mask = img_mask.resize((512, 512))
                
            if cls_name in ['brain', 'liver', 'retinal']:
                ps_img, ps_mask = img.copy(), img_mask.copy()

                gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray_img, 35, 255, cv2.THRESH_BINARY) #+ cv2.THRESH_OTSU
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
                contour = max(contours, key=cv2.contourArea)
                fg_mask = np.zeros(gray_img.shape[:2], np.uint8)
                cv2.drawContours(fg_mask, [contour], -1, 255, -1)
                # 使用掩码计算最小外接矩形框
                coords = np.column_stack(np.where(fg_mask > 0))
                fg_y, fg_x, fg_h, fg_w = cv2.boundingRect(coords)

                if fg_w > 50 and fg_h > 50:
                    crop_img = img.crop((fg_x, fg_y, fg_x + fg_w, fg_y + fg_h))
                    crop_mask = img_mask.crop((fg_x, fg_y, fg_x + fg_w, fg_y + fg_h))
                
                    if torch.rand(1) > 0.5:
                        ps_crop_img, ps_crop_mask = self.cutpaste_transform(crop_img, crop_mask)
                        ps_img.paste(ps_crop_img, (fg_x, fg_y))
                        ps_mask.paste(ps_crop_mask, (fg_x, fg_y))
                    else:
                        ps_img, ps_mask = self.perlinpaste_transform(np.array(img), np.array(img_mask))
                        
                        ps_fg_mask = np.expand_dims(ps_mask/255.0 * fg_mask/255.0, axis=2)
                        ps_fg_img = np.array(img) * (1-ps_fg_mask) + ps_img * ps_fg_mask
                        ps_fg_mask = ps_fg_mask[:,:,0] * 255

                        ps_img, ps_mask = Image.fromarray(np.uint8(ps_fg_img)), Image.fromarray(np.uint8(ps_fg_mask))
            else:
                if torch.rand(1) > 0.5:
                    ps_img, ps_mask = self.cutpaste_transform(img, img_mask)
                else:
                    ps_img, ps_mask = self.perlinpaste_transform(np.array(img), np.array(img_mask))
                    ps_img, ps_mask = Image.fromarray(np.uint8(ps_img)), Image.fromarray(np.uint8(ps_mask))
        

        # transforms
        img = self.transform(img) if self.transform is not None else img
        img_mask = self.target_transform(   
            img_mask) if self.target_transform is not None and img_mask is not None else img_mask
        img_mask = [] if img_mask is None else img_mask

        if self.mode == 'train':
            ps_img = self.transform(ps_img) if self.transform is not None else ps_img
            ps_mask = self.target_transform(   
                ps_mask) if self.target_transform is not None and ps_mask is not None else ps_mask
            
            return {'img': img, 'img_mask': img_mask, 'ps_img': ps_img, 'ps_mask': ps_mask,
                    'cls_name': cls_name, 'anomaly': anomaly,
                    'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    
        else:
            return {'img': img, 'img_mask': img_mask, 'cls_name': cls_name, 'anomaly': anomaly,
                    'img_path': os.path.join(self.root, img_path), "cls_id": self.class_name_map_class_id[cls_name]}    
