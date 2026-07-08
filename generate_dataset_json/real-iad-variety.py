'''
 # @ Author: Bin-Bin Gao
 # @ Create Time: 2025-07-25 10:19:17
 # @ Modified by: Your name
 # @ Modified time: 2025-07-25 22:11:44
 # @ Description: Detect and Describe Any Vision Anomalies with Large Vision-Language Models
 '''



import os
import glob
import json

class IndustryDBSolver(object):
    def __init__(self, root='./data/Real-IAD-Variety"'):
        self.data_path = root
    
        self.meta_path = f'{self.data_path}/meta.json'

        self.json_files = [json_file for json_file in os.listdir(f'{self.data_path}') if ".json" in json_file]

        self.cls_names = sorted([item.split('.json')[0] for item in self.json_files])
        
    def run(self):
          
        info = dict(
            categories = self.cls_names,
            defects = [],
            modalities = {'photography': self.cls_names},
            number_normal = {item: 0 for item in self.cls_names},
            number_abnormal = {item: 0 for item in self.cls_names},
            train={}, 
            test={}
            )
       

        for json_file in self.json_files:
            cls_name = json_file.split('.json')[0]
            
            file_path = os.path.join(self.json_path, json_file)

         
            with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f) 
                
            for phase in ['train', 'test']:
                info[phase][cls_name] = []
                cls_info = []

                for item in data[phase]:

                    view_id =  item["img_path"].split('/')[-1].split('_')[1]  # Assuming the view is indicated in the filename

                    if item["specie_name"] not in info['defects'] and item["specie_name"] not in ['OK']:
                        info['defects'].append(item["specie_name"])

                    info_img = dict(
                        img_path = item["img_path"],
                        mask_path= item["mask_path"] if item['mask_path'] else "",
                        annotation_type='mask',
                        view_id = view_id,
                        cls_name=cls_name,
                        specie_name=item["specie_name"],
                        modality='photography',
                        anomaly=item['anomaly'],    
                    )
                    is_abnormal = True if item['anomaly'] != 0 else False
                    
                    if is_abnormal:
                        cls_info.append(info_img)
                        info['number_abnormal'][cls_name] += 1
                    else:
                        cls_info.append(info_img)
                        info['number_normal'][cls_name] += 1
                info[phase][cls_name] = cls_info

        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

        print('meta_path', self.meta_path)
        
if __name__ == '__main__':
    root = "/fuxi_team2/persons/danylgao/weicun_ceph/datasets/Real-IAD-Variety"
    json_path = '/fuxi_team2/persons/danylgao/weicun_ceph/datasets/Real-IAD-Variety-Json-v0829'
    runner = IndustryDBSolver(root='./data/Real-IAD-Variety')
    runner.run()