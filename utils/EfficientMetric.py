import torch
from torch.nn import functional as F
from metrics import AUPR, AUPRO, AUROC, F1Max

class Evaluator(object):
    def __init__(self, device, metrics=[]):
        if len(metrics) == 0:
            self.metrics = [
                'I-AUROC', 'I-AP', 'I-F1max', 
                'P-AUROC', 'P-AP', 'P-F1max', 'P-AUPRO'
            ]
        else:
            self.metrics = metrics

        self.aupr = AUPR().to(device)
        self.aupro = AUPRO().to(device)
        self.auroc = AUROC().to(device)
        self.f1max = F1Max().to(device)

    def run(self, results, cls_name=None, logger=None):
        # 重置所有指标的内部状态，避免跨类别调用时状态累积
        self.auroc.reset()
        self.aupr.reset()
        self.aupro.reset()
        self.f1max.reset()

        if cls_name is None:
            gt_px = results['gt_masks']
            pr_px = results['pr_masks']

            gt_sp = results['gt_anomalys']
            pr_sp = results['pr_anomalys']
        else:
            idxes = results['cls_names'] == cls_name

            gt_px = results['gt_masks'][idxes] 
            pr_px = results['pr_masks'][idxes]

            gt_sp = results['gt_anomalys'][idxes]
            pr_sp = results['pr_anomalys'][idxes]
      
        if len(gt_px.shape) == 4:
            gt_px = gt_px.squeeze(1)
        if len(pr_px.shape) == 4:
            pr_px = pr_px.squeeze(1)
        
        # min-max normalization
        pr_px = (pr_px - pr_px.min()) / (pr_px.max() - pr_px.min() + 1e-12)
        pr_sp = (pr_sp - pr_sp.min()) / (pr_sp.max() - pr_sp.min() + 1e-12)
     
        eval_results = {}
        for metric in self.metrics:
            if metric.startswith('I-AUROC'):
                iauroc = self.auroc(pr_sp, gt_sp).item()
                if iauroc < 0.5:
                    iauroc =  1 - 0.5
                eval_results[metric] = iauroc
                
            elif metric.startswith('P-AUROC'):
                pauroc = self.auroc(pr_px.ravel(), gt_px.ravel()).item()
                if pauroc < 0.5:
                    pauroc =  1 - 0.5
                eval_results[metric] = pauroc
                    
            elif metric.startswith('I-AP'):
                eval_results[metric] = self.aupr(pr_sp, gt_sp).item()
                
            elif metric.startswith('P-AP'):
                eval_results[metric] = self.aupr(pr_px.ravel(), gt_px.ravel()).item()
                  
            elif metric.startswith('I-F1max'):
                eval_results[metric] = self.f1max(pr_sp, gt_sp).item()
            
            elif metric.startswith('P-F1max'):
                eval_results[metric] = self.f1max(pr_px.ravel(), gt_px.ravel()).item()
            
            elif metric.startswith('P-AUPRO'):
                eval_results[metric] = self.aupro(pr_px, gt_px).item()
         
        return eval_results