
from torch.utils.data import DataLoader

from datasets.semantickitti import SemanticKitti, semantickitti_collate

from omegaconf import OmegaConf
import MinkowskiEngine as ME

from utils.models.minkunet import MinkUNet34

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from tqdm import tqdm
import os
from datetime import datetime
from omegaconf import OmegaConf

from sklearn.metrics import confusion_matrix, jaccard_score
import numpy as np

from pathlib import Path


import sys, csv



class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes+1

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes-1) if n not in self.ignore], dtype=np.int64)
    
    #self.include = np.append(self.include, -1)
    
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes,
                                 self.n_classes),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify


    # check
    assert(x_row.shape == y_row.shape)

    # create indexes
    idxs = tuple(np.stack((x_row, y_row), axis=0))

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.conf_matrix, idxs, 1)


  def getStats(self):
    # remove fp from confusion on the ignore classes cols
    conf = self.conf_matrix.copy()
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = np.diag(conf)
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn
  def getIoU(self):
    tp, fp, fn = self.getStats()
    intersection = tp
    union = tp + fp + fn + 1e-15
    iou = intersection[self.include] / union[self.include]
    iou_mean = (intersection[self.include] / union[self.include]).mean()
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getacc(self):
    tp, fp, fn = self.getStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum() + 1e-15
    acc_mean = total_tp / total
    return acc_mean  # returns "acc mean"
    
  def get_confusion(self):
    return self.conf_matrix.copy()




class Tester :

    def __init__(self, config, loader, net, chkp_path, output_folder, save_predictions) -> None:

        self.config = config
        self.device = self.config.training.device
        self.net = net
        self.loader =loader
        self.output_folder = output_folder
        self.net = self.net.to(self.device)
        self.chkp_path = chkp_path
        self.save_predictions = save_predictions

        

        def load_student_model(checkpoint_path, model):
            # reloads model
            def clean_state_dict(state):
                # clean state dict from names of PL
                for k in list(ckpt.keys()):
                    if "target_model" in k:
                        ckpt[k.replace("target_model.", "")] = ckpt[k]
                    elif "student_model" in k:
                        ckpt[k.replace("student_model.", "")] = ckpt[k]
                    elif "source_model" in k:
                        ckpt[k.replace("source_model.", "")] = ckpt[k]
                    del ckpt[k]
                return state

            
            try :
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
            except KeyError:
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"]

            print(ckpt.keys())
            ckpt = clean_state_dict(ckpt)
            
            model.load_state_dict(ckpt, strict=True)
            return model

        def load_model(checkpoint_path, model):
            # reloads model
            def clean_state_dict(state):
                # clean state dict from names of PL
                for k in list(ckpt.keys()):
                    if "model" in k:
                        ckpt[k.replace("model.", "")] = ckpt[k]
                    del ckpt[k]
                return state

            try :
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
            except KeyError:
                ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"]

            
            #ckpt = clean_state_dict(ckpt)
            model.load_state_dict(ckpt, strict=True)
            
            return model

        self.checkpoint = load_model(self.chkp_path, self.net)

        self.init_folder()

        print("Checkpoint restored")



        self.class_strings = self.loader.dataset.class2names
        self.ignore = self.config.dataset.ignore_label
        self.n_classes = self.config.model.out_classes

        self.remap_lut = self.loader.dataset.remap_lut_val

        self.evaluator = iouEval(n_classes=self.n_classes, ignore=self.ignore)


    def init_folder(self) :
        print("Initializing folder structure...")
        sequences = self.loader.dataset.split[self.loader.dataset.phase]
        _, ckpt_name = os.path.split(self.chkp_path)
        for sequence in sequences :
            os.makedirs(os.path.join(self.output_folder, ckpt_name, 'sequences', sequence, 'predictions'), exist_ok=True)





    def test(self) :
        
        softmax = torch.nn.Softmax(dim=1)
        self.net.eval()

        bin_files = self.loader.dataset.label_path
        _, ckpt_name = os.path.split(self.chkp_path)

        remap_function = np.vectorize(lambda x: self.class_inv_remap[x])


        with torch.no_grad() :

            all_ious = []

            for data in tqdm(self.loader, position=0, leave=True) :
                coords, feats, labels, index = data['coordinates'], data['features'], data['labels'], data['idx']
                reverse_indices = data['reverse_indices']

                out = self.net(ME.SparseTensor(features=feats, coordinates=coords, device=self.device)).F

                preds = torch.argmax(softmax(out.squeeze()), dim=1).cpu().detach().numpy()

                self.evaluator.addBatch(preds, labels.cpu().numpy())

                m_accuracy = self.evaluator.getacc()
                m_jaccard, class_jaccard = self.evaluator.getIoU()

                print('Validation set:\n'
                        'Acc avg {m_accuracy:.3f}\n'
                        'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                                        m_jaccard=m_jaccard))


               
                batch_size = torch.unique(coords[:, 0]).max() + 1

                
                

                if self.save_predictions :
                   
                   for b in range(batch_size) :
                      
                      sample_idx = index[b]
                      b_idx = coords[:, 0] == b
                      points = coords[b_idx, 1:]
                      p = preds[b_idx]
                      l = labels[b_idx]          

                      #sequence = self.loader.dataset.files['input'][sample_idx].split('/')[0]


                      p = remap_function(p)
                      p = p[reverse_indices[b]].astype(np.uint32)

                      

                      pred_path = os.path.join(self.output_folder, ckpt_name, 'sequences', bin_files[sample_idx]).replace('labels', 'predictions')
                      p.tofile(pred_path)











 




