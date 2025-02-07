import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from utils.losses import CELoss, DICELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score, confusion_matrix
import open3d as o3d
from utils.losses.metrics import stats_iou_per_class, stats_accuracy_per_class, stats_pfa_per_class, ignore_cm_adaption, stats_overall_accuracy
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt


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
        # remove -1 labels
        mask = y_row != -1
        x_row = x_row[mask]
        y_row = y_row[mask]
        # create indexes
        idxs = tuple(np.stack((x_row, y_row), axis=0))
        # make confusion matrix (cols = gt, rows = pred)
        np.add.at(self.conf_matrix, idxs, 1)


    def getStats(self):
        # remove fp from confusion on the ignore classes cols
        conf = self.conf_matrix.copy()
        conf[:, self.ignore] = 0
        conf[self.ignore, :] = 0

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




class PLTTrainer(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 training_dataset,
                 validation_dataset,
                 optimizer_name='SGD',
                 criterion='CELoss',
                 lr=1e-3,
                 batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 val_batch_size=6,
                 train_num_workers=10,
                 val_num_workers=10,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None):

        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)




        if criterion == 'CELoss':
            self.criterion = CELoss(ignore_label=self.training_dataset.ignore_label,
                                    weight=None)

        elif criterion == 'DICELoss':
            self.criterion = DICELoss(ignore_label=self.training_dataset.ignore_label)

        elif criterion == 'SoftDICELoss':
            if self.num_classes == 19:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label, is_kitti=True)
            else:
                self.criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)

        else:
            raise NotImplementedError


        self.ignore_label = self.training_dataset.ignore_label
        self.validation_phases = ['source_validation', 'target_validation']

        self.meaniou = 0
        self.outputs = []    


    def training_step(self, batch, batch_idx):
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        loss = self.criterion(out, labels)

        _, preds = out.max(1)
        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join('training', p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict['training/loss'] = loss
        results_dict['training/iou'] = np.mean(iou_tmp[present_labels])
        results_dict['training/lr'] = self.trainer.optimizers[0].param_groups[0]["lr"]
        results_dict['training/epoch'] = self.current_epoch

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.batch_size
            )
        return loss
        
    def validation_step(self, batch, batch_idx):

        phase = 'target_validation'
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.model(stensor).F
        labels = batch['labels'].long()

        

        # loss = self.criterion(out, labels)
        conf = F.softmax(out, dim=-1)
        preds = conf.max(dim=-1).indices
        conf = conf.max(dim=-1).values


        iou_tmp = jaccard_score(preds.detach().cpu().numpy(), labels.cpu().numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0)
        
        


        present_labels, class_occurs = np.unique(labels.cpu().numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]

        
        self.meaniou += np.mean(iou_tmp[present_labels])        
        #print(self.meaniou / (batch_idx+1))
        
        
        iou_tmp = torch.from_numpy(iou_tmp)

        iou = -torch.ones_like(iou_tmp)
        iou[present_labels] = iou_tmp[present_labels]

        self.outputs.append({'iou': iou})
        mean_iou = []
        # mean_loss = []

        for return_dict in self.outputs:
            iou_tmp = return_dict['iou']
            # loss_tmp = return_dict['loss']

            nan_idx = iou_tmp == -1
            iou_tmp[nan_idx] = float('nan')
            mean_iou.append(iou_tmp.unsqueeze(0))
            # mean_loss.append(loss_tmp)

        mean_iou = torch.cat(mean_iou, dim=0).numpy()

        per_class_iou = np.nanmean(mean_iou, axis=0) * 100
        # loss = np.mean(mean_loss)

        results = {'iou': np.nanmean(per_class_iou)}    
        print(per_class_iou)
        print(results)   

        
        
        return {'iou': iou}
    
        

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""

        print('on train start')
        self.last_step = 0
        self.meaniou = 0
        self.outputs = []    

    def validation_epoch_end(self, outputs):

        mean_iou = []
        phase = 'target_validation'


        for return_dict in outputs:
            iou_tmp = return_dict['iou']
            # loss_tmp = return_dict['loss']

            nan_idx = iou_tmp == -1
            iou_tmp[nan_idx] = float('nan')
            mean_iou.append(iou_tmp.unsqueeze(0))
            # mean_loss.append(loss_tmp)

        mean_iou = torch.cat(mean_iou, dim=0).numpy()

        per_class_iou = np.nanmean(mean_iou, axis=0) * 100
        # loss = np.mean(mean_loss)

        results_dict = {f'{phase}/iou': np.mean(per_class_iou)}

        for c in range(per_class_iou.shape[0]):
            class_name = self.training_dataset.class2names[c]
            results_dict[os.path.join(phase, class_name + '_iou')] = per_class_iou[c]

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.val_batch_size,
                add_dataloader_idx=False
            )


        self.meaniou = 0
        self.outputs = []

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError
            

            print(self.scheduler_name)

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")

            else:
                raise NotImplementedError

            return [optimizer], {"scheduler" : scheduler}


class PLTTester(pl.core.LightningModule):
    r"""
    Segmentation Module for MinkowskiEngine for training on one domain.
    """

    def __init__(self,
                 model,
                 criterion='CELoss',
                 dataset=None,
                 clear_cache_int=2,
                 num_classes=19,
                 checkpoint_path=None,
                 save_predictions=False,
                 save_folder=None):

        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)

        self.ignore_label = self.dataset.ignore_label

        self.output_folder = os.path.join(os.path.dirname(os.path.dirname(self.checkpoint_path)))

        os.makedirs(self.output_folder, exist_ok=True)

        # if criterion == 'CELoss':
        #     self.criterion = CELoss(ignore_label=self.ignore_label,
        #                             weight=None)
        # else:
        #     raise NotImplementedError
        self.remap_function = np.vectorize(lambda x: self.dataset.class_inv_remap[x])
        self.init_folder()

        self.cm = np.zeros((self.num_classes+1, self.num_classes+1), dtype=np.int64)
        self.evaluator = iouEval(n_classes=self.num_classes, ignore=self.ignore_label)


    def init_folder(self) :
        print("Initializing folder structure...")
        sequences = self.dataset.split[self.dataset.phase]
        _, self.ckpt_name = os.path.split(self.checkpoint_path)
        for sequence in sequences :
            os.makedirs(os.path.join(self.output_folder, self.ckpt_name, 'sequences', sequence, 'predictions'), exist_ok=True)
            #os.makedirs(os.path.join(self.output_folder, self.ckpt_name, 'sequences', sequence, 'labels'), exist_ok=True)




    def test_step(self, batch, batch_idx, dataloader_idx=0):
        bin_files = self.dataset.label_path
        phase = 'test'
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        if self.save_predictions :

            reverse_indices = batch['reverse_indices']

        out = self.model(stensor).F
        labels = batch['labels'].long()


        

        # loss = self.criterion(out, labels)
        conf = F.softmax(out, dim=-1)
        preds = conf.max(dim=-1).indices
        conf = conf.max(dim=-1).values




        self.evaluator.addBatch(preds.cpu().numpy(), labels.cpu().numpy())




        m_accuracy = self.evaluator.getacc()
        m_jaccard, class_jaccard = self.evaluator.getIoU()

        print('Validation set:\n'
                'Acc avg {m_accuracy:.3f}\n'
                'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                                m_jaccard=m_jaccard))      




        if self.save_predictions and (batch_idx % 10 == 0):
            coords = batch["coordinates"].cpu()
            labels = batch["labels"].cpu()
            preds = preds.cpu()
            conf = conf.cpu()

            batch_size = torch.unique(coords[:, 0]).max() + 1
            sample_idx = batch["idx"]
            for b in range(batch_size.int()):
                s_idx = int(sample_idx[b].item())
                b_idx = coords[:, 0] == b
                p = preds[b_idx].cpu().detach().numpy()
                #l = labels[b_idx].cpu().detach().numpy()
                
                p = self.remap_function(p)
                p = p[reverse_indices[b].cpu().detach().numpy()].astype(np.uint32)
                #l = l[reverse_indices[b].cpu().detach().numpy()].astype(np.uint32)
                
                pred_path = os.path.join(self.output_folder, self.ckpt_name, 'sequences', os.path.basename(os.path.dirname(os.path.dirname(bin_files[dataloader_idx]))), 'predictions', os.path.basename(bin_files[s_idx])).replace('labels', 'predictions')
                #label_path = os.path.join(self.output_folder, self.ckpt_name, 'sequences', os.path.basename(os.path.dirname(os.path.dirname(bin_files[dataloader_idx]))), 'labels', os.path.basename(bin_files[s_idx]))
                p.tofile(pred_path)
                #l.tofile(label_path)


    def on_test_epoch_end(self):

        class_names = self.dataset.class2names

        """ plt.figure(figsize=(6, 6))  # Optional, for better size
        sns.heatmap((self.evaluator.conf_matrix / self.evaluator.conf_matrix.sum(axis=0, keepdims=True) * 100), annot=True, fmt='.1f', cmap='Blues', cbar=True)
        plt.title("Confusion Matrix")
        plt.xlabel("True Class")
        plt.ylabel("predicted Class")
        plt.xticks(ticks=np.arange(self.num_classes) + 0.5, labels=class_names, rotation=45, ha='right')  # For horizontal axis
        plt.yticks(ticks=np.arange(self.num_classes) + 0.5, labels=class_names, rotation=0)  # For vertical axis

        plt.show() """

        
        m_accuracy = self.evaluator.getacc()
        m_jaccard, class_jaccard = self.evaluator.getIoU()
        class_jaccard = np.round(np.array(class_jaccard)*100, 2)
        results = {}
        print(class_jaccard)
        print(self.dataset.class2names)
   
        for c in range(class_jaccard.shape[0]):
            class_name = self.dataset.class2names[c]
            results[class_name] = class_jaccard[c]

        results['mIoU'] = round(m_jaccard*100, 2)

        os.makedirs(os.path.join(self.trainer.weights_save_path, 'results'), exist_ok=True)
        csv_columns = list(results.keys())

        _, ckpt_name = os.path.split(self.checkpoint_path)
        ckpt_name = ckpt_name[:-5]

        csv_file = os.path.join(self.trainer.weights_save_path, 'results', ckpt_name+'_test.csv')
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            writer.writerow(results)

