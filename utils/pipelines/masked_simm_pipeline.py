import os
import numpy as np
import torch
import torch.nn.functional as F

import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d
import random
from pytorch_metric_learning import losses

from pytorch_metric_learning.distances import CosineSimilarity
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt

class SimMaskedAdaptation(pl.core.LightningModule):
    def __init__(self,
                 config,
                 student_model,
                 teacher_model,
                 momentum_updater,
                 training_dataset,
                 source_validation_dataset,
                 target_validation_dataset,
                 optimizer_name="SGD",
                 source_criterion='SoftDICELoss',
                 target_criterion='SoftDiceLoss',
                 other_criterion=None,
                 source_weight=0.5,
                 target_weight=0.5,
                 filtering=None,
                 lr=1e-3,
                 train_batch_size=12,
                 val_batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None,
                 update_every=1,
                 weighted_sampling=False,
                 target_confidence_th=0.95,
                 selection_perc=0.5,
                 save_mix=False):
        


        super().__init__()
        for name, value in list(vars().items()):
            if name != "self":
                setattr(self, name, value)


        self.ignore_label = self.training_dataset.ignore_label

        # ########### LOSSES ##############
        if source_criterion == 'CELoss':
            self.source_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif source_criterion == 'SoftDICELoss':
            self.source_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        # ########### LOSSES ##############
        if target_criterion == 'CELoss':
            self.target_criterion = CELoss(ignore_label=self.training_dataset.ignore_label, weight=None)
        elif target_criterion == 'SoftDICELoss':
            self.target_criterion = SoftDICELoss(ignore_label=self.training_dataset.ignore_label)
        else:
            raise NotImplementedError

        # in case, we will use an extra criterion
        self.other_criterion = other_criterion

        # ############ WEIGHTS ###############
        self.source_weight = source_weight
        self.target_weight = target_weight

        # ############ LABELS ###############
        self.ignore_label = self.training_dataset.ignore_label
        # self.target_pseudo_buffer = pseudo_buffer

        # init
        self.save_hyperparameters(config.to_dict())

        # others
        self.validation_phases = ['source_validation', 'target_validation']
        # self.validation_phases = ['pseudo_target']

        self.class2mixed_names = self.training_dataset.class2names
        self.class2mixed_names = np.append(self.class2mixed_names, ["target_label"], axis=0)

        self.voxel_size = self.training_dataset.voxel_size

        # self.knn_search = KNN(k=self.propagation_size, transpose_mode=True)

        if self.training_dataset.weights is not None and self.weighted_sampling:
            tot = self.source_validation_dataset.weights.sum()
            self.sampling_weights = 1 - self.source_validation_dataset.weights/tot

        else:
            self.sampling_weights = None

    @property
    def momentum_pairs(self):
        """Defines base momentum pairs that will be updated using exponential moving average.
        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """
        return [(self.student_model, self.teacher_model)]

    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0
        self.meaniou = 0
        self.outputs = []
        self.loss_func = losses.ContrastiveLoss(distance = CosineSimilarity(),pos_margin=1, neg_margin=0)

    def random_sample(self, points, sub_num):
        """
        :param points: input points of shape [N, 3]
        :return: np.ndarray of N' points sampled from input points
        """

        num_points = points.shape[0]

        if sub_num is not None:
            if sub_num <= num_points:
                sampled_idx = np.random.choice(np.arange(num_points), sub_num, replace=False)
            else:
                over_idx = np.random.choice(np.arange(num_points), sub_num - num_points, replace=False)
                sampled_idx = np.concatenate([np.arange(num_points), over_idx])
        else:
            sampled_idx = np.arange(num_points)

        return sampled_idx

    @staticmethod
    def switch_off(labels, switch_classes):
        for s in switch_classes:
            class_idx = labels == s
            labels[class_idx] = -1

        return labels

    def sample_classes(self, origin_classes, num_classes, is_pseudo=False):

        if not is_pseudo:
            if self.weighted_sampling and self.sampling_weights is not None:

                sampling_weights = self.sampling_weights[origin_classes] * (1/self.sampling_weights[origin_classes].sum())

                selected_classes = np.random.choice(origin_classes, num_classes,
                                                    replace=False, p=sampling_weights)

            else:
                selected_classes = np.random.choice(origin_classes, num_classes, replace=False)

        else:
            selected_classes = origin_classes

        return selected_classes

    def mask(self, origin_pts, origin_labels, origin_features,
             dest_pts, dest_labels, dest_features, is_pseudo=False):

        # to avoid when filtered labels are all -1
        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = np.unique(origin_labels)
            origin_present_classes = origin_present_classes[origin_present_classes != -1]

            num_classes = int(self.selection_perc * origin_present_classes.shape[0])

            selected_classes = self.sample_classes(origin_present_classes, num_classes, is_pseudo)

            print(selected_classes)

            selected_idx = []
            selected_pts = []
            selected_labels = []
            selected_features = []

            if not self.training_dataset.augment_mask_data:
                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]

                    selected_idx.append(class_idx)
                    selected_pts.append(origin_pts[class_idx])
                    selected_labels.append(origin_labels[class_idx])
                    selected_features.append(origin_features[class_idx])

                if len(selected_pts) > 0:
                    # selected_idx = np.concatenate(selected_idx, axis=0)
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            else:

                for sc in selected_classes:
                    class_idx = np.where(origin_labels == sc)[0]

                    class_pts = origin_pts[class_idx]
                    num_pts = class_pts.shape[0]
                    sub_num = int(0.5 * num_pts)

                    # random subsample
                    random_idx = self.random_sample(class_pts, sub_num=sub_num)
                    #print("selected indices for class ", sc, '  ', random_idx.shape[0])
                    class_idx = class_idx[random_idx]
                    class_pts = class_pts[random_idx]

                    # get transformation
                    voxel_mtx, affine_mtx = self.training_dataset.mask_voxelizer.get_transformation_matrix()

                    rigid_transformation = affine_mtx @ voxel_mtx
                    # apply transformations
                    homo_coords = np.hstack((class_pts, np.ones((class_pts.shape[0], 1), dtype=class_pts.dtype)))
                    class_pts = homo_coords @ rigid_transformation.T[:, :3]
                    class_labels = np.ones_like(origin_labels[class_idx]) * sc
                    class_features = origin_features[class_idx]

                    selected_idx.append(class_idx)
                    selected_pts.append(class_pts)
                    selected_labels.append(class_labels)
                    selected_features.append(class_features)

                if len(selected_pts) > 0:
                    # selected_idx = np.concatenate(selected_idx, axis=0)
                    selected_pts = np.concatenate(selected_pts, axis=0)
                    selected_labels = np.concatenate(selected_labels, axis=0)
                    selected_features = np.concatenate(selected_features, axis=0)

            if len(selected_pts) > 0:
                dest_idx = dest_pts.shape[0]
                dest_pts = np.concatenate([dest_pts, selected_pts], axis=0)
                dest_labels = np.concatenate([dest_labels, selected_labels], axis=0)
                dest_features = np.concatenate([dest_features, selected_features], axis=0)

                mask = np.ones(dest_pts.shape[0])
                mask[:dest_idx] = 0
            else :
                mask = np.ones(dest_pts.shape[0])

            if self.training_dataset.augment_data:
                # get transformation
                voxel_mtx, affine_mtx = self.training_dataset.voxelizer.get_transformation_matrix()
                rigid_transformation = affine_mtx @ voxel_mtx
                # apply transformations
                homo_coords = np.hstack((dest_pts, np.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype)))
                dest_pts = homo_coords @ rigid_transformation.T[:, :3]

        return dest_pts, dest_labels, dest_features, mask.astype(bool)

    def mask_data(self, batch, is_oracle=False):
        # source
        batch_source_pts = batch['source_coordinates'].cpu().numpy()
        batch_source_labels = batch['source_labels'].cpu().numpy()
        batch_source_features = batch['source_features'].cpu().numpy()

        # target
        batch_target_idx = batch['target_coordinates'][:, 0].cpu().numpy()
        batch_target_pts = batch['target_coordinates'].cpu().numpy()

        batch_target_features = batch['target_features'].cpu().numpy()

        batch_size = int(np.max(batch_target_idx).item() + 1)

        if is_oracle:
            batch_target_labels = batch['target_labels'].cpu().numpy()

        else:
            batch_target_labels = batch['pseudo_labels'].cpu().numpy()

        new_batch = {'masked_target_pts': [],
                     'masked_target_labels': [],
                     'masked_target_features': [],
                     'masked_source_pts': [],
                     'masked_source_labels': [],
                     'masked_source_features': []}

        target_order = np.arange(batch_size)

        for b in range(batch_size):
            source_b_idx = batch_source_pts[:, 0] == b
            target_b = target_order[b]
            target_b_idx = batch_target_idx == target_b

            # source
            source_pts = batch_source_pts[source_b_idx, 1:] * self.voxel_size
            source_labels = batch_source_labels[source_b_idx]
            source_features = batch_source_features[source_b_idx]

            # target
            target_pts = batch_target_pts[target_b_idx, 1:] * self.voxel_size
            target_labels = batch_target_labels[target_b_idx]
            target_features = batch_target_features[target_b_idx]

            masked_target_pts, masked_target_labels, masked_target_features, masked_target_mask = self.mask(origin_pts=source_pts,
                                                                                                            origin_labels=source_labels,
                                                                                                            origin_features=source_features,
                                                                                                            dest_pts=target_pts,
                                                                                                            dest_labels=target_labels,
                                                                                                            dest_features=target_features)

            masked_source_pts, masked_source_labels, masked_source_features, masked_source_mask = self.mask(origin_pts=target_pts,
                                                                                                            origin_labels=target_labels,
                                                                                                            origin_features=target_features,
                                                                                                            dest_pts=source_pts,
                                                                                                            dest_labels=source_labels,
                                                                                                            dest_features=source_features,
                                                                                                            is_pseudo=True)



            _, _, _, masked_target_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_target_pts,
                                                                          features=masked_target_features,
                                                                          labels=masked_target_labels,
                                                                          quantization_size=self.training_dataset.voxel_size,
                                                                          return_index=True)

            _, _, _, masked_source_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_source_pts,
                                                                      features=masked_source_features,
                                                                      labels=masked_source_labels,
                                                                      quantization_size=self.training_dataset.voxel_size,
                                                                      return_index=True)

            masked_target_pts = masked_target_pts[masked_target_voxel_idx]
            masked_target_labels = masked_target_labels[masked_target_voxel_idx]
            masked_target_features = masked_target_features[masked_target_voxel_idx]

            masked_source_pts = masked_source_pts[masked_source_voxel_idx]
            masked_source_labels = masked_source_labels[masked_source_voxel_idx]
            masked_source_features = masked_source_features[masked_source_voxel_idx]

            masked_target_pts = np.floor(masked_target_pts/self.training_dataset.voxel_size)
            masked_source_pts = np.floor(masked_source_pts/self.training_dataset.voxel_size)

            batch_index = np.ones([masked_target_pts.shape[0], 1]) * b
            masked_target_pts = np.concatenate([batch_index, masked_target_pts], axis=-1)

            batch_index = np.ones([masked_source_pts.shape[0], 1]) * b
            masked_source_pts = np.concatenate([batch_index, masked_source_pts], axis=-1)

            new_batch['masked_target_pts'].append(masked_target_pts)
            new_batch['masked_target_labels'].append(masked_target_labels)
            new_batch['masked_target_features'].append(masked_target_features)
            new_batch['masked_source_pts'].append(masked_source_pts)
            new_batch['masked_source_labels'].append(masked_source_labels)
            new_batch['masked_source_features'].append(masked_source_features)

        for k, i in new_batch.items():
            if k in ['masked_target_pts', 'masked_target_features', 'masked_source_pts', 'masked_source_features']:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(self.device)
            else:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))

        return new_batch

    def training_step(self, batch, batch_idx):
        '''
        :param batch: training batch
        :param batch_idx: batch idx
        :return: None
        '''

        '''
        batch.keys():
            - source_coordinates
            - source_labels
            - source_features
            - source_idx
            - target_coordinates
            - target_labels
            - target_features
            - target_idx
        '''
        # Must clear cache at regular interval



        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        # target batch
        target_stensor = ME.SparseTensor(coordinates=batch['target_coordinates'].int(),
                                         features=batch['target_features'])

        target_labels = batch['target_labels'].long().cpu()

        source_stensor = ME.SparseTensor(coordinates=batch['source_coordinates'].int(),
                                         features=batch['source_features'])

        source_labels = batch['source_labels'].long().cpu()

        #print(target_stensor, target_labels)

        self.teacher_model.eval()
        
        with torch.no_grad():

            target_pseudo = self.teacher_model(target_stensor).F.cpu()

            if self.filtering == 'confidence':
                target_confidence_th = self.target_confidence_th
                target_pseudo = F.softmax(target_pseudo, dim=-1)
                target_conf, target_pseudo = target_pseudo.max(dim=-1)
                filtered_target_pseudo = -torch.ones_like(target_pseudo)
                valid_idx = target_conf > target_confidence_th
                filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
                target_pseudo = filtered_target_pseudo.long()

            else:
                target_pseudo = F.softmax(target_pseudo, dim=-1)
                target_conf, target_pseudo = target_pseudo.max(dim=-1)
        

        out, features = self.student_model(target_stensor, get_all=True)

        label_mapping = {
            0: [2,4,14],
            1: [14, 1, 10],
            2: [14, 7, 0],
            3: [14, 0, 2],
            4: [14, 5, 0],
            5:[6, 2, 14],
            6: [7, 2, 8],
            7: [6, 2, 5],
            8: [10, 7, 16],
            9: [10, 8, 16],
            10: [8, 16, 10],
            11: [10, 8, 16],
            12: [14, 13, 10],
            13: [14, 12, 10],
            14: [16, 13, 10],
            15: [14, 5, 17] ,
            16: [10, 14, 15],
            17: [14, 15, 5],
            18: [17, 15, 14]
            }
        
        confusing_classes = torch.tensor([label_mapping[label.item()] for label in target_labels[target_labels!=-1]]) # (N, 3)



        final_weights = self.student_model.final.kernel



        distance_matrix = torch.abs(3*final_weights[:, target_labels[target_labels!=-1]] - final_weights[:, confusing_classes[:, 0]] - final_weights[:, confusing_classes[:, 1]]
                                        - final_weights[:, confusing_classes[:, 2]]).detach().cpu().numpy()
        
        
        # distance_matrix of shape (96, N)
        
        distance_matrix, all_index = torch.topk(torch.tensor(distance_matrix), 10, dim=0, largest=False) # reyurns the indices of the top 5 smallest values

        mask = target_labels != -1

        features = features.F[mask.cuda()].gather(dim=1, index=all_index.t().cuda())

        #sns.heatmap(distance_matrix[:, :400], annot=True, cmap='viridis', fmt=".2f",
        #                xticklabels=[f"Class {i}" for i in range(6)],
        #            yticklabels=[f"Class {i}" for i in range(5)])
        #plt.title("Euclidean Distance between Class Weights")
        #plt.xlabel("Class")
        #plt.ylabel("Class")
        #plt.show()   




        
        contrastive_loss = self.loss_func(features[::10], target_labels[mask][::10]) 

        print('contrastive loss:', contrastive_loss)



        batch['pseudo_labels'] = target_pseudo
        batch['source_labels'] = source_labels
        masked_batch = self.mask_data(batch, is_oracle=False)

        s2t_stensor = ME.SparseTensor(coordinates=masked_batch["masked_target_pts"].int(),
                                      features=masked_batch["masked_target_features"])

        t2s_stensor = ME.SparseTensor(coordinates=masked_batch["masked_source_pts"].int(),
                                      features=masked_batch["masked_source_features"])

        s2t_labels = masked_batch["masked_target_labels"]
        t2s_labels = masked_batch["masked_source_labels"]

        s2t_out = self.student_model(s2t_stensor).F.cpu()
        t2s_out = self.student_model(t2s_stensor).F.cpu()

        s2t_loss = self.target_criterion(s2t_out, s2t_labels.long())
        t2s_loss = self.target_criterion(t2s_out, t2s_labels.long())

        print('contrastive loss:', contrastive_loss)





        final_loss = s2t_loss + t2s_loss + 0.1*contrastive_loss

        results_dict = {'s2t_loss': s2t_loss.detach(),
                    't2s_loss': t2s_loss.detach(), 
                    'contrastive_loss': contrastive_loss.detach()}

        with torch.no_grad():
            self.student_model.eval()
            target_out = self.student_model(target_stensor).F.cpu()
            _, target_preds = target_out.max(dim=-1)

            target_iou_tmp = jaccard_score(target_preds.numpy(), target_labels.numpy(), average=None,
                                            labels=np.arange(0, self.num_classes),
                                            zero_division=0.)
            present_labels, class_occurs = np.unique(target_labels.numpy(), return_counts=True)
            present_labels = present_labels[present_labels != self.ignore_label]
            present_names = self.training_dataset.class2names[present_labels].tolist()
            present_names = ['student/' + p + '_target_iou' for p in present_names]
            results_dict.update(dict(zip(present_names, target_iou_tmp.tolist())))
            results_dict['student/target_iou'] = np.mean(target_iou_tmp[present_labels])

        self.student_model.train()

        valid_idx = torch.logical_and(target_pseudo != -1, target_labels != -1)
        correct = (target_pseudo[valid_idx] == target_labels[valid_idx]).sum()
        pseudo_acc = correct / valid_idx.sum()

        results_dict['teacher/acc'] = pseudo_acc
        results_dict['teacher/confidence'] = target_conf.mean()

        ann_pts = (target_pseudo != -1).sum()
        results_dict['teacher/annotated_points'] = ann_pts/target_pseudo.shape[0]

        for k, v in results_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v).float()
            else:
                v = v.float()
            self.log(
                name=k,
                value=v,
                logger=True,
                sync_dist=True,
                rank_zero_only=True,
                batch_size=self.train_batch_size,
                add_dataloader_idx=False
            )

        return final_loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.
        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
            dataloader_idx (int): index of the dataloader.
        """
        if self.trainer.global_step > self.last_step and self.trainer.global_step % self.update_every == 0:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # update tau
            cur_step = self.trainer.global_step
            if self.trainer.accumulate_grad_batches:
                cur_step = cur_step * self.trainer.accumulate_grad_batches
            self.momentum_updater.update_tau(
                cur_step=cur_step,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step

    def validation_step(self, batch, batch_idx):

        phase = 'target_validation'
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.student_model(stensor).F
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
    
        
    

    def validation_epoch_end(self, outputs):

        mean_iou = []
        phase = 'target_validation'


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
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.student_model.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.student_model.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            if self.scheduler_name == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
            elif self.scheduler_name == 'ExponentialLR':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            elif self.scheduler_name == 'CyclicLR':
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr/10000, max_lr=self.lr,
                                                              step_size_up=5, mode="triangular2")
            elif self.scheduler_name == 'OneCycleLR':
                steps_per_epoch = int(len(self.training_dataset) / self.train_batch_size)
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr,
                                                                steps_per_epoch=steps_per_epoch,
                                                                epochs=self.trainer.max_epochs)

            else:
                raise NotImplementedError

            return [optimizer], {'scheduler': scheduler}

