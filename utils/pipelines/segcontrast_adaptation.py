
import os
import numpy as np
import torch
import torch.nn.functional as F

import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d
from torchmetrics.functional.pairwise import pairwise_linear_similarity
from utils.models.minkunet import ProjectionHEAD2, MinkUNet34, ProjectionHead
from utils.models.moco_original import MoCo
import random
from time import time
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sample_elements(tensor, n):
    if tensor.size(0) <= n :
        return tensor
    else :
        indices = torch.randperm(tensor.size(0))[:n]
        return tensor[indices]

def visualize_pcd_clusters(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])





class Adaptation(pl.core.LightningModule):
    def __init__(self,
                 config, 
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
        self.save_hyperparameters(self.config.to_dict())

        # others
        self.validation_phases = ['target_validation']
        # self.validation_phases = ['pseudo_target']

        self.class2mixed_names = self.training_dataset.class2names
        self.class2mixed_names = np.append(self.class2mixed_names, ["target_label"], axis=0)

        self.voxel_size = self.training_dataset.voxel_size

        if self.training_dataset.weights is not None and self.weighted_sampling:
            tot = self.source_validation_dataset.weights.sum()
            self.sampling_weights = 1 - self.source_validation_dataset.weights/tot

        else:
            self.sampling_weights = None
                     
        self.moco = MoCo(model_head=ProjectionHead, config=config)
        

    @property
    def momentum_pairs(self):
        """Defines base momentum pairs that will be updated using exponential moving average.
        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """
        return [(self.moco.model_q, self.moco.model_k)]       


    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0
        self.meaniou = 0
        self.outputs = []
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
             dest_pts, dest_labels, dest_features, is_pseudo=False, ismix3d=False):


        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = torch.unique(origin_labels)
            origin_present_classes = origin_present_classes[origin_present_classes != -1]

            if not ismix3d:
                num_classes = int(self.selection_perc * origin_present_classes.shape[0])
                selected_classes = self.sample_classes(origin_present_classes, num_classes, is_pseudo)
            else:
                selected_classes = origin_present_classes

            selected_pts = []
            selected_labels = []
            selected_features = []

            for sc in selected_classes:
                class_idx = (origin_labels == sc).nonzero(as_tuple=True)[0]

                selected_pts.append(origin_pts[class_idx])
                selected_labels.append(origin_labels[class_idx])
                selected_features.append(origin_features[class_idx])

            if len(selected_pts) > 0:
                selected_pts = torch.cat(selected_pts, dim=0)
                selected_labels = torch.cat(selected_labels, dim=0)
                selected_features = torch.cat(selected_features, dim=0)

 
            if len(selected_pts) > 0:
                dest_pts = torch.cat([dest_pts, selected_pts], dim=0)
                dest_labels = torch.cat([dest_labels, selected_labels], dim=0)
                dest_features = torch.cat([dest_features, selected_features], dim=0)

                mask = torch.ones(dest_pts.size(0), dtype=torch.bool, device=self.device)
                mask[:dest_pts.size(0) - selected_pts.size(0)] = False

            if self.training_dataset.augment_data:
                # Get transformation
                voxel_mtx, affine_mtx = self.training_dataset.voxelizer.get_transformation_matrix()
                rigid_transformation = affine_mtx @ voxel_mtx

                # Apply transformations
                homo_coords = torch.cat((dest_pts, torch.ones((dest_pts.shape[0], 1), dtype=dest_pts.dtype, device=self.device)), dim=1)
                dest_pts = homo_coords @ torch.tensor(rigid_transformation[:, :3], dtype=dest_pts.dtype,device=self.device)

        return dest_pts, dest_labels, dest_features, mask


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
                     'masked_source_pts' : [],
                     'masked_source_labels' : [],
                     'masked_source_features' : []}

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

            # mask destination points are 0

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
                                                                                                            dest_features=source_features)

            _, _, _, masked_target_voxel_idx = ME.utils.sparse_quantize(coordinates=masked_target_pts,
                                                                          features=masked_target_features,
                                                                          labels=masked_target_labels,
                                                                          quantization_size=self.training_dataset.voxel_size,
                                                                          return_index=True)
            

            _, _, _, masked_source_voxel_idx = ME.utils.sparse_quantize(
                coordinates=masked_source_pts,
                features=masked_source_features,
                labels=masked_source_labels,
                quantization_size=self.training_dataset.voxel_size,
                return_index=True
            )


            masked_target_pts = masked_target_pts[masked_target_voxel_idx]
            masked_target_labels = masked_target_labels[masked_target_voxel_idx]
            masked_target_features = masked_target_features[masked_target_voxel_idx]
            masked_target_mask = masked_target_mask[masked_target_voxel_idx]


            masked_target_pts = np.floor(masked_target_pts/self.training_dataset.voxel_size)




            masked_source_pts = masked_source_pts[masked_source_voxel_idx]
            masked_source_labels = masked_source_labels[masked_source_voxel_idx]
            masked_source_features = masked_source_features[masked_source_voxel_idx]
            masked_source_mask = masked_source_mask[masked_source_voxel_idx]

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

    def mix3d(self, batch) :

       # source
    # Extract data from the batch
        batch_source_pts = batch['source_coordinates']
        batch_source_labels = batch['source_labels']
        batch_source_features = batch['source_features']

        batch_target_idx = batch['target_coordinates'][:, 0]
        batch_target_pts = batch['target_coordinates']
        batch_target_features = batch['target_features']

        batch_target_labels = batch['pseudo_labels']

        batch_size =  int(torch.max(batch_target_idx).item() + 1)

        new_batch = {'mixed_pts': [],
                    'mixed_labels': [],
                    'mixed_features': []}

        target_order = torch.arange(batch_size, device=self.device)

        for b in range(batch_size):
            source_b_idx = batch_source_pts[:, 0] == b
            target_b = target_order[b]
            target_b_idx = batch_target_idx == target_b

            # Extract source and target data
            source_pts = batch_source_pts[source_b_idx, 1:] * self.voxel_size
            source_labels = batch_source_labels[source_b_idx]
            source_features = batch_source_features[source_b_idx]

            target_pts = batch_target_pts[target_b_idx, 1:] * self.voxel_size
            target_labels = batch_target_labels[target_b_idx]
            target_features = batch_target_features[target_b_idx]

            # Perform mixing
            mixed_pts, mixed_labels, mixed_features, _ = self.mask(origin_pts=source_pts,
                                                                    origin_labels=source_labels,
                                                                    origin_features=source_features,
                                                                    dest_pts=target_pts,
                                                                    dest_labels=target_labels,
                                                                    dest_features=target_features,
                                                                    ismix3d=True)

            # Quantize and concatenate the mixed data
            _, _, mixed_voxel_idx = ME.utils.sparse_quantize(coordinates=mixed_pts,
                                                                        features=mixed_features,
                                                                        quantization_size=self.training_dataset.voxel_size,
                                                                        return_index=True)
            

            
            mixed_pts = mixed_pts[mixed_voxel_idx]
            mixed_labels = mixed_labels[mixed_voxel_idx]
            mixed_features = mixed_features[mixed_voxel_idx]

            mixed_pts = torch.floor(mixed_pts/self.training_dataset.voxel_size)
            batch_index = torch.ones([mixed_pts.shape[0], 1], device=self.device) * b
            mixed_pts = torch.cat([batch_index, mixed_pts], dim=-1)

            new_batch['mixed_pts'].append(mixed_pts)
            new_batch['mixed_labels'].append(mixed_labels)
            new_batch['mixed_features'].append(mixed_features)

        # Convert to tensors
        for k, i in new_batch.items():
            new_batch[k] = torch.cat(i, dim=0).to(self.device) if k in ['mixed_pts', 'mixed_features'] else torch.cat(i, dim=0)

        return new_batch



    def training_step(self, batch, batch_idx):
        
        # Must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()


        # target batch
        target_stensor = ME.SparseTensor(coordinates=batch['target_coordinates'].int().cuda(),
                                         features=batch['target_features'].cuda())

        target_labels = batch['target_labels'].long()

        source_stensor = ME.SparseTensor(coordinates=batch['source_coordinates'].int().cuda(),
                                         features=batch['source_features'].cuda())

        source_labels = batch['source_labels'].long()

        self.moco.model_k.eval()

        with torch.no_grad():

            target_pseudo = self.moco.model_k(target_stensor).F

            target_confidence_th = self.target_confidence_th
            target_pseudo = F.softmax(target_pseudo, dim=-1)
            target_conf, target_pseudo = target_pseudo.max(dim=-1)
            filtered_target_pseudo = -torch.ones_like(target_pseudo)
            valid_idx = target_conf > target_confidence_th
            filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
            target_pseudo = filtered_target_pseudo.long()




        batch['pseudo_labels'] = target_pseudo
        batch['source_labels'] = source_labels




        mixed_batch = self.mix3d(batch)

        mixed_tensor = ME.SparseTensor(coordinates=mixed_batch["mixed_pts"].int(),
                                        features=mixed_batch["mixed_features"])
        
        mixed_labels = mixed_batch["mixed_labels"]




        #out_seg, tgt_seg = self.moco(source_stensor, source_labels, target_stensor, target_pseudo, step=self.trainer.global_step)
        q_seg, q_labels, queue, queue_labels = self.moco(source_stensor, source_labels.cuda(), mixed_tensor, mixed_labels.cuda(), step=self.trainer.global_step)
        
        s_out = self.moco.model_q(source_stensor).F.cpu()
        t_out = self.moco.model_q(mixed_tensor).F.cpu()

 
        loss = self.target_criterion(s_out,source_labels.long())
        target_loss = self.target_criterion(t_out, mixed_labels.long())


        loss_mmd = self.cmmd(q_seg, q_labels, queue, queue_labels)
        #loss_mmd = self.target_criterion(out_seg, tgt_seg.long())

        #final one
        final_loss = loss + target_loss + self.config.adaptation.cmmd.lamda_cmmd * loss_mmd
        results_dict = {'cmmd': loss_mmd.detach(),
                            'final_loss': final_loss.detach(),
                            'target_loss': target_loss.detach(),
                    'source_loss': loss.detach()
                    }

        with torch.no_grad():
            self.moco.model_q.eval()
            target_out = self.moco.model_q(target_stensor).F.cpu()
            _, target_preds = target_out.max(dim=-1)

            target_iou_tmp = jaccard_score(target_preds.cpu().detach().numpy(), target_labels.cpu().detach().numpy(), average=None,
                                            labels=np.arange(0, self.num_classes),
                                            zero_division=0.)
            present_labels, class_occurs = np.unique(target_labels.cpu().detach().numpy(), return_counts=True)
            present_labels = present_labels[present_labels != self.ignore_label]
            present_names = self.training_dataset.class2names[present_labels].tolist()
            present_names = ['student/' + p + '_target_iou' for p in present_names]
            results_dict.update(dict(zip(present_names, target_iou_tmp.tolist())))
            results_dict['student/target_iou'] = np.mean(target_iou_tmp[present_labels])

        self.moco.model_q.train()

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



       
    def get_classes(self, source_labels, target_pseudo, min_points=10, n_classes=5) :

        s_classes = torch.unique(source_labels).unsqueeze(1).cpu().detach().numpy()
        t_classes = torch.unique(target_pseudo).unsqueeze(1).cpu().detach().numpy()

        classes =  list(np.intersect1d(s_classes, t_classes))

        new_classes = []
        for c in classes:
            if (source_labels == c).sum() > min_points and (target_pseudo == c).sum() > min_points:
                new_classes.append(int(c))

        new_classes = np.array(new_classes)
        if len(new_classes) > n_classes :
            new_classes = np.random.choice(new_classes, n_classes, replace=False)

        new_classes = new_classes[new_classes!=self.ignore_label]
        new_classes = np.sort(new_classes)
        print('selected classes for cmmd', new_classes)
        return new_classes 
    
    def cmmd(self, q_seg, q_labels, queue, queue_labels):
        queue = torch.transpose(queue, 0, 1)
        self.all_pos, self.all_neg = [], []
        losses = []
        layer = 0
        loss_tensors = [torch.zeros(len(q_labels), len(q_labels)).cuda()]
        pos_loss = 0
        temperature = self.config.adaptation.cmmd.temperature

        for (k, c) in enumerate(q_labels):
            query = q_seg[q_labels == c]
            key = queue[queue_labels == c]
            pos_loss = self.mmd_linear(query, key, neg=False)
            self.log(name="pos_loss", value=pos_loss.item(), logger=True)
            self.all_pos.append(pos_loss.item())
            loss_tensors[layer][k, 0] = -pos_loss / temperature
            neg_loss = 0
            l = 1
            neg_classes = q_labels[q_labels != c]

            for j in neg_classes:
                if j != c:
                    key_j = queue[queue_labels == j]
                    b = self.mmd_linear(query, key_j, neg=True)
                    self.log(name="neg_loss", value=b.item(), logger=True)
                    loss_tensors[layer][k, l] = -b / temperature
                    l = l + 1
                    self.all_neg.append(b.item())

        losses.append(F.cross_entropy(loss_tensors[layer], torch.zeros(size=(len(q_labels), 1)).squeeze(1).cuda().long()))
        layer = layer + 1
        return losses[0]

    def cmmd2(self, q_seg, q_labels, queue, queue_labels):

        queue = torch.transpose(queue, 0, 1)
        self.all_pos, self.all_neg = [], []
        loss_tensors = torch.zeros(len(q_labels), len(q_labels)).cuda()
        temperature = self.config.adaptation.cmmd.temperature

        unique_labels = torch.unique(q_labels)

        for (k, c) in enumerate(unique_labels):
            query = q_seg[q_labels == c]
            key = queue[queue_labels == c]
            pos_loss = self.mmd_linear(query, key, neg=False)
            self.log(name="pos_loss", value=pos_loss.item(), logger=True)
            self.all_pos.append(pos_loss.item())
            loss_tensors[k, k] = -pos_loss / temperature
            neg_classes = q_labels[q_labels != c]

            for j in neg_classes:
                if j != c:
                    index_j = torch.where(unique_labels == j)
                    if loss_tensors[index_j, k] != 0:
                        loss_tensors[k, index_j] = loss_tensors[index_j, k]
                        continue

                    key_j = queue[queue_labels == j]
                    b = self.mmd_linear(query, key_j, neg=True)
                    self.log(name="neg_loss", value=b.item(), logger=True)
                    loss_tensors[k, index_j] = -b / temperature
                    self.all_neg.append(b.item())

        
        diag = loss_tensors.diagonal()
        first_col = loss_tensors[:, 0]
        
        loss_tensors[:, 0] =diag
        loss_tensors.diag()[:] = first_col
                    
        #return torch.log_softmax(-torch.div(loss_tensors.diagonal(), torch.sum(loss_tensors, dim=1)), dim=).mean()
        

        return F.cross_entropy(loss_tensors, torch.zeros(size=(len(q_labels), 1)).squeeze(1).cuda().long())





    def mmd_linear(self, X, Y, k=2, sigma=1, neg=False):
        X = X.contiguous()
        Y = Y.contiguous()

        n = (X.shape[0] // 2) * 2
        m = (Y.shape[0] // 2) * 2

        k = self.config.adaptation.cmmd.k

        if self.config.adaptation.cmmd.kernel == 'gaussian':
            with torch.no_grad():
                l = 1000
                total = torch.cat([sample_elements(X, l), sample_elements(Y, l)], dim=0)
                sigma = max(torch.sum(torch.cdist(total, total, p=2).data) / (total.size()[0]**2-total.size()[0]), 0.01)

        if neg:
            self.log(name="neg_sigma", value=sigma, logger=True)
        else:
            self.log(name="pos_sigma", value=sigma, logger=True)

        if self.config.adaptation.cmmd.kernel == 'gaussian':
            rbf = lambda A, B: torch.exp(-torch.cdist(A.contiguous(), B.contiguous(), p=2) / (2*sigma**2))
        else:
            rbf = pairwise_linear_similarity

        if m <= 5000:
            mmd2 = rbf(X, X).mean() + rbf(Y, Y).mean() - 2*rbf(X, Y).mean() 
        else:
            mmd2 = rbf(X, X).mean() + rbf(Y[:m:k], Y[1:m:k]).mean() - rbf(X, Y[1:m:k]).mean() - rbf(X, Y[:m:k]).mean()
        
        return mmd2


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

        out = self.moco.model_q(stensor).F
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
                optimizer = torch.optim.SGD(self.moco.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.moco.parameters(),
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(self.moco.parameters(),
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(self.moco.parameters(),
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

            return [optimizer], {"scheduler": scheduler}
