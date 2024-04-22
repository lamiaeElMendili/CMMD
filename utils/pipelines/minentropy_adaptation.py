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

from utils.losses.metrics import stats_iou_per_class, stats_accuracy_per_class, stats_pfa_per_class, ignore_cm_adaption, stats_overall_accuracy


class iouEval:
  def __init__(self, n_classes, ignore=None):
    # classes
    self.n_classes = n_classes

    # What to include and ignore from the means
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array(
        [n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    
    #self.include = np.append(self.include, -1)
    
    print("[IOU EVAL] IGNORE: ", self.ignore)
    print("[IOU EVAL] INCLUDE: ", self.include)

    # reset the class counters
    self.reset()

  def num_classes(self):
    return self.n_classes

  def reset(self):
    self.conf_matrix = np.zeros((self.n_classes+1,
                                 self.n_classes+1),
                                dtype=np.int64)

  def addBatch(self, x, y):  # x=preds, y=targets
    # sizes should be matching
    x_row = x.reshape(-1)  # de-batchify
    y_row = y.reshape(-1)  # de-batchify

    present_labels, _ = np.unique(y_row, return_counts=True)
    present_labels = present_labels[present_labels != self.ignore]

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
    iou = intersection / union
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



class Adaptation(pl.core.LightningModule):
    def __init__(self,
                 model,
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
                 scheduler_name=None):

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



        # ############ LABELS ###############
        self.ignore_label = self.training_dataset.ignore_label
        # self.target_pseudo_buffer = pseudo_buffer

        # init

        # others
        self.validation_phases = ['target_validation']
        # self.validation_phases = ['pseudo_target']

        self.last_step = 0
        self.meaniou = 0
        self.outputs = []

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
             dest_pts, dest_labels, dest_features, is_pseudo=False):

        # to avoid when filtered labels are all -1
        if (origin_labels == -1).sum() < origin_labels.shape[0]:
            origin_present_classes = np.unique(origin_labels)
            origin_present_classes = origin_present_classes[origin_present_classes != -1]

            num_classes = int(self.selection_perc * origin_present_classes.shape[0])

            selected_classes = self.sample_classes(origin_present_classes, num_classes, is_pseudo)

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
        #print(target_stensor, target_labels)


        #self.model.model_k.eval()

        
        

        #with torch.no_grad():

        #    target_pseudo = self.model.model_k(target_stensor).F

        #    target_confidence_th = self.target_confidence_th
        #    target_pseudo = F.softmax(target_pseudo, dim=-1)
        #    target_conf, target_pseudo = target_pseudo.max(dim=-1)
        #    filtered_target_pseudo = -torch.ones_like(target_pseudo)
        #    valid_idx = target_conf > target_confidence_th
        #    filtered_target_pseudo[valid_idx] = target_pseudo[valid_idx]
        #    target_pseudo = filtered_target_pseudo.long()


        #batch['pseudo_labels'] = target_pseudo
        #batch['source_labels'] = source_labels

        #masked_batch = self.mask_data(batch, is_oracle=False)

        #s2t_stensor = ME.SparseTensor(coordinates=masked_batch["masked_target_pts"].int(),
        #                              features=masked_batch["masked_target_features"])

        #t2s_stensor = ME.SparseTensor(coordinates=masked_batch["masked_source_pts"].int(),
        #                              features=masked_batch["masked_source_features"])

        #s2t_labels = masked_batch["masked_target_labels"]
        #t2s_labels = masked_batch["masked_source_labels"]






        #q_seg, q_labels, queue, queue_labels = self.model(source_stensor, source_labels.cuda(), s2t_stensor, s2t_labels.cuda(), step=self.trainer.global_step)
                        
        s_out = self.model(source_stensor).F.cpu()
        t_out = self.model(target_stensor).F.cpu()
        
        s_loss = self.target_criterion(s_out, source_labels.long())
        t_loss = self.entropy_loss(t_out)


            


        final_loss = s_loss + 0.0001*t_loss
        results_dict = {'final_loss': final_loss.detach(),
                    's_loss': s_loss.detach(),
                    'entropy_loss': t_loss.detach()
                        }

        with torch.no_grad():
            self.model.eval()
            target_out = self.model(target_stensor).F.cpu()
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

        self.model.train()

       

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


    def entropy_loss(self, v):
        """
            Entropy loss for probabilistic prediction vectors
            input: batch_size x channels x h x w
            output: batch_size x 1 x h x w
        """

        n, c = v.size()
        v = torch.softmax(v, dim=1)        
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * np.log2(c))    


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

        new_loss = loss_tensors.clone()
        
        new_loss[:, 0] =diag
        new_loss.diag()[:] = first_col
                    
        #return torch.log_softmax(-torch.div(loss_tensors.diagonal(), torch.sum(loss_tensors, dim=1)), dim=).mean()
        

        return F.cross_entropy(new_loss, torch.zeros(size=(len(q_labels), 1)).squeeze(1).cuda().long())





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
        
        #return torch.round(mmd2, decimals=5)
        return mmd2





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
