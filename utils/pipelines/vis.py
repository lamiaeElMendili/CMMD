import os
import numpy as np
import torch
import torch.nn.functional as F

import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d

from utils.models.minkunet import ProjectionHEAD2, MinkUNet34, ProjectionHead
from utils.models.moco import MoCo
import random

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




        seed = 1
        set_seed(seed)



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
                     
        self.moco = MoCo(model_head=ProjectionHEAD2, config=config)


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
                     'masked_target_mask' : [],
                     'masked_source_pts': [],
                     'masked_source_labels': [],
                     'masked_source_features': [],
                     'masked_source_mask' : []}

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
            masked_target_mask = masked_target_mask[masked_target_voxel_idx]

            masked_source_pts = masked_source_pts[masked_source_voxel_idx]
            masked_source_labels = masked_source_labels[masked_source_voxel_idx]
            masked_source_features = masked_source_features[masked_source_voxel_idx]
            masked_source_mask = masked_source_mask[masked_source_voxel_idx]

            masked_target_pts = np.floor(masked_target_pts/self.training_dataset.voxel_size)
            masked_source_pts = np.floor(masked_source_pts/self.training_dataset.voxel_size)

            batch_index = np.ones([masked_target_pts.shape[0], 1]) * b
            masked_target_pts = np.concatenate([batch_index, masked_target_pts], axis=-1)

            batch_index = np.ones([masked_source_pts.shape[0], 1]) * b
            masked_source_pts = np.concatenate([batch_index, masked_source_pts], axis=-1)

            new_batch['masked_target_pts'].append(masked_target_pts)
            new_batch['masked_target_labels'].append(masked_target_labels)
            new_batch['masked_target_features'].append(masked_target_features)
            new_batch['masked_source_mask'].append(masked_source_mask)
            new_batch['masked_target_mask'].append(masked_target_mask)
            new_batch['masked_source_pts'].append(masked_source_pts)
            new_batch['masked_source_labels'].append(masked_source_labels)
            new_batch['masked_source_features'].append(masked_source_features)

        for k, i in new_batch.items():
            if k in ['masked_target_pts', 'masked_target_features', 'masked_source_pts', 'masked_source_features', 'masked_target_mask', 'masked_source_mask']:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0)).to(self.device)
            else:
                new_batch[k] = torch.from_numpy(np.concatenate(i, axis=0))

        return new_batch

    def training_step(self, batch, batch_idx):

        

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

        self.moco.model_k.eval()
        with torch.no_grad():

            target_pseudo = self.moco.model_k(target_stensor).F.cpu()

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

        
        self.classes = self.get_classes(source_labels, target_pseudo, n_classes=19)
            

        batch['pseudo_labels'] = target_pseudo
        batch['source_labels'] = source_labels
        masked_batch = self.mask_data(batch, is_oracle=False)


        s2t_stensor = ME.SparseTensor(coordinates=masked_batch["masked_target_pts"].int(),
                                      features=masked_batch["masked_target_features"])


        s2t_labels = masked_batch["masked_target_labels"]     

        points = torch.cat([target_stensor.C[:, 1:], source_stensor.C[:, 1:]], dim=0) 
        print(points.shape)
        from mpl_toolkits.mplot3d import Axes3D

        for c in self.classes :
            pcd = o3d.geometry.PointCloud()
            pcd_s = o3d.geometry.PointCloud()
            #pcd.points = o3d.utility.Vector3dVector(target_stensor.C.numpy()[target_pseudo == c][:,1:])
            #pcd_s.points = o3d.utility.Vector3dVector(source_stensor.C.numpy()[source_labels == c][:,1:])
            pcd.points = o3d.utility.Vector3dVector(target_stensor.C.numpy()[:,1:])
            #labels = target_stensor.F.squeeze()
            labels = target_pseudo
            import matplotlib.pyplot as plt
            colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
            colors[labels < 0] = 0

            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
            #pcd.points = o3d.utility.Vector3dVector(s2t_stensor.C.numpy()[:,1:])
            #import matplotlib.pyplot as plt
            #fig = plt.figure()
            #ax = fig.add_subplot(111, projection='3d')

            # Plot the points
            #ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')
            #ax.scatter(s2t_stensor.C.numpy()[:,1], s2t_stensor.C.numpy()[:,2], s2t_stensor.C.numpy()[:,3], c='r', marker='+')

            # Set labels and title
            #ax.set_xlabel('X Label')
            #ax.set_ylabel('Y Label')
            #ax.set_zlabel('Z Label')
            #ax.set_title('3D Point Cloud')

            # Show the plot
            #plt.show()
            #pcd.paint_uniform_color([1, 0, 0])
            #pcd_s.paint_uniform_color([0, 0, 1])
            #o3d.visualization.draw_geometries([pcd, pcd_s])
            o3d.visualization.draw_geometries([pcd])



        return 



       
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
        print('unique classes are ', new_classes)
        return new_classes 
    
    def cmmd(self, q_seg, q_labels, queue, queue_labels) :

        queue = torch.transpose(queue, 0, 1)

        self.all_pos, self.all_neg = [], []
        
        losses = [] 
        layer = 0      

 
        cl = len(self.classes)-1
    

        loss_tensors = [torch.zeros(len(self.classes), cl+1).cuda()]




        pos_loss = 0
        temperature = 1

        
        for (k, c) in enumerate(self.classes) :
            
            query = q_seg[q_labels == c]
            key = queue[queue_labels == c]



            


    

            
            #pos_loss = self.mmd_loss(s_c, t_c, step=self.current_epoch)
                
            

            #print(f'Number of positive source samples in class {c} is ', s_c.shape[0], f'Number of target samples in class c is ', t_c.shape[0])

            
            pos_loss = self.mmd_linear(query, key, neg=False)


            #print("pos_loss class ", c, pos_loss.item())
            #    if j != c :
            #        pos_loss += mmd_linear(s_c[:, j].unsqueeze(1), t_c[:, j].unsqueeze(1))

            self.log(
                name="pos_loss",
                value=pos_loss.item(),
                logger=True
            )

            #self.log(
            #    name=f"{c}_ssamples-tsamples",
            #    value=s_c.shape[0] - t_c.shape[0],
            #    logger=True
            #)


            self.all_pos.append(pos_loss.item())
            loss_tensors[layer][k, 0] = -pos_loss/ temperature
            neg_loss = 0
            l = 1

            #if neg_classes  == None:

            neg_classes = self.classes[self.classes!=c]


            for j in neg_classes :
                if j != c :
                    

                    key_j = queue[queue_labels == j]




                    #t_j = t_j[self.sample_dist(t_j, n_elements2)].squeeze()

                    #b = self.mmd_loss(s_c, t_j, step=self.current_epoch, neg=True)

                    b = self.mmd_linear(query, key_j, neg=True)
                    #print('neg between ', c, ' and ', j, ' is ', b.item())
                    self.log(
                        name="neg_loss",
                        value=b.item(),
                        logger=True
                    ) 
                    loss_tensors[layer][k, l] = -b / temperature
                    l = l+1
                    self.all_neg.append(b.item())

        

        #losses.append(-torch.log10(torch.div(loss_tensors[layer], denominator)[:, 0]).mean())
        losses.append(F.cross_entropy(loss_tensors[layer], torch.zeros(size=(len(self.classes),1)).squeeze(1).cuda().long()))

        #wandb.log({'Target train/Pos loss' : np.array(self.all_pos).mean()})  
        #wandb.log({'Target train/Neg loss' : np.array(self.all_neg).mean()})  

        #losses.append(-torch.log_softmax(loss_tensors[layer], dim=1)[:, 0].mean())
        layer = layer + 1
        #print(loss_tensors)
        return losses[0]



    def mmd_linear(self, X, Y,k=2, sigma=1, neg=False):

        X = X.contiguous()
        Y = Y.contiguous()



        
        #X = X / torch.norm(X, p=2, dim=1, keepdim=True)
        #Y = Y / torch.norm(Y, p=2, dim=1, keepdim=True)

        n = (X.shape[0] // 2) * 2
        m = (Y.shape[0] // 2) * 2

        k = 10

        with torch.no_grad() :

            #sigma = torch.median(torch.cdist(X, X))
            l = 1000
            total = torch.cat([sample_elements(X, l), sample_elements(Y, l)], dim=0)
            sigma = max(torch.sum(torch.cdist(total, total, p=2).data) / (total.size()[0]**2-total.size()[0]), 0.01)

            #xx = X.detach().cpu().numpy()
            #yy = Y.detach().cpu().numpy()
            #sigma=10
            #sigma = torch.median(torch.cdist(torch.cat([X, Y], dim=0), torch.cat([X, Y], dim=0), p=2))
            #sigma = torch.median(torch.cdist(X, Y)) 
        

        sigmas = [sigma*i for i in [1]]
        mmd2 = 0

        if neg :
            self.log(
                name="neg_sigma",
                value=sigma,
                logger=True
            )
        else :
            self.log(
                name="pos_sigma",
                value=sigma,
                logger=True
            )


        


        for sigma in sigmas :



            rbf = lambda A, B: torch.exp(-torch.cdist(A.contiguous(), B.contiguous(), p=2) / (2*sigma**2)).mean()
            #rbf = lambda A, B :  torch.mm(A, B.T).mean()
            
            #print(torch.mm(X, Y.T))


            #mmd2 = -rbf(X, Y)
        
            mmd2 += rbf(X[:n:k], X[1:n:k]) + rbf(Y[:m:k], Y[1:m:k])- rbf(X[:n:k], Y[1:m:k]) - rbf(X[1:n:k], Y[:m:k])
            
            del sigma
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
        if False :
        #if self.trainer.global_step > self.last_step and self.trainer.global_step % self.update_every == 0:
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
        preds = conf.max(dim=-1).indices.detach().cpu().numpy()
        conf = conf.max(dim=-1).values


        classes = np.unique(preds)
        print(classes)

        for c in classes :
            pcd = o3d.geometry.PointCloud()
            pcd_s = o3d.geometry.PointCloud()
            print(stensor.C.numpy()[preds == c].shape)
            pcd.points = o3d.utility.Vector3dVector(stensor.C.numpy()[labels == c][:,1:])
            pcd_s.points = o3d.utility.Vector3dVector(stensor.C.numpy()[labels == c][:,1:])
            #pcd.points = o3d.utility.Vector3dVector(target_stensor.C.numpy()[:,1:])
            import matplotlib.pyplot as plt

            pcd.paint_uniform_color([1, 0, 0])
            pcd_s.paint_uniform_color([0, 0, 1])
            o3d.visualization.draw_geometries([pcd, pcd_s])        
        return

        
    

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