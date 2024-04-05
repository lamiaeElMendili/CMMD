import os
import numpy as np
import torch
import torch.nn.functional as F


import MinkowskiEngine as ME
from utils.losses import CELoss, SoftDICELoss
import pytorch_lightning as pl
from sklearn.metrics import jaccard_score
import open3d as o3d
from MinkowskiEngine.modules.resnet_block import BasicBlock



def sample_elements(tensor, n):
    if tensor.size(0) <= n :
        return tensor
    else :
        indices = torch.randperm(tensor.size(0))[:n]
        return tensor[indices]

def mmd_linear(X, Y,k=2, sigma=1):

    X = X.contiguous()
    Y = Y.contiguous()



    
    #X = X / torch.norm(X, p=2, dim=1, keepdim=True)
    #Y = Y / torch.norm(Y, p=2, dim=1, keepdim=True)

    n = (X.shape[0] // 2) * 2
    m = (Y.shape[0] // 2) * 2

    k = 2

    with torch.no_grad() :

        #sigma = torch.median(torch.cdist(X, X))
        l = 5000
        total = torch.cat([sample_elements(X, l), sample_elements(Y, l)], dim=0)
        sigma = torch.sum(torch.cdist(total, total, p=2).data) / (total.size()[0]**2-total.size()[0])

        #xx = X.detach().cpu().numpy()
        #yy = Y.detach().cpu().numpy()
        #sigma=10
        #sigma = torch.median(torch.cdist(torch.cat([X, Y], dim=0), torch.cat([X, Y], dim=0), p=2))
        #sigma = torch.median(torch.cdist(X, Y)) 
    

    sigmas = [sigma*i for i in [1]]
    mmd2 = 0

    for sigma in sigmas :

        rbf = lambda A, B: torch.exp(-torch.cdist(A.contiguous(), B.contiguous(), p=2) / (2*sigma**2)).mean()
        #rbf = lambda A, B :  torch.mm(A, B.T).mean()
        
        #print(torch.mm(X, Y.T))


        #mmd2 = -rbf(X, Y)
    
        mmd2 += rbf(X[:n:k], X[1:n:k]) + rbf(Y[:m:k], Y[1:m:k])- rbf(X[:n:k], Y[1:m:k]) - rbf(X[1:n:k], Y[:m:k])
        
        del sigma
    return mmd2



class Adaptation(pl.core.LightningModule):
    def __init__(self,
                 source_model,
                 training_dataset,
                 source_validation_dataset,
                 target_validation_dataset,
                 optimizer_name="SGD",
                 source_criterion='SoftDICELoss',
                 target_criterion='SoftDiceLoss',
                 other_criterion=None,
                 lr=1e-3,
                 train_batch_size=12,
                 val_batch_size=12,
                 weight_decay=1e-4,
                 momentum=0.98,
                 num_classes=19,
                 clear_cache_int=2,
                 scheduler_name=None,
                 weighted_sampling=False):

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
        
    
        self.devcie = 'cuda'

        # in case, we will use an extra criterion
        self.other_criterion = other_criterion

        # ############ LABELS ###############
        self.ignore_label = self.training_dataset.ignore_label
        # self.target_pseudo_buffer = pseudo_buffer

        # init


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
        self.source_model = self.source_model.to(self.device)

        # from 1 - 8
        self.layer_idx = 8
        self.lamda = 0.1


    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0



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
        
        self.get_layers(self.source_model)



        target_out = self.source_model(target_stensor).F       
        source_out = self.source_model(source_stensor).F

        source_features = self.activation[f"block{self.layer_idx}.1"].F
        target_features = self.activation[f"block{self.layer_idx}.1"].F
        

        loss = self.source_criterion(source_out.to(self.device), source_labels)

        mmd_loss = mmd_linear(source_features, target_features)
        
        print(mmd_loss)


        final_loss = loss + self.lamda*mmd_loss


        
         


        results_dict = {
                    'mmd_loss' : mmd_loss.detach(),
                    'final_loss': final_loss.detach()}

        with torch.no_grad():
            self.source_model.eval()
            target_out = self.source_model(target_stensor).F.cpu()
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

        self.source_model.train()

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


    def getActivation(self, name):
        #the hook signature
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def get_layers(self, model) :
       
        self.activation = {}

        for idx, (name, layer) in enumerate(model.named_modules()) :
            
            if isinstance(layer, BasicBlock) :
                layer.register_forward_hook(self.getActivation(name))




    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        phase = self.validation_phases[dataloader_idx]
        # input batch
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])

        # must clear cache at regular interval
        if self.global_step % self.clear_cache_int == 0:
            torch.cuda.empty_cache()

        out = self.source_model(stensor).F.cpu()

        labels = batch['labels'].long().cpu()
        if phase == 'source_validation':
            loss = self.source_criterion(out, labels)
        else:
            loss = self.target_criterion(out, labels)

        soft_pseudo = F.softmax(out[:, :-1], dim=-1)

        conf, preds = soft_pseudo.max(1)

        iou_tmp = jaccard_score(preds.detach().numpy(), labels.numpy(), average=None,
                                labels=np.arange(0, self.num_classes),
                                zero_division=0.)

        present_labels, class_occurs = np.unique(labels.numpy(), return_counts=True)
        present_labels = present_labels[present_labels != self.ignore_label]
        present_names = self.training_dataset.class2names[present_labels].tolist()
        present_names = [os.path.join(phase, p + '_iou') for p in present_names]
        results_dict = dict(zip(present_names, iou_tmp.tolist()))

        results_dict[f'{phase}/loss'] = loss
        results_dict[f'{phase}/iou'] = np.mean(iou_tmp[present_labels])

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

    def configure_optimizers(self):
        if self.scheduler_name is None:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD([{'params': self.source_model.parameters()}],
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam([{'params': self.source_model.parameters()}],
                                             lr=self.lr,
                                             weight_decay=self.weight_decay)
            else:
                raise NotImplementedError

            return optimizer
        else:
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD([{'params': self.source_model.parameters()}],
                                            lr=self.lr,
                                            momentum=self.momentum,
                                            weight_decay=self.weight_decay,
                                            nesterov=True)
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam([{'params': self.source_model.parameters()}],
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

            return [optimizer], [scheduler]

