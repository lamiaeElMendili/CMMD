import os
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME
import random
from pytorch_lightning import seed_everything

import utils.models as models
from utils.datasets.initialization import get_dataset, get_concat_dataset
from configs import get_config
from utils.collation import CollateFN, CollateMerged
from utils.pipelines.masked_simm_pipeline import SimMaskedAdaptation

from utils.common.momentum import MomentumUpdater

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")

parser.add_argument("--method",
                    default="cmmd / adabn / cosmix",
                    type=str,
                    help="method")

def load_model(checkpoint_path, model):
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
            elif "model" in k:
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

def load_model_uda(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL

        new_state_dict = {}
        for k in list(ckpt.keys()):
            if "model" in k:
                ckpt[k.replace("model.", "")] = ckpt[k]
            del ckpt[k]

        for k in list(ckpt.keys()):
            if "bn" in k:
                new_state_dict[k.replace(".bn.", ".bn.bns.0.")] = ckpt[k]
                new_state_dict[k.replace(".bn.", ".bn.bns.1.")] = ckpt[k]
            else :
                new_state_dict[k] = ckpt[k]
            
     
        return new_state_dict

    
    try :
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    except KeyError:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"]

    

    ckpt = clean_state_dict(ckpt)

    
    
    model.load_state_dict(ckpt, strict=True)
    return model


def adapt(config, method):

    def get_dataloader(dataset, batch_size, shuffle=False, pin_memory=True, collation=None):
        if collation is None:
            collation = CollateFN()
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          collate_fn=collation,
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
    try:
        source_mapping_path = config.source_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for source!'):
        source_mapping_path = None

    try:
        target_mapping_path = config.target_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path for target!'):
        target_mapping_path = None

    source_training_dataset, source_validation_dataset, _ = get_dataset(dataset_name=config.source_dataset.name,
                                                                        dataset_path=config.source_dataset.dataset_path,
                                                                        voxel_size=config.source_dataset.voxel_size,
                                                                        augment_data=config.source_dataset.augment_data,
                                                                        version=config.source_dataset.version,
                                                                        sub_num=config.source_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.source_dataset.ignore_label,
                                                                        mapping_path=source_mapping_path,
                                                                        target_name=config.target_dataset.name,
                                                                        weights_path=config.source_dataset.weights_path)

    target_training_dataset, target_validation_dataset, _ = get_dataset(dataset_name=config.target_dataset.name,
                                                                        dataset_path=config.target_dataset.dataset_path,
                                                                        voxel_size=config.target_dataset.voxel_size,
                                                                        augment_data=config.target_dataset.augment_data,
                                                                        version=config.target_dataset.version,
                                                                        sub_num=config.target_dataset.num_pts,
                                                                        num_classes=config.model.out_classes,
                                                                        ignore_label=config.target_dataset.ignore_label,
                                                                        mapping_path=target_mapping_path)

    training_dataset = get_concat_dataset(source_dataset=source_training_dataset,
                                          target_dataset=target_training_dataset,
                                          augment_data=config.masked_dataset.augment_data,
                                          augment_mask_data=config.masked_dataset.augment_mask_data,
                                          remove_overlap=config.masked_dataset.remove_overlap)

    training_collation = CollateMerged()

    training_dataloader = get_dataloader(training_dataset,
                                         batch_size=config.pipeline.dataloader.train_batch_size,
                                         shuffle=True,
                                         collation=training_collation)


    target_validation_dataloader = get_dataloader(target_validation_dataset,
                                                  batch_size=config.pipeline.dataloader.train_batch_size,
                                                  shuffle=False,
                                                  collation=CollateFN())

    validation_dataloaders = [target_validation_dataloader]

    from utils.models.minkunet import MinkUNet34

    student_model = MinkUNet34(
        in_channels=config.model.in_feat_size,
        out_channels=config.model.out_classes
        )
    
    student_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(student_model)
    
    teacher_model = MinkUNet34(
        in_channels=config.model.in_feat_size,
        out_channels=config.model.out_classes
    )

    teacher_model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(teacher_model)
    
    
    if method == 'moco' or method == 'cosmix' or method == 'st' :
        teacher_model = load_model(config.adaptation.teacher_checkpoint, teacher_model)
        print(f'--> Loaded teacher checkpoint {config.adaptation.teacher_checkpoint}')

        student_model = load_model(config.adaptation.student_checkpoint, student_model)
        print(f'--> Loaded student checkpoint {config.adaptation.teacher_checkpoint}')



    momentum_updater = MomentumUpdater(base_tau=0.999, final_tau=0.999)

    if config.adaptation.self_paced:
        target_confidence_th = np.linspace(config.adaptation.target_confidence_th, 0.6, config.pipeline.epochs)
    else:
        target_confidence_th = config.adaptation.target_confidence_th
        
    
    if method == 'cmmd' :
        from utils.pipelines.cmmd_adaptation import Adaptation
        pl_module = Adaptation(training_dataset=training_dataset,
                                        source_validation_dataset=source_validation_dataset,
                                        target_validation_dataset=target_validation_dataset,
                                        student_model=student_model,
                                        teacher_model=teacher_model,
                                        momentum_updater=momentum_updater,
                                        source_criterion=config.adaptation.losses.source_criterion,
                                        target_criterion=config.adaptation.losses.target_criterion,
                                        other_criterion=config.adaptation.losses.other_criterion,
                                        source_weight=config.adaptation.losses.source_weight,
                                        target_weight=config.adaptation.losses.target_weight,
                                        filtering=config.adaptation.filtering,
                                        optimizer_name=config.pipeline.optimizer.name,
                                        train_batch_size=config.pipeline.dataloader.train_batch_size,
                                        val_batch_size=config.pipeline.dataloader.val_batch_size,
                                        lr=config.pipeline.optimizer.lr,
                                        num_classes=config.model.out_classes,
                                        clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                        scheduler_name=config.pipeline.scheduler.name,
                                        update_every=config.adaptation.momentum.update_every,
                                        weighted_sampling=config.adaptation.weighted_sampling,
                                        target_confidence_th=target_confidence_th,
                                        selection_perc=config.adaptation.selection_perc)

    elif method == 'st' :
        from utils.pipelines.st_adaptation import Adaptation
        pl_module = Adaptation(training_dataset=training_dataset,
                                    source_validation_dataset=source_validation_dataset,
                                    target_validation_dataset=target_validation_dataset,
                                    student_model=student_model,
                                    teacher_model=teacher_model,
                                    momentum_updater=momentum_updater,
                                    source_criterion=config.adaptation.losses.source_criterion,
                                    target_criterion=config.adaptation.losses.target_criterion,
                                    other_criterion=config.adaptation.losses.other_criterion,
                                    source_weight=config.adaptation.losses.source_weight,
                                    target_weight=config.adaptation.losses.target_weight,
                                    filtering=config.adaptation.filtering,
                                    optimizer_name=config.pipeline.optimizer.name,
                                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                                    lr=config.pipeline.optimizer.lr,
                                    num_classes=config.model.out_classes,
                                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                    scheduler_name=config.pipeline.scheduler.name,
                                    update_every=config.adaptation.momentum.update_every,
                                    weighted_sampling=config.adaptation.weighted_sampling,
                                    target_confidence_th=target_confidence_th,
                                    selection_perc=config.adaptation.selection_perc)        
    elif method == 'cmmd-cosmix' :
        from utils.pipelines.cmmd_adaptation_cosmix import Adaptation
        pl_module = Adaptation(config=config, training_dataset=training_dataset,
                                        source_validation_dataset=source_validation_dataset,
                                        target_validation_dataset=target_validation_dataset,
                                        momentum_updater=momentum_updater,
                                        source_criterion=config.adaptation.losses.source_criterion,
                                        target_criterion=config.adaptation.losses.target_criterion,
                                        other_criterion=config.adaptation.losses.other_criterion,
                                        source_weight=config.adaptation.losses.source_weight,
                                        target_weight=config.adaptation.losses.target_weight,
                                        filtering=config.adaptation.filtering,
                                        optimizer_name=config.pipeline.optimizer.name,
                                        train_batch_size=config.pipeline.dataloader.train_batch_size,
                                        val_batch_size=config.pipeline.dataloader.val_batch_size,
                                        lr=config.pipeline.optimizer.lr,
                                        num_classes=config.model.out_classes,
                                        clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                        scheduler_name=config.pipeline.scheduler.name,
                                        update_every=config.adaptation.momentum.update_every,
                                        weighted_sampling=config.adaptation.weighted_sampling,
                                        target_confidence_th=target_confidence_th,
                                        selection_perc=config.adaptation.selection_perc)      
    elif method == 'adabn' :
        from utils.pipelines.adabn_adaptation import Adaptation
        
        pl_module = Adaptation(
                    source_model=student_model,
                    target_model=teacher_model,
                    training_dataset=training_dataset,
                    source_validation_dataset=source_validation_dataset,
                    target_validation_dataset=target_validation_dataset,
                    source_criterion=config.adaptation.losses.source_criterion,
                    target_criterion=config.adaptation.losses.target_criterion,
                    other_criterion=config.adaptation.losses.other_criterion,
                    optimizer_name=config.pipeline.optimizer.name,
                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                    lr=config.pipeline.optimizer.lr,
                    num_classes=config.model.out_classes,
                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                    scheduler_name=config.pipeline.scheduler.name)
        
    elif method == 'mmd' :
        from utils.pipelines.mmd_adaptation import Adaptation
        
        pl_module = Adaptation(
                    source_model=student_model,
                    training_dataset=training_dataset,
                    source_validation_dataset=source_validation_dataset,
                    target_validation_dataset=target_validation_dataset,
                    source_criterion=config.adaptation.losses.source_criterion,
                    target_criterion=config.adaptation.losses.target_criterion,
                    other_criterion=config.adaptation.losses.other_criterion,
                    optimizer_name=config.pipeline.optimizer.name,
                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                    lr=config.pipeline.optimizer.lr,
                    num_classes=config.model.out_classes,
                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                    scheduler_name=config.pipeline.scheduler.name)

    elif method == 'minent' :
        from utils.pipelines.minentropy_adaptation import Adaptation
        
        pl_module = Adaptation(
                    source_model=student_model,
                    training_dataset=training_dataset,
                    source_validation_dataset=source_validation_dataset,
                    target_validation_dataset=target_validation_dataset,
                    source_criterion=config.adaptation.losses.source_criterion,
                    target_criterion=config.adaptation.losses.target_criterion,
                    other_criterion=config.adaptation.losses.other_criterion,
                    optimizer_name=config.pipeline.optimizer.name,
                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                    lr=config.pipeline.optimizer.lr,
                    num_classes=config.model.out_classes,
                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                    scheduler_name=config.pipeline.scheduler.name)    

    elif method == 'cosmix' :
        from utils.pipelines.masked_simm_pipeline import SimMaskedAdaptation
        
        pl_module = SimMaskedAdaptation(config=config,training_dataset=training_dataset,
                                    source_validation_dataset=source_validation_dataset,
                                    target_validation_dataset=target_validation_dataset,
                                    student_model=student_model,
                                    teacher_model=teacher_model,
                                    momentum_updater=momentum_updater,
                                    source_criterion=config.adaptation.losses.source_criterion,
                                    target_criterion=config.adaptation.losses.target_criterion,
                                    other_criterion=config.adaptation.losses.other_criterion,
                                    source_weight=config.adaptation.losses.source_weight,
                                    target_weight=config.adaptation.losses.target_weight,
                                    filtering=config.adaptation.filtering,
                                    optimizer_name=config.pipeline.optimizer.name,
                                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                                    lr=config.pipeline.optimizer.lr,
                                    num_classes=config.model.out_classes,
                                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                    scheduler_name=config.pipeline.scheduler.name,
                                    update_every=config.adaptation.momentum.update_every,
                                    weighted_sampling=config.adaptation.weighted_sampling,
                                    target_confidence_th=target_confidence_th,
                                    selection_perc=config.adaptation.selection_perc)

    elif method == 'st' :
        from utils.pipelines.st_adaptation import Adaptation
        pl_module = Adaptation(training_dataset=training_dataset,
                                    source_validation_dataset=source_validation_dataset,
                                    target_validation_dataset=target_validation_dataset,
                                    student_model=student_model,
                                    teacher_model=teacher_model,
                                    momentum_updater=momentum_updater,
                                    source_criterion=config.adaptation.losses.source_criterion,
                                    target_criterion=config.adaptation.losses.target_criterion,
                                    other_criterion=config.adaptation.losses.other_criterion,
                                    source_weight=config.adaptation.losses.source_weight,
                                    target_weight=config.adaptation.losses.target_weight,
                                    filtering=config.adaptation.filtering,
                                    optimizer_name=config.pipeline.optimizer.name,
                                    train_batch_size=config.pipeline.dataloader.train_batch_size,
                                    val_batch_size=config.pipeline.dataloader.val_batch_size,
                                    lr=config.pipeline.optimizer.lr,
                                    num_classes=config.model.out_classes,
                                    clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                    scheduler_name=config.pipeline.scheduler.name,
                                    update_every=config.adaptation.momentum.update_every,
                                    weighted_sampling=config.adaptation.weighted_sampling,
                                    target_confidence_th=target_confidence_th,
                                    selection_perc=config.adaptation.selection_perc) 

    elif method == 'moco' :
        from utils.pipelines.segcontrast_adaptation import Adaptation
        pl_module = Adaptation(config=config, training_dataset=training_dataset,
                                        source_validation_dataset=source_validation_dataset,
                                        target_validation_dataset=target_validation_dataset,
                                        momentum_updater=momentum_updater,
                                        source_criterion=config.adaptation.losses.source_criterion,
                                        target_criterion=config.adaptation.losses.target_criterion,
                                        other_criterion=config.adaptation.losses.other_criterion,
                                        source_weight=config.adaptation.losses.source_weight,
                                        target_weight=config.adaptation.losses.target_weight,
                                        filtering=config.adaptation.filtering,
                                        optimizer_name=config.pipeline.optimizer.name,
                                        train_batch_size=config.pipeline.dataloader.train_batch_size,
                                        val_batch_size=config.pipeline.dataloader.val_batch_size,
                                        lr=config.pipeline.optimizer.lr,
                                        num_classes=config.model.out_classes,
                                        clear_cache_int=config.pipeline.lightning.clear_cache_int,
                                        scheduler_name=config.pipeline.scheduler.name,
                                        update_every=config.adaptation.momentum.update_every,
                                        weighted_sampling=config.adaptation.weighted_sampling,
                                        target_confidence_th=target_confidence_th,
                                        selection_perc=config.adaptation.selection_perc)




    run_time = time.strftime("%Y_%m_%d_%H:%M", time.gmtime())


    run_name = f'{method}_{run_time}' 

    save_dir = os.path.join(config.pipeline.save_dir, config.source_dataset.name, config.target_dataset.name, f'{method}_{run_time}')


    if config.pipeline.wandb.project_name is not None:
        project_name = config.pipeline.wandb.project_name
    else:
        project_name = f'{config.source_dataset.name}->{config.target_dataset.name}'


    wandb_logger = WandbLogger(project=project_name,
                               name=run_name,
                               offline=config.pipeline.wandb.offline)
    

    loggers = [wandb_logger]

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(save_dir, 'checkpoints'), save_top_k=-1,save_on_train_epoch_end=False)
    callbacks = [checkpoint_callback]

    if config.pipeline.gpus is not None:
        strategy = "ddp" if len(config.pipeline.gpus) > 1 else None
    else:
        strategy = None
        
    print(save_dir)

    trainer = Trainer(max_epochs=config.pipeline.epochs,
                      gpus=config.pipeline.gpus,
                      accelerator=strategy,
                      default_root_dir=config.pipeline.save_dir,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      logger=loggers,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      val_check_interval=1.0,
                      num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                      resume_from_checkpoint=config.pipeline.lightning.resume_checkpoint,
                      callbacks=callbacks)

    trainer.fit(pl_module,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloaders)


if __name__ == '__main__':
    args = parser.parse_args()
    method = args.method
    config = get_config(args.config_file)
    


    seed_everything(config.pipeline.seed, workers=True)
    adapt(config, method)
