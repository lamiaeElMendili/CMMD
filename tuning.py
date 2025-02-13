import optuna
from optuna.trial import TrialState


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

import utils.models as models
from utils.datasets.initialization import get_dataset, get_concat_dataset
from configs import get_config
from utils.collation import CollateFN, CollateMerged
from utils.common.momentum import MomentumUpdater
from utils.models.minkunet import MinkUNet34

from utils.pipelines.segcontrast_adaptation import Adaptation
from tqdm import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")


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

def get_dataloader(dataset, batch_size, shuffle=False, pin_memory=True, collation=None):
    if collation is None:
        collation = CollateFN()

    return DataLoader(dataset,
                        batch_size=batch_size,
                        collate_fn=collation,
                        shuffle=shuffle,
                        num_workers=config.pipeline.dataloader.num_workers,
                        pin_memory=pin_memory)


def objective(trial) :
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
    teacher_model = load_model(config.adaptation.teacher_checkpoint, teacher_model)
    print(f'--> Loaded teacher checkpoint {config.adaptation.teacher_checkpoint}')

    student_model = load_model(config.adaptation.student_checkpoint, student_model)
    print(f'--> Loaded student checkpoint {config.adaptation.teacher_checkpoint}')

    momentum_updater = MomentumUpdater(base_tau=0.999, final_tau=0.999)
    target_confidence_th = config.adaptation.target_confidence_th

    hyperparameters = {
            "config.adaptation.cmmd.feature_dimension" : trial.suggest_categorical("config.adaptation.cmmd.feature_dimension", [128, 64, 32]),
            "config.adaptation.cmmd.lamda_source" : trial.suggest_categorical("config.adaptation.cmmd.lamda_source", [0.0, 1.0]),
            "config.adaptation.cmmd.kernel" : trial.suggest_categorical("config.adaptation.cmmd.kernel", ['linear', 'gaussian']),
            "target_confidence_th" : trial.suggest_float("target_confidence_th", 0.85, 0.9),
            "config.adaptation.cmmd.queue_size" : trial.suggest_int("config.adaptation.cmmd.queue_size", 65536, 200000)
        }


    adapter = Adaptation(config=config, training_dataset=training_dataset,
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
    
    for attr, val in hyperparameters.items() :
       setattr(adapter, attr, val)   
    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                            name=f'{study_name}_optuna',
                            offline=config.pipeline.wandb.offline)


    loggers = [wandb_logger]   
    save_dir = os.path.join(config.pipeline.save_dir)
    strategy = None
    trainer = Trainer(max_epochs=1,
                      gpus=config.pipeline.gpus,
                      accelerator=strategy,
                      logger=loggers,
                      default_root_dir=config.pipeline.save_dir,
                      limit_train_batches=0.5,
                      weights_save_path=save_dir,
                      precision=config.pipeline.precision,
                      check_val_every_n_epoch=config.pipeline.lightning.check_val_every_n_epoch,
                      num_sanity_val_steps=config.pipeline.lightning.num_sanity_val_steps,
                      resume_from_checkpoint=config.pipeline.lightning.resume_checkpoint)
    
    trainer.fit(adapter,
                train_dataloaders=training_dataloader,
                val_dataloaders=validation_dataloaders)


    val_miou = trainer.logged_metrics['target_validation/iou']
    trial.report(val_miou, 1)

    
    return val_miou
                       







if __name__ == "__main__" :
    args = parser.parse_args()
    config = get_config(args.config_file)
    

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

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

    source_validation_dataloader = get_dataloader(source_validation_dataset,
                                                  batch_size=config.pipeline.dataloader.train_batch_size,
                                                  shuffle=False,
                                                  collation=CollateFN())

    target_validation_dataloader = get_dataloader(target_validation_dataset,
                                                  batch_size=config.pipeline.dataloader.train_batch_size,
                                                  shuffle=False,
                                                  collation=CollateFN())

    validation_dataloaders = [target_validation_dataloader]



    study_name = 'cmmd'
    output_folder = os.path.join(config.pipeline.save_dir, 'optuna')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    #if os.path.exists(f'{output_folder}/{study_name}.db'):
    #    os.remove(f'{output_folder}/{study_name}.db')

    study = optuna.create_study(study_name=study_name, sampler=optuna.samplers.TPESampler(), 
                                storage=f'sqlite:///{output_folder}/{study_name}.db' , direction="maximize",
                                load_if_exists=True)


    study.enqueue_trial({
            "config.adaptation.cmmd.feature_dimension" : 128,
            "config.adaptation.cmmd.lamda_source" : 0.0,
            "config.adaptation.cmmd.kernel" : 'linear',
            "target_confidence_th" : 0.9,
            "config.adaptation.cmmd.queue_size" : 65536
        })

    study.enqueue_trial({
                "config.adaptation.cmmd.feature_dimension" : 128,
                "config.adaptation.cmmd.lamda_source" : 1.0,
                "config.adaptation.cmmd.kernel" : 'linear',
                "target_confidence_th" : 0.9,
                "config.adaptation.cmmd.queue_size" : 65536
            })



    study.optimize(objective, n_trials=100)
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
