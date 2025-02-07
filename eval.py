import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import MinkowskiEngine as ME

import utils.models as models
from utils.datasets.initialization import get_dataset
from configs import get_config
from utils.collation import CollateFN
from utils.pipelines import PLTTester

parser = argparse.ArgumentParser()
parser.add_argument("--config_file",
                    default="configs/source/synlidar_semantickitti.yaml",
                    type=str,
                    help="Path to config file")
parser.add_argument("--eval_source",
                    action='store_true',
                    default=False)
parser.add_argument("--eval_target",
                    action='store_true',
                    default=False)
parser.add_argument("--resume_path",
                    type=str,
                    default=None)

parser.add_argument("--is_student",
                    default=False,
                    action='store_true')


parser.add_argument("--save_predictions",
                    default=False,
                    action='store_true')


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

    
    ckpt = clean_state_dict(ckpt)
    model.load_state_dict(ckpt, strict=True)
    
    return model


def load_student_model(checkpoint_path, model):
    # reloads model
    def clean_state_dict(state):
        # clean state dict from names of PL
        for k in list(ckpt.keys()):  
            if "target_model" in k:
                print("target_model found")
                ckpt[k.replace("target_model.", "")] = ckpt[k] 
                del ckpt[k]        
            elif "student_model" in k:
                print("student_model found")
                ckpt[k.replace("student_model.", "")] = ckpt[k]
                del ckpt[k]
            elif "teacher_model" in k:
                print("student_model found")
                ckpt[k.replace("teacher_model.", "")] = ckpt[k]
                del ckpt[k]
            elif "source_model" in k:
                print("source_model found")
                ckpt[k.replace("source_model.", "")] = ckpt[k]
                del ckpt[k]
            elif "moco.model_q" in k:
                print("moco.model_q found")
                ckpt[k.replace("moco.model_q.", "")] = ckpt[k]
                del ckpt[k]
            elif "model" in k :
                print("model found in ", k)
                ckpt[k.replace("model.", "")] = ckpt[k] 
                del ckpt[k]
            elif 'moco' in k :
                del ckpt[k]

        return state

    
    try :
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]
    except KeyError:
        ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))["model_state_dict"]

    
    ckpt = clean_state_dict(ckpt)

    model.load_state_dict(ckpt, strict=True)
    return model


def test(config, resume_checkpoint):

    def get_dataloader(dataset, shuffle=False, pin_memory=True):
        return DataLoader(dataset,
                          batch_size=config.pipeline.dataloader.train_batch_size*2,
                          collate_fn=CollateFN(),
                          shuffle=shuffle,
                          num_workers=config.pipeline.dataloader.num_workers,
                          pin_memory=pin_memory)
        
    try:
        mapping_path = config.target_dataset.mapping_path
    except AttributeError('--> Setting default class mapping path!'):
        mapping_path = None


    validation_dataset, target_dataset, _ = get_dataset(dataset_name=config.target_dataset.name,
                                                                            dataset_path=config.target_dataset.dataset_path,
                                                                            voxel_size=config.target_dataset.voxel_size,
                                                                            augment_data=config.target_dataset.augment_data,
                                                                            version=config.target_dataset.version,
                                                                            sub_num=config.target_dataset.num_pts,
                                                                            num_classes=config.model.out_classes,
                                                                            ignore_label=config.target_dataset.ignore_label,
                                                                            mapping_path=mapping_path,
                                                                            save_predictions=args.save_predictions,)

    

    validation_dataloader = get_dataloader(validation_dataset, shuffle=False)
    target_dataloader = get_dataloader(target_dataset, shuffle=False)

    validation_dataloader = [validation_dataloader]
    target_dataloader = [target_dataloader]

    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if not args.is_student:
        model = load_model(resume_checkpoint, model)
        print(f'--> LOADED MODEL FROM {resume_checkpoint}')
    else:
        model = load_student_model(resume_checkpoint, model)
        print(f'--> LOADED STUDENT MODEL FROM {resume_checkpoint}')

    dataset = validation_dataset if args.eval_source else target_dataset

    main_dir, _ = os.path.split(resume_checkpoint)

    save_dir = os.path.join(main_dir, 'evaluation')
    save_preds_dir = os.path.join(main_dir, 'predictions')

    plt_model = PLTTester(model,
                          criterion=config.adaptation.losses.source_criterion,
                          dataset=dataset,
                          num_classes=config.model.out_classes,
                          checkpoint_path=resume_checkpoint,
                          save_predictions=args.save_predictions,
                          save_folder=save_preds_dir)

    wandb_logger = WandbLogger(project=config.pipeline.wandb.project_name,
                               name=save_dir,
                               offline=True)

    os.makedirs(save_dir, exist_ok=True)
    loggers = [wandb_logger]

    tester = Trainer(max_epochs=config.pipeline.epochs,
                     devices=[0],
                     logger=loggers,
                     default_root_dir=save_dir,
                     val_check_interval=1.0,
                     num_sanity_val_steps=0, 
                     accelerator='gpu')

    if args.eval_source:
        tester.test(plt_model, dataloaders=validation_dataloader, verbose=False)
    elif args.eval_target:
        tester.test(plt_model, dataloaders=target_dataloader, verbose=False)
    else:
        print('Not evaluating!')


def multiple_test(config, path):
    list_checkpoint = os.listdir(os.path.join(path, 'checkpoints'))
    list_checkpoint = [c for c in list_checkpoint if c.endswith('.ckpt')]
    #list_checkpoint = [c for c in list_checkpoint if c.endswith('.ckpt') or c.endswith('.pt')]
    #list_checkpoint = ['epoch=7-step=19871.ckpt']
    #list_checkpoint = [check for check in list_checkpoint if 'v1' not in check]
    print(list_checkpoint)
    for checkpoint in list_checkpoint:
        #if True :
        if not os.path.isfile(os.path.join(path, 'checkpoints', 'evaluation', 'results', checkpoint[:-5]+'_test.csv')):
            print(f'############### EVALUATING {checkpoint} ####################')
            test(config, os.path.join(path, 'checkpoints', checkpoint))
        else:
            print(f'############### EVALUATION {checkpoint} SKIPPED - ALREADY PRESENT ####################')


if __name__ == '__main__':
    args = parser.parse_args()

    config = get_config(args.config_file)

    # fix random seed
    os.environ['PYTHONHASHSEED'] = str(config.pipeline.seed)
    np.random.seed(config.pipeline.seed)
    torch.manual_seed(config.pipeline.seed)
    torch.cuda.manual_seed(config.pipeline.seed)
    torch.backends.cudnn.benchmark = True

    #all_folders = os.listdir(args.resume_path)
    all_folders = [args.resume_path]

    for checkpoint_folder in all_folders:
        #if checkpoint_folder.startswith('cosmix'):
        print(checkpoint_folder)

        multiple_test(config, checkpoint_folder)
