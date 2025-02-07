from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from utils.models.minkunet import MinkUNet34
import torch
import argparse
import os
import MinkowskiEngine as ME


from utils.datasets.semantickitti import SemanticKITTIDataset
from utils.datasets.semanticposs import SemanticPOSSDataset
from utils.collation import CollateFN
from tester import Tester
import numpy as np

splits = ["train", "valid", "test"]

synlidar2kitti = np.array(['car', 'bicycle', 'motorcycle',  'truck', 'other-vehicle', 'person',
                           'bicyclist', 'motorcyclist',
                           'road', 'parking', 'sidewalk', 'other-ground',
                           'building', 'fence', 'vegetation', 'trunk',
                           'terrain', 'pole', 'traffic-sign'])


nuscenes2kitti = np.array(['car', 'bicycle', 'motorcycle',  'truck', 'bus', 'person',
                           'road', 'sidewalk', 'terrain', 'vegetation'])



synlidar2poss = np.array(['person', 'rider', 'car', 'trunk',
                          'plants', 'traffic-sign', 'pole', 'garbage-can',
                          'building', 'cone', 'fence', 'bike', 'ground'])



nuscenes2semanticposs = np.array(['person', 'bicycle', 'car',  'ground', 'vegetation', 'manmade',
                           'road', 'sidewalk', 'terrain', 'vegetation'])


if __name__ == "__main__" :

    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--dataset', '-d',
        choices=['semantickitti', 'semanticposs'], help='Choose between semantickitti and semanticposs')

    parser.add_argument(
        '--source', '-sd',
        choices=['synlidar', 'nuscenes'], help='Choose between semantickitti and semanticposs')
    


    parser.add_argument(
            '--split', '-s',
            type=str,
            required=False,
            choices=["train", "val", "test"],
            default="valid",
            help='Split to evaluate on. One of ' +
            str(splits) + '. Defaults to %(default)s',
        )

    parser.add_argument(
        '--checkpoint_folder', '-chkp',
        type=str,
        required=True
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True
    )

    parser.add_argument("--save_predictions",
                        default=False,
                        action='store_true')


    FLAGS, unparsed = parser.parse_known_args()

    ds = FLAGS.dataset
    split = FLAGS.split
    checkpoint_folder = FLAGS.checkpoint_folder
    config_file = FLAGS.config
    save_predictions = FLAGS.save_predictions




    config = OmegaConf.load(config_file)


    if ds == 'semantickitti' :

        dataset = SemanticKITTIDataset(dataset_path=config.dataset.dataset_path,
                                        mapping_path=config.dataset.mapping_path,
                                        version='full',
                                        phase='validation',
                                        voxel_size=config.dataset.voxel_size,
                                        augment_data=False,
                                        num_classes=config.model.out_classes,
                                        ignore_label=config.dataset.ignore_label,
                                        weights_path=None)
        
        dataset.class2names = synlidar2kitti if FLAGS.source == 'synlidar' else nuscenes2kitti



    elif ds == 'semanticposs' :
        dataset = SemanticPOSSDataset(dataset_path=config.dataset.dataset_path,
                                        mapping_path=config.dataset.mapping_path,
                                        version='full',
                                        phase='validation',
                                        voxel_size=config.dataset.voxel_size,
                                        augment_data=False,
                                        num_classes=config.model.out_classes,
                                        ignore_label=config.dataset.ignore_label,
                                        weights_path=None, 
                                        return_inverse=True)
        
        dataset.class2names = synlidar2poss if FLAGS.source == 'synlidar' else nuscenes2semanticposs


    collate_fn = CollateFN()

    loader =  DataLoader(
            dataset=dataset, 
            batch_size=config.dataset.batch_size,
            num_workers=config.dataset.n_workers,
            collate_fn=collate_fn
        )
    
    net = MinkUNet34(config.model.in_feat_size, config.model.out_classes)


    net = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(net)


    if os.path.isfile(checkpoint_folder) :

        output_folder = os.path.join(os.path.dirname(os.path.dirname(checkpoint_folder)), f'evaluation-{ds}-{split}')

        os.makedirs(output_folder, exist_ok=True)

        print(f'Testing Checkpoint {checkpoint_folder}')

        tester = Tester(
            config=config,
            loader=loader,
            net=net, 
            chkp_path=checkpoint_folder,
            output_folder=output_folder,
            save_predictions=save_predictions
        )

        tester.test()     


    else :

        output_folder = os.path.join(checkpoint_folder, f'evaluation-{ds}-{split}')

        os.makedirs(output_folder, exist_ok=True)
        list_checkpoints = os.listdir(os.path.join(checkpoint_folder, 'checkpoints'))

        for chkp in  list_checkpoints:


            print(f'Testing Checkpoint {chkp}')

            tester = Tester(
                config=config,
                loader=loader,
                net=net, 
                chkp_path=os.path.join(checkpoint_folder, 'checkpoints', chkp),
                output_folder=output_folder,
                save_predictions=save_predictions
            )

            tester.test()

