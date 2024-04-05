from torch.utils.data import DataLoader

from omegaconf import OmegaConf
from utils.models.minkunet import MinkUNet34
import torch
import argparse
import os
import MinkowskiEngine as ME


from datasets.semantickitti import semantickitti_collate, SemanticKitti
from datasets.synlidar import SynLiDAR, synlidar_collate

from tester import Tester


splits = ["train", "valid", "test"]




if __name__ == "__main__" :

    parser = argparse.ArgumentParser("./train.py")

    parser.add_argument(
        '--dataset', '-d',
        choices=['semantickitti', 'semanticposs', 'synlidar'], help='Choose between semantickitti and semanticposs')

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
        dataset = SemanticKitti(
            config=config,
            split=split
        )   

        collate_fn = semantickitti_collate


    elif ds == 'synlidar' :
        dataset = SynLiDAR(
            config=config, 
            split=split
        )
        collate_fn = synlidar_collate



    loader =  DataLoader(
            dataset=dataset, 
            batch_size=config.dataset.batch_size,
            num_workers=config.training.n_workers,
            collate_fn=collate_fn
        )
    
    net = MinkUNet34(
        in_channels=config.training.in_features,
        out_channels=dataset.n_classes
    )

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

