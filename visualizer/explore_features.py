import sys
sys.path.insert(0, '../cosmix-uda')

import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import torch
from torch.utils.data import Subset
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from configs import get_config
from utils.datasets.initialization import get_dataset
from utils.collation import CollateFN
import random
import utils.models as models
import numpy as np


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
            else :
                if "model" in k :
                    print("model found in ", k)
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


def get_tsne(config, checkpoint_path, is_source) :
    np.random.seed(0)  # For reproducibility
    mapping_path = config.target_dataset.mapping_path


    _, dataset, _ = get_dataset(dataset_name=config.target_dataset.name,
                                dataset_path=config.target_dataset.dataset_path,
                                voxel_size=config.target_dataset.voxel_size,
                                augment_data=config.target_dataset.augment_data,
                                version=config.target_dataset.version,
                                sub_num=config.target_dataset.num_pts,
                                num_classes=config.model.out_classes,
                                ignore_label=config.target_dataset.ignore_label,
                                mapping_path=mapping_path)


    # Sample 200 indices randomly from the loader
    indices = torch.randperm(len(dataset))[:10000]

    print("--> Sampled 200 indices from the loader.")

    subset = Subset(dataset, indices)

    class_mapping = dataset.class2names

    print("--> Created a subset of the dataset.")   

    loader = DataLoader(subset,
                          batch_size=config.pipeline.dataloader.train_batch_size,
                          collate_fn=CollateFN(),
                          shuffle=True,
                          num_workers=1,
                          pin_memory=True)
    
    print("--> Created a dataloader.")


    Model = getattr(models, config.model.name)
    model = Model(config.model.in_feat_size, config.model.out_classes)

    model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)

    if is_source:
        model = load_model(checkpoint_path, model)
        print("Source model loaded from ", checkpoint_path)
    else:
        model = load_student_model(checkpoint_path, model)
        print("Student model loaded from ", checkpoint_path)

    class_features = {i: [] for i in range(config.model.out_classes)}

    n_points_per_class = 50
    num_classes = config.model.out_classes

    notdone = True
    
    for batch in loader:
        if not notdone:
            break
        print("Processing batch ...")
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        labels = batch['labels'].long().detach().cpu().numpy()

        out = model(stensor, is_seg=True).F
        print("labels shape:", labels.shape)

        conf = F.softmax(out, dim=-1)
        preds = conf.max(dim=-1).indices.detach().cpu().numpy() 

        confusing_classes = (-out).argsort()[:, :19].detach().cpu().numpy() 
        mask = labels != confusing_classes[:, 0]
        confusing_classes = confusing_classes[mask]  # Remove correct predictions
        labels = labels[mask]
        print("Confusing classes:", confusing_classes.shape, labels.shape)
        
    

        state_dict = model.state_dict()
        final_weights = state_dict['final.kernel'] # (features, num_classes)         
        num_channels = final_weights.shape[0]

        print("Final weights shape:", final_weights.shape)
        

        distance_matrix = np.zeros((num_channels, num_classes))

        #distance_matrix = torch.cdist(final_weights[:, labels], final_weights[:, confusing_classes[:, 0]]).detach().cpu().numpy()
        #distance_matrix = torch.abs(final_weights[:, 10] - final_weights[:, confusing_classes[:, 9]]).detach().cpu().numpy()
        ##_, all_index = torch.topk(torch.tensor(distance_matrix), 5, dim=0, largest=False) # reyurns the indices of the top 5 smallest values
        #print(distance_matrix.shape, _.shape)
        
        #distance_matrix = _
        #print("Distance matrix shape:", distance_matrix.shape)

        for confused_class in range(num_classes):

            for i in range(num_classes):
                distance_matrix[:, i] = torch.abs(final_weights[:, i] - final_weights[:, confused_class]).detach().cpu().numpy()


            sns.heatmap(distance_matrix, annot=True, cmap='viridis', fmt=".2f",
                            xticklabels=[f"Class {i}" for i in range(num_classes)],
                        yticklabels=[f"Class {i}" for i in range(5)])
            plt.title("Euclidean Distance between Class Weights")
            plt.xlabel("Class")
            plt.ylabel("Class")
            plt.show()   





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file",
                        default="configs/source/synlidar2semantickitti.yaml",
                        type=str,
                        help="Path to config file")


    parser.add_argument("--resume_path",
                        type=str,
                        help="Path to the ckpt file")
    
    parser.add_argument("--is_source",
                        action="store_true",
                        help="Whether the model is pretrained on source or not")
    


    args = parser.parse_args()

    config = get_config(args.config_file)

    checkpoint_path = args.resume_path

    get_tsne(config, checkpoint_path, args.is_source)