import sys
sys.path.insert(0, '../cosmix-uda')


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
            elif "student_model" in k:
                print("student_model found")
                ckpt[k.replace("student_model.", "")] = ckpt[k]
            elif "teacher_model" in k:
                print("student_model found")
                ckpt[k.replace("teacher_model.", "")] = ckpt[k]
            elif "source_model" in k:
                print("source_model found")
                ckpt[k.replace("source_model.", "")] = ckpt[k]
            elif "moco.model_q" in k:
                print("moco.model_q found")
                ckpt[k.replace("moco.model_q.", "")] = ckpt[k]
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

    n_points_per_class = 1000

    notdone = True

    for batch in loader:
        if not notdone:
            break
        print("Processing batch ...")
        stensor = ME.SparseTensor(coordinates=batch["coordinates"].int(), features=batch["features"])
        features = model(stensor, is_seg=False).F.detach().cpu().numpy()
        
        
        labels = batch['labels'].long().detach().cpu().numpy()
        
        for i in range(config.model.out_classes):
            class_indices = labels == i
            class_features[i].extend(features[class_indices])
            # Sample n_points_per_class features per class
            if len(class_features[i]) > n_points_per_class:
                class_features[i] = class_features[i][:n_points_per_class]

            
            total_length = sum(len(lst) for lst in class_features.values())
            print([len(lst) for lst in class_features.values()])
            if total_length >= n_points_per_class* config.model.out_classes:
                print("Total length exceeds n_points_per_class, breaking the loop.")
                notdone = False
                break


            




    tsne = TSNE(n_components=2, random_state=42)

    all_features = np.concatenate([np.array(class_features[i]) for i in range(config.model.out_classes)], axis=0)
    all_labels = np.concatenate([np.full((len(class_features[i]), 1), fill_value=i) for i in range(config.model.out_classes)], axis=0).squeeze()

    print(all_features.shape, all_labels.shape)

    # Fit t-SNE to all features
    tsne_embeddings = tsne.fit_transform(all_features)
    print("--> t-SNE fit to all features.")
    print(tsne_embeddings.shape)

    tsne_data = pd.DataFrame({
        'tSNE_1': tsne_embeddings[:, 0],  # First t-SNE dimension
        'tSNE_2': tsne_embeddings[:, 1],  # Second t-SNE dimension
        'class': all_labels  # Class labels
    })

    # Define class mapping for legend
    tsne_data['class_name'] = tsne_data['class'].apply(lambda x: class_mapping[x])

    # Set the figure size
    plt.figure(figsize=(8, 6))



    custom_palette = sns.color_palette("tab20", n_colors=config.model.out_classes)



    # Use seaborn's scatter plot to create the plot
    sns.scatterplot(
        data=tsne_data,
        x='tSNE_1',
        y='tSNE_2',
        hue='class_name',  # Color by class
        style='class_name',  # Differentiating with style (optional)
        palette=custom_palette,  # Color palette for distinct classes
        s=50,  # Marker size
        alpha=0.7,  # Transparency
        edgecolor='none',  # No outlines on the shapes
    )

    # Title, labels, and legend
    plt.title('t-SNE Visualization of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Class')

    # Show the plot
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