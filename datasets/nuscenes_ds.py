import os

from torch.utils.data import Dataset
import numpy as np
import yaml

import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from omegaconf import OmegaConf

from nuscenes import NuScenes as _NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from scipy.linalg import expm, norm


class NuScenes(Dataset) :
    def __init__(self, config, split='train') -> None:
        super().__init__()

        self.config = config
        self.name = 'NuScenes'
        self.root_dir = self.config.dataset.root_dir

        self.voxel_size = self.config.dataset.voxel_size
        self.augment = self.config.training.augment
        self.split = split

        self.num_points = config.dataset.num_points
        self.rotations = [(-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)]
        self.scale_augmentation_bound = (0.95, 1.05)

        self.nusc =_NuScenes(version='v1.0-trainval',
                             dataroot=self.root_dir,
                             verbose=True)
        

        self.scenes = create_splits_scenes()[self.split]
        
        print(self.scenes)

        

        self.files = {
            'input' : []
        }



        for scene_idx in range(len(self.nusc.scene)) :
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in self.scenes :
                token = scene["first_sample_token"]

                while token != "" :
                    current_sample = self.nusc.get("sample", token=token)
                    token = current_sample["next"]

                    self.files["input"].append(current_sample["data"]['LIDAR_TOP'])


        

        remapdict = self.config.learning_map

        learning_map_inv = self.config.learning_map_inv

        label_names = self.config.labels

        self.class2names = {label_id: label_names[class_id] for label_id, class_id in learning_map_inv.items()}

        print(self.class2names)




        maxkey = max(remapdict.keys())

        self.remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.remap_lut[list(remapdict.keys())] = list(remapdict.values())

        self.ignore_label = self.config.learning_ignore

        self.n_classes = len(np.unique(list(remapdict.values())))

        if self.ignore_label is not None :
            self.n_classes -= 1

        
        print("Number of classes for learning ", self.n_classes)





    def __getitem__(self, index):

        token = self.files["input"][index]
    
        bin_data = self.nusc.get('sample_data', token=token)
        label_data = self.nusc.get('lidarseg', token=token)
        
        
        points = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, bin_data["filename"])).points.T

        pos = points[:, 0:3]

        if self.config.dataset.use_intensity :
            features = points[:, 3]
        else :
            features = np.ones((pos.shape[0], 1), dtype=np.float32)

        if self.split == 'test' :
            labels = np.zeros((points.shape[0], ), dtype=np.int32)
        else :
            labels = load_bin_file(os.path.join(self.nusc.dataroot, label_data["filename"]))
            labels = labels.reshape((-1))
            labels = self.remap_labels(labels)



        indices = np.arange(pos.shape[0])
        if self.split == 'train' and self.config.training.augment :
            indices = self.sample(points)
            pos = pos[indices]
            features = features[indices]
            labels = labels[indices]
            pos = self.augmentation(pos)

        sparse_points, sparse_features, sparse_labels, sparse_indices = ME.utils.sparse_quantize(coordinates=pos, features=features, labels=labels, 
                                                                                 ignore_label=self.ignore_label, quantization_size=self.config.dataset.voxel_size, 
                                                                                 return_index=True)
        
        sparse_points, sparse_features, sparse_labels = torch.from_numpy(sparse_points), torch.from_numpy(sparse_features), torch.from_numpy(sparse_labels)

        sparse_indices = torch.from_numpy(sparse_indices)

        return {
            'points' : sparse_points, 
            'features' : sparse_features, 
            'labels' : sparse_labels,
            'indices' : sparse_indices
        }
        

        
    def sample(self, points) :

        total_points = points.shape[0]
        
        if self.num_points <= total_points :
            indices = np.random.choice(np.arange(total_points), self.num_points, replace=False)

        else :
            indices = indices = np.random.choice(np.arange(total_points), self.num_points, replace=True)
            

        return indices

    def augmentation(self, pos) :

        rotations = []
        voxelization_matrix = np.eye(3)
        for (i, rot) in enumerate(self.rotations) :
            axis = np.zeros(3)
            axis[i] = 1
            angle = np.random.uniform(*rot)
            rotations.append(expm(np.cross(np.eye(3), axis / norm(axis) * angle)))

        np.random.shuffle(rotations)
        rotations = rotations[0] @ rotations[1] @ rotations[2]
            
        scale = np.random.uniform(*self.scale_augmentation_bound)
        np.fill_diagonal(voxelization_matrix, scale)

        pos = pos @ rotations @ voxelization_matrix
        return pos


    def __len__(self) :
        return len(self.files['input']) 

    def remap_labels(self, label) :
 
        lower_half = self.remap_lut[label]  # do the remapping of semantics 
        return lower_half.astype(np.uint32)








def nuscenes_collate(data) :
      batch_coords = [batch['points'] for batch in data]
      batch_coords = ME.utils.batched_coordinates(batch_coords)
      batch_features = [batch['features'] for batch in data]
      batch_labels = [batch['labels'] for batch in data]
      batch_indices = [batch['indices'] for batch in data]
      

      return {
          'points' : batch_coords,
          'features' : torch.cat(batch_features, dim=0).float(),
          'labels' : torch.cat(batch_labels, dim=0).int(),
          'indices' : torch.cat(batch_indices, dim=0).int()
      }












if __name__ == "__main__" :

    config_file = r'/mnt/home/lamiae/CMMD/3DCMMD/configs/training/nuscenes.yaml'


    config = OmegaConf.load(config_file)

    ns = NuScenes(config=config,
                  split='train')
    

    batch = ns[0]
    labels = batch['labels']

    print(torch.unique(labels))
    