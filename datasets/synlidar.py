

import os

from torch.utils.data import Dataset
import numpy as np
import yaml

from datasets.utils import flatten
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from scipy.linalg import expm, norm

from omegaconf import OmegaConf

class SynLiDAR(Dataset) :

    def __init__(self, config, split='train') -> None: 
        """split=[train, val, test]"""
        super().__init__()
        
        self.config = config
        self.name = 'SynLiDAR'
        self.root_dir = self.config.dataset.root_dir
        self.voxel_size = self.config.dataset.voxel_size
        self.augment = self.config.training.augment
        self.split = split
        self.num_points = config.dataset.num_points

        self.rotations = [(-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)]
        self.scale_augmentation_bound = (0.95, 1.05)





        self.dataset_path = os.path.join(self.root_dir, 'sequences')



        self.sequences = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

        self.get_sequences()

        print('SEQUENCES', self.splits.keys())
        print('TRAIN SEQUENCES', self.splits[self.split].keys())

        split_sequences = list(self.splits[self.split].keys())

        
        self.files = {
            'input' : flatten([[os.path.join(sequence, 'velodyne', f'{frame:06}.bin') for frame in self.splits[split][sequence]] for sequence in split_sequences]),
            }
        


        if self.split != 'test' :
            self.files['labels'] = [i.replace('.bin', '.label') for i in self.files['input']]
            self.files['labels'] = [i.replace('velodyne', 'labels') for i in self.files['labels']]



        print(f'Found {self.__len__()} frames in Total')

        

        remapdict = config.learning_map
    

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

        self.weights = np.load('datasets/weights/synlidar2semantickitti_correct.npy')



    def get_sequences(self) :
        split_path = os.path.join(self.dataset_path, 'synlidar.pkl')

        if not os.path.exists(split_path) :
            self.splits = {'train': {s: [] for s in self.sequences},
                            'val': {s: [] for s in self.sequences}}

            for sequence in self.sequences:
                num_frames = len(os.listdir(os.path.join(self.dataset_path, sequence, 'labels')))
                valid_frames = []

                for v in np.arange(num_frames):
                    pcd_path = os.path.join(self.dataset_path, sequence, 'velodyne', f'{int(v):06d}.bin')
                    label_path = os.path.join(self.dataset_path, sequence, 'labels', f'{int(v):06d}.label')

                    if os.path.isfile(pcd_path) and os.path.isfile(label_path):
                        valid_frames.append(v)
                train_selected = np.random.choice(valid_frames, int(num_frames/10), replace=False)


                for t in train_selected:
                    valid_frames.remove(t)

                validation_selected = np.random.choice(valid_frames, int(num_frames/100), replace=False)
                self.splits['train'][sequence].extend(train_selected)
                self.splits['val'][sequence].extend(validation_selected)            
            torch.save(self.splits, split_path)

        else :
            self.splits = torch.load(split_path)


    def __getitem__(self, index) :

        bin_filename = os.path.join(self.dataset_path, self.files['input'][index])
        

        points = np.fromfile(bin_filename, dtype=np.float32).reshape((-1, 4))
        pos = points[:, :3]

        if self.config.dataset.use_intensity :
            features = points[:, 3]
        else :
            features = np.ones((pos.shape[0], 1), dtype=np.float32)


        if self.split == 'test' :
            labels = np.zeros((points.shape[0], ), dtype=np.int32)
        else :
            label_filename = os.path.join(self.dataset_path, self.files['labels'][index])
            labels = np.fromfile(label_filename, dtype=np.uint32)
            
            labels = labels.reshape((-1))
            labels = self.remap_labels(labels)



        
        indices = np.arange(pos.shape[0])
        if self.split == 'train' and self.config.training.augment :
            indices = self.sample(points)
            pos = pos[indices]
            features = features[indices]
            labels = labels[indices]
            pos = self.augmentation(pos)

        sparse_points, sparse_features, sparse_labels, sparse_indices, reverse_indices = ME.utils.sparse_quantize(coordinates=pos, features=features, labels=labels, 
                                                                                 ignore_label=self.ignore_label, quantization_size=self.config.dataset.voxel_size, 
                                                                                 return_index=True, return_inverse=True)



        sparse_points, sparse_features, sparse_labels, reverse_indices = torch.from_numpy(sparse_points), torch.from_numpy(sparse_features), torch.from_numpy(sparse_labels), \
                    torch.from_numpy(reverse_indices)

        sparse_indices = torch.from_numpy(sparse_indices)

        return {
            'points' : sparse_points, 
            'features' : sparse_features, 
            'labels' : sparse_labels,
            'indices' : sparse_indices,
            'index' : index,
            'reverse_indices' : reverse_indices
        }


    def sample(self, points) :

        total_points = points.shape[0]

        
        if self.num_points <= total_points :
            indices = np.random.choice(np.arange(total_points), self.num_points, replace=False)

        else :
            indices = np.random.choice(np.arange(total_points), self.num_points - total_points, replace=False)
            indices = np.concatenate([np.arange(total_points), indices])

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
        lower_half = label & 0xFFFF   # get lower half for semantics

        
        lower_half = self.remap_lut[lower_half]  # do the remapping of semantics
        
        return lower_half.astype(np.uint32)





def synlidar_collate(data) :
      batch_coords = [batch['points'] for batch in data]
      batch_coords = ME.utils.batched_coordinates(batch_coords)
      batch_features = [batch['features'] for batch in data]
      batch_labels = [batch['labels'] for batch in data]
      batch_indices = [batch['indices'] for batch in data]
      batch_index = [batch['index'] for batch in data]
      batch_reverse_indices = [batch['reverse_indices'] for batch in data]
      

      return {
          'points' : batch_coords,
          'features' : torch.cat(batch_features, dim=0).float(),
          'labels' : torch.cat(batch_labels, dim=0).int(),
          'indices' : torch.cat(batch_indices, dim=0).int(),
          'index' : batch_index,
          'reverse_indices' : torch.cat(batch_reverse_indices, dim=0).int(),
      }


if __name__ == "__main__" :


    config_file = r'/mnt/home/lamiae/CMMD/3DCMMD/configs/training/synlidar-semantickitti.yaml'


    config = OmegaConf.load(config_file)



    sl = SynLiDAR(config=config, split='train')

