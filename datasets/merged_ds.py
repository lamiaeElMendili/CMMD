from torch.utils.data import Dataset
import numpy as np
import MinkowskiEngine as ME
import torch
import collections
from scipy.linalg import expm, norm

class MergeDataset(Dataset) :
    def __init__(self, source_training_dataset, target_training_dataset, config) -> None:
        super().__init__()

        self.source_training_dataset = source_training_dataset
        self.target_training_dataset = target_training_dataset
        self.config = config
        self.voxel_size = self.config.dataset.voxel_size

        self.num_points = config.dataset.num_points
        self.rotations = [(-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20), (-np.pi / 20, np.pi / 20)]
        self.scale_augmentation_bound = (0.95, 1.05)


        self.translation_augmentation_bound = None

        self.scale_augmentation_bound_mask = (0.95, 1.05)
        self.rotation_augmentation_bound_mask = (None, None, (-np.pi / 20, np.pi / 20))
        self.translation_augmentation_ratio_bound_mask = None


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


        self.weights = self.source_training_dataset.weights

        
        print("Number of classes for learning ", self.n_classes)




    
    def __getitem__(self, index) :
        if index < len(self.source_training_dataset) :
            source_data = self.source_training_dataset.__getitem__(index)
        else :
            new_index = np.random.choice(len(self.source_training_dataset), 1)
            source_data = self.source_training_dataset.__getitem__(int(new_index))

        
        target_data = self.target_training_dataset.__getitem__(index)

        source_data = {f'source_{k}': v for k, v in source_data.items()}
        target_data = {f'target_{k}': v for k, v in target_data.items()}  
        data = {**source_data, **target_data}    

        return data  



    
    def __len__(self) :
        return len(self.target_training_dataset)
        
    def M(self, axis, theta):
        return expm(np.cross(np.eye(3), axis / norm(axis) * theta))

    # for cosmix
    def get_transformation_matrix(self, use_augmentation) :
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)

        # Transform pointcloud coordinate to voxel coordinate.
        # 1. Random rotation
        rot_mat = np.eye(3)
        if use_augmentation and self.rotation_augmentation_bound_mask is not None:
            if isinstance(self.rotation_augmentation_bound_mask, collections.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.rotation_augmentation_bound_mask):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(self.M(axis, theta))
                # Use random order
                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat
        # 2. Scale and translate to the voxel space.
        scale = 1
        if use_augmentation and self.scale_augmentation_bound_mask is not None:
            scale *= np.random.uniform(*self.scale_augmentation_bound_mask)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)

        # 3. Translate
        if use_augmentation and self.translation_augmentation_ratio_bound_mask is not None:
            tr = [np.random.uniform(*t) for t in self.translation_augmentation_ratio_bound_mask]
            rotation_matrix[:3, 3] = tr
        # Get final transformation matrix.
        return voxelization_matrix, rotation_matrix



def merge_collate(data) :
      
      batch_source_coords = [batch['source_points'] for batch in data]
      batch_source_coords = ME.utils.batched_coordinates(batch_source_coords)
      batch_source_features = [batch['source_features'] for batch in data]
      batch_source_labels = [batch['source_labels'] for batch in data]
      batch_source_indices = [batch['source_indices'] for batch in data]

      batch_target_coords = [batch['target_points'] for batch in data]
      batch_target_coords = ME.utils.batched_coordinates(batch_target_coords)
      batch_target_features = [batch['target_features'] for batch in data]
      batch_target_labels = [batch['target_labels'] for batch in data]
      batch_target_indices = [batch['target_indices'] for batch in data]


      batch_target_index = [batch['target_index'] for batch in data]
      batch_target_reverse_indices = [batch['target_reverse_indices'] for batch in data]
      

      return {
          'source_points' : batch_source_coords,
          'source_features' : torch.cat(batch_source_features, dim=0).float(),
          'source_labels' : torch.cat(batch_source_labels, dim=0).int(),
          'source_indices' : torch.cat(batch_source_indices, dim=0).int(),

          'target_points' : batch_target_coords,
          'target_features' : torch.cat(batch_target_features, dim=0).float(),
          'target_labels' : torch.cat(batch_target_labels, dim=0).int(),
          'target_indices' : torch.cat(batch_target_indices, dim=0).int(),

          'target_index' : batch_target_index,
          'target_reverse_indices' : torch.cat(batch_target_reverse_indices, dim=0).int(),
      }








