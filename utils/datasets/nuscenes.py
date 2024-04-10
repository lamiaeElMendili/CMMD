import os
import torch
import yaml
import numpy as np
import tqdm

import MinkowskiEngine as ME
from utils.datasets.dataset import BaseDataset


from nuscenes import NuScenes as NuScenes_
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_io import load_bin_file
from nuscenes.utils.splits import create_splits_scenes


ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))


class NuScenesDataset(BaseDataset):
    def __init__(self,
                 version: str = 'full',
                 phase: str = 'train',
                 dataset_path: str = '/data/csaltori/SemanticKITTI/data/sequences',
                 mapping_path: str = '_resources/semantic-kitti.yaml',
                 weights_path: str = None,
                 voxel_size: float = 0.05,
                 use_intensity: bool = False,
                 augment_data: bool = False,
                 sub_num: int = 50000,
                 device: str = None,
                 num_classes: int = 7,
                 ignore_label: int = None):

        if weights_path is not None:
            weights_path = os.path.join(ABSOLUTE_PATH, weights_path)
        super().__init__(version=version,
                         phase=phase,
                         dataset_path=dataset_path,
                         voxel_size=voxel_size,
                         sub_num=sub_num,
                         use_intensity=use_intensity,
                         augment_data=augment_data,
                         device=device,
                         num_classes=num_classes,
                         ignore_label=ignore_label,
                         weights_path=weights_path)
        


        self.nusc = NuScenes_(version=version, dataroot=dataset_path, verbose=True)

        phase_scenes = create_splits_scenes()[phase]
        

        # create a list of camera & lidar scans
        self.list_keyframes = []
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                current_sample_token = scene["first_sample_token"]

                # Loop to get all successive keyframes
                list_data = []
                while current_sample_token != "":
                    current_sample = self.nusc.get("sample", current_sample_token)
                    list_data.append(current_sample["data"])
                    current_sample_token = current_sample["next"]

                # Add new scans in the list
                self.list_keyframes.extend(list_data)



        self.name = 'NuScenesDataset'
        self.maps = yaml.safe_load(open(os.path.join(ABSOLUTE_PATH, mapping_path), 'r'))

        self.pcd_path = []
        self.label_path = []

        remap_dict_val = self.maps["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())

        self.remap_lut_val = remap_lut_val





    def __len__(self):
        return len(self.list_keyframes)

    def __getitem__(self, i: int):

        data = self.list_keyframes[i]
        lidar_token = data['LIDAR_TOP']
        lidar_rec = self.nusc.get('sample_data', lidar_token)
        pcd = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_rec['filename']))
        pcd = pcd.points.T
        points = pcd[:,:3]

        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)

        lidarseg_label_filename = os.path.join(self.nusc.dataroot, self.nusc.get('lidarseg', lidar_token)['filename'])
        labels = load_bin_file(lidarseg_label_filename)

        labels = self.remap_labels(labels)




        sampled_idx = np.arange(points.shape[0])
        if self.phase == 'train' and self.augment_data:
            print(points.shape, self.sub_num)
            sampled_idx = self.random_sample(points)
            points = points[sampled_idx]
            colors = colors[sampled_idx]
            labels = labels[sampled_idx]

            voxel_mtx, affine_mtx = self.voxelizer.get_transformation_matrix()

            rigid_transformation = affine_mtx @ voxel_mtx
            # Apply transformations

            homo_coords = np.hstack((points, np.ones((points.shape[0], 1), dtype=points.dtype)))
            # coords = np.floor(homo_coords @ rigid_transformation.T[:, :3])
            points = homo_coords @ rigid_transformation.T[:, :3]

        if self.ignore_label is None:
            vox_ign_label = -100
        else:
            vox_ign_label = self.ignore_label

        quantized_coords, feats, labels, voxel_idx = ME.utils.sparse_quantize(points,
                                                                               colors,
                                                                               labels=labels,
                                                                               ignore_label=vox_ign_label,
                                                                               quantization_size=self.voxel_size,
                                                                               return_index=True)

        missing_pts = self.sub_num - quantized_coords.shape[0]
        if isinstance(quantized_coords, np.ndarray):
            quantized_coords = torch.from_numpy(quantized_coords)

        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats)

        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)

        if isinstance(voxel_idx, np.ndarray):
            voxel_idx = torch.from_numpy(voxel_idx)

        if sampled_idx is not None:
            sampled_idx = sampled_idx[voxel_idx]
            sampled_idx = torch.from_numpy(sampled_idx)
        else:
            sampled_idx = None

        return {"coordinates": quantized_coords,
                "features": feats,
                "labels": labels,
                "sampled_idx": sampled_idx,
                "idx": torch.tensor(i)}

    def remap_labels(self, label):
        label = label.reshape((-1))
        sem_label = label & 0xFFFF  # semantic label in lower half
        inst_label = label >> 16  # instance id in upper half
        assert ((sem_label + (inst_label << 16) == label).all())
        sem_label = self.remap_lut_val[sem_label]
        return sem_label.astype(np.int32)

    def get_dataset_weights(self):
        weights = np.zeros(self.remap_lut_val.max()+1)
        for l in tqdm.tqdm(range(len(self.label_path)), desc='Loading weights', leave=True):
            label_tmp = self.label_path[l]
            label = self.load_label_kitti(label_tmp)
            lbl, count = np.unique(label, return_counts=True)
            if self.ignore_label is not None:
                if self.ignore_label in lbl:
                    count = count[lbl != self.ignore_label]
                    lbl = lbl[lbl != self.ignore_label]

            weights[lbl] += count

        return weights

    def get_data(self, i: int):
        pcd_tmp = self.pcd_path[i]
        label_tmp = self.label_path[i]

        pcd = np.fromfile(pcd_tmp, dtype=np.float32).reshape((-1, 4))
        label = self.load_label_kitti(label_tmp)
        points = pcd[:, :3]
        if self.use_intensity:
            colors = pcd[:, 3][..., np.newaxis]
        else:
            colors = np.ones((points.shape[0], 1), dtype=np.float32)
        data = {'points': points, 'colors': colors, 'labels': label}

        points = data['points']
        colors = data['colors']
        labels = data['labels']

        return {"coordinates": points,
                "features": colors,
                "labels": labels,
                "idx": i}







if __name__ == "__main__" :
    ds = NuScenesDataset(version='full', 
                         phase='train', 
                         dataset_path='../../../CMMD/datasets/NuScenes/', 
                         mapping_path='_resources/semantic-kitti.yaml', 
                         weights_path=None, voxel_size=0.1, 
                         use_intensity=False, 
                         augment_data=False, 
                         sub_num=50000, device=None, 
                         num_classes=7, ignore_label=-1)