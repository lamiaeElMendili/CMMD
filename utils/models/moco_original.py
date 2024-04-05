import numpy as np
import MinkowskiEngine as ME
import torch

import numpy as np
import open3d as o3d
import torch.nn as nn

from utils.models.minkunet import MinkUNet34
import time

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
    print(ckpt.keys())
    
    model.load_state_dict(ckpt, strict=True)
    return model

def overlap_clusters(cluster_i, cluster_j, min_cluster_point=20):
    # get unique labels from pcd_i and pcd_j
    unique_i = np.unique(cluster_i)
    unique_j = np.unique(cluster_j)

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]

    # also remove clusters with few points
    for cluster in unique_ij.copy():
        ind_i = np.where(cluster_i == cluster)
        ind_j = np.where(cluster_j == cluster)

        if len(ind_i[0]) < min_cluster_point or len(ind_j[0]) < min_cluster_point:
            unique_ij = np.delete(unique_ij, unique_ij == cluster)
        
    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j

def clusters_hdbscan(points_set, n_clusters):
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1., approx_min_span_tree=True,
                                gen_min_span_tree=True, leaf_size=100,
                                metric='euclidean', min_cluster_size=20, min_samples=None
                            )

    clusterer.fit(points_set)

    labels = clusterer.labels_.copy()

    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusters_from_pcd(pcd, n_clusters):
    # clusterize pcd points
    labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10))
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[1:], counts[1:])))
    cluster_info = cluster_info[cluster_info[:,1].argsort()]

    clusters_labels = cluster_info[::-1][:n_clusters, 0]
    labels[np.in1d(labels, clusters_labels, invert=True)] = -1

    return labels

def clusterize_pcd(points, n_clusters):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # segment plane (ground)
    _, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
    pcd_ = pcd.select_by_index(inliers, invert=True)

    labels_ = np.expand_dims(clusters_from_pcd(pcd_, n_clusters), axis=-1)

    # that is a blessing of array handling
    # pcd are an ordered list of points
    # in a list [a, b, c, d, e] if we get the ordered indices [1, 3]
    # we will get [b, d], however if we get ~[1, 3] we will get the opposite indices
    # still ordered, i.e., [a, c, e] which means listing the inliers indices and getting
    # the invert we will get the outliers ordered indices (a sort of indirect indices mapping)
    labels = np.ones((points.shape[0], 1)) * -1
    mask = np.ones(labels.shape[0], dtype=bool)
    mask[inliers] = False

    labels[mask] = labels_

    return np.concatenate((points, labels), axis=-1)

def visualize_pcd_clusters(point_set):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

def visualize_pcd_clusters_compare(point_set, pi, pj):
    pcd_ = o3d.geometry.PointCloud()
    pcd_.points = o3d.utility.Vector3dVector(point_set[:,:3])

    pi[:,-1], pj[:,-1] = overlap_clusters(pi[:,-1], pj[:,-1])
    point_set[:,-1], pi[:,-1] = overlap_clusters(point_set[:,-1], pi[:,-1])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set[:,:3])

    labels = point_set[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])

    pcd_i = o3d.geometry.PointCloud()
    pcd_i.points = o3d.utility.Vector3dVector(pi[:,:3])

    labels = pi[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_i.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_i])
    pcd_i.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_i])

    pcd_j = o3d.geometry.PointCloud()
    pcd_j.points = o3d.utility.Vector3dVector(pj[:,:3])

    labels = pj[:, -1]
    import matplotlib.pyplot as plt
    colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    colors[labels < 0] = 0

    pcd_j.colors = o3d.utility.Vector3dVector(np.zeros_like(colors[:, :3]))
    o3d.visualization.draw_geometries([pcd_j])
    pcd_j.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd_j])

    # pcd_2 = o3d.geometry.PointCloud()
    # point_set_[:,2] += 10.
    # pcd_2.points = o3d.utility.Vector3dVector(point_set_[:,:3])

    # labels = point_set_[:, -1]
    # import matplotlib.pyplot as plt
    # colors = plt.get_cmap("prism")(labels / (labels.max() if labels.max() > 0 else 1))
    # colors[labels < 0] = 0

    # pcd_2.colors = o3d.utility.Vector3dVector(colors[:, :3])

    o3d.visualization.draw_geometries([pcd])
    o3d.visualization.draw_geometries([pcd_i])
    o3d.visualization.draw_geometries([pcd_j])
    #return pcd_


def array_to_sequence(batch_data):
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]


def list_segments_points(p_coord, p_feats, labels, classes, batch_size=1):

    c_coord = []
    c_feats = []
    c_labels = []

    seg_batch_count = 0

    # Iterate over unique segment labels
    for segment_lbl in torch.unique(labels):
        segment_lbl = segment_lbl.item()
        if segment_lbl not in classes:
            continue

        # Filter coordinates, features, and labels for the current segment label
        segment_ind = labels == segment_lbl
        segment_coord = p_coord[segment_ind]
        segment_feats = p_feats[segment_ind]

        # Update batch index in coordinates
        segment_coord[:, 0] = seg_batch_count
        seg_batch_count += 1

        # Append to lists
        c_coord.append(segment_coord)
        c_feats.append(segment_feats)
        c_labels.append(segment_lbl)

    # Concatenate coordinate and feature tensors
    seg_coord = torch.vstack(c_coord)
    seg_feats = torch.vstack(c_feats)
    
    # Convert labels to tensor
    c_labels = torch.tensor(c_labels).cuda()

    # Create SparseTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_tensor = ME.SparseTensor(features=seg_feats, coordinates=seg_coord, device=device)

    return sparse_tensor, c_labels


def numpy_to_sparse_tensor(p_coord, p_feats, p_label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    if p_label is not None:
        p_label = ME.utils.batched_coordinates(array_to_torch_sequence(p_label), dtype=torch.float32)[:, 1:]
    
        return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            ), p_label.cuda()

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

def point_set_to_coord_feats(point_set, labels, resolution, num_points, deterministic=False):
    p_feats = point_set.copy()
    p_coord = np.round(point_set[:, :3] / resolution)
    p_coord -= p_coord.min(0, keepdims=1)

    _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)
    if len(mapping) > num_points:
        if deterministic:
            # for reproducibility we set the seed
            np.random.seed(42)
        mapping = np.random.choice(mapping, num_points, replace=False)

    return p_coord[mapping], p_feats[mapping], labels[mapping]

def collate_points_to_sparse_tensor(pi_coord, pi_feats, pj_coord, pj_feats):
    # voxelize on a sparse tensor
    points_i = numpy_to_sparse_tensor(pi_coord, pi_feats)
    points_j = numpy_to_sparse_tensor(pj_coord, pj_feats)

    return points_i, points_j


class SparseAugmentedCollation:
    def __init__(self, resolution, num_points=80000, segment_contrast=False):
        self.resolution = resolution
        self.num_points = num_points
        self.segment_contrast = segment_contrast

    def __call__(self, list_data):
        points_i, points_j = list(zip(*list_data))

        points_i = np.asarray(points_i)
        points_j = np.asarray(points_j)

        pi_feats = []
        pi_coord = []
        pi_cluster = []

        pj_feats = []
        pj_coord = []
        pj_cluster = []

        for pi, pj in zip(points_i, points_j):
            # pi[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pi, feats_pi, cluster_pi = point_set_to_coord_feats(pi[:,:-1], pi[:,-1], self.resolution, self.num_points)
            pi_coord.append(coord_pi)
            pi_feats.append(feats_pi)

            # pj[:,:-1] will be the points and intensity values, and the labels will be the cluster ids
            coord_pj, feats_pj, cluster_pj = point_set_to_coord_feats(pj[:,:-1], pj[:,-1], self.resolution, self.num_points)
            pj_coord.append(coord_pj)
            pj_feats.append(feats_pj)

            cluster_pi, cluster_pj = overlap_clusters(cluster_pi, cluster_pj)

            if self.segment_contrast:
                # we store the segment labels per point
                pi_cluster.append(cluster_pi)
                pj_cluster.append(cluster_pj)

        pi_feats = np.asarray(pi_feats)
        pi_coord = np.asarray(pi_coord)

        pj_feats = np.asarray(pj_feats)
        pj_coord = np.asarray(pj_coord)

        segment_i = np.asarray(pi_cluster)
        segment_j = np.asarray(pj_cluster)

        # if not segment_contrast segment_i and segment_j will be an empty list
        return (pi_coord, pi_feats, segment_i), (pj_coord, pj_feats, segment_j)

class SparseCollation:
    def __init__(self, resolution, num_points=80000):
        self.resolution = resolution
        self.num_points = num_points

    def __call__(self, list_data):
        points_set, labels = list(zip(*list_data))

        points_set = np.asarray(points_set)
        labels = np.asarray(labels)

        p_feats = []
        p_coord = []
        p_label = []
        for points, label in zip(points_set, labels):
            coord, feats, label_ = point_set_to_coord_feats(points, label, self.resolution, self.num_points, True)
            p_feats.append(feats)
            p_coord.append(coord)
            p_label.append(label_)

        p_feats = np.asarray(p_feats)
        p_coord = np.asarray(p_coord)
        p_label = np.asarray(p_label)

        # if we directly map coords and feats to SparseTensor it will loose the map over the coordinates
        # if the mapping between point and voxels are necessary, please use TensorField
        # as in https://nvidia.github.io/MinkowskiEngine/demo/segmentation.html?highlight=segmentation
        # we first create TensorFields and from it we create the sparse tensors, so we can map the coordinate
        # features across different SparseTensors, i.e. output prediction and target labels

        return p_coord, p_feats, p_label






latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}

class MoCo(nn.Module):
    def __init__(self, model_head, K=65536, m=0.99999, T=1.0, config=None):
        super(MoCo, self).__init__()

        
        self.config=config
        self.K = self.config.adaptation.cmmd.queue_size
        self.m = m
        self.T = T
        self.batch_size = self.config.pipeline.dataloader.train_batch_size

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


        d = self.config.adaptation.cmmd.feature_dimension
        self.normalization = self.config.adaptation.cmmd.normalization

        self.model_q = student_model
        self.model_k = teacher_model


        self.head_q = model_head(96, d)#.type(dtype)

        #self.head_k = model_head(96, d)#.type(dtype)

        # initialize model k and q
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # initialize headection head k and q
        #for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
        #    param_k.data.copy_(param_q.data)
        #    param_k.requires_grad = False

        self.register_buffer('queue_pcd', torch.randn(d,self.K))
        if self.normalization:
            self.queue_pcd = nn.functional.normalize(self.queue_pcd, dim=0)

        self.register_buffer('queue_seg', torch.randn(d,self.K))
        if self.normalization:
            self.queue_seg = nn.functional.normalize(self.queue_seg, dim=0)
    
        self.register_buffer('k_labels', torch.zeros(self.K, dtype=torch.long))


        self.register_buffer("queue_pcd_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("queue_seg_ptr", torch.zeros(1, dtype=torch.long))

        if torch.cuda.device_count() > 1:
            self.model_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_q)
            self.head_q = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_q)

            self.model_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.model_k)
            self.head_k = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.head_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.model_q.parameters(), self.model_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        #for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
        #    param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue_pcd(self, keys):
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_pcd_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if ptr + batch_size <= self.K:
            self.queue_pcd[:, ptr:ptr + batch_size] = keys.T
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue_pcd[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue_pcd[:, :head_size] = keys.T[:, tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_pcd_ptr[0] = ptr

    def _dequeue_and_enqueue_seg(self, keys, k_labels):
        # gather keys before updating queue
        if torch.cuda.device_count() > 1:
            # similar to shuffling, since for each gpu the number of segments may not be the same
            # we create a aux variable keys_gather of size (1, MAX_SEG_BATCH, 128)
            # add the current seg batch to [0,:CURR_SEG_BATCH, 128] gather them all in
            # [NUM_GPUS,MAX_SEG_BATCH,128] and concatenate only the filled seg batches
            seg_size = torch.from_numpy(np.array([keys.shape[0]])).cuda()
            all_seg_size = concat_all_gather(seg_size)

            keys_gather = torch.ones((1, all_seg_size.max(), keys.shape[-1])).cuda()
            keys_gather[0, :keys.shape[0],:] = keys[:,:]

            all_keys = concat_all_gather(keys_gather)
            gather_keys = None

            for k in range(len(all_seg_size)):
                if gather_keys is None:
                    gather_keys = all_keys[k][:all_seg_size[k],:]
                else:
                    gather_keys = torch.cat((gather_keys, all_keys[k][:all_seg_size[k],:]))


            keys = gather_keys

        batch_size = keys.shape[0]

        ptr = int(self.queue_seg_ptr)
        #assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)


        #k_labels = torch.unique(k_labels)

        if ptr + batch_size <= self.K:
            self.queue_seg[:, ptr:ptr + batch_size] = keys.T
            self.k_labels[ptr:ptr + batch_size] = k_labels
        else:
            tail_size = self.K - ptr
            head_size = batch_size - tail_size
            self.queue_seg[:, ptr:self.K] = keys.T[:, :tail_size]
            self.queue_seg[:, :head_size] = keys.T[:, tail_size:]

            self.k_labels[ptr:self.K] = k_labels[:tail_size]
            self.k_labels[:head_size] = k_labels[tail_size:]

        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_seg_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features

        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))
        for bidx in newx:
            batch_size.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(batch_size).cuda())
        max_size = torch.max(all_size)

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = []
        shuffle_f = []
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda())
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx)

        shuffle_c = torch.stack(shuffle_c)
        shuffle_f = torch.stack(shuffle_f)

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c)
        f_gather = concat_all_gather(shuffle_f)

        batch_size_all = c_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        c_this = []
        f_this = []
        batch_id = []

        # after shuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final shuffled coordinates and features, build back the sparse tensor
        c_this = np.array(c_this)
        f_this = np.array(f_this)
        x_this = numpy_to_sparse_tensor(c_this, f_this)

        return x_this, idx_unshuffle


    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size = []

        # sparse tensor should be decomposed
        c, f = x.decomposed_coordinates_and_features

        # each pcd has different size, get the biggest size as default
        newx = list(zip(c, f))
        for bidx in newx:
            batch_size.append(len(bidx[0]))
        all_size = concat_all_gather(torch.tensor(batch_size).cuda())
        max_size = torch.max(all_size)

        # create a tensor with shape (batch_size, max_size)
        # copy each sparse tensor data to the begining of the biggest sized tensor
        shuffle_c = []
        shuffle_f = []
        for bidx in range(len(newx)):
            shuffle_c.append(torch.ones((max_size, newx[bidx][0].shape[-1])).cuda())
            shuffle_c[bidx][:len(newx[bidx][0]),:] = newx[bidx][0]

            shuffle_f.append(torch.ones((max_size, newx[bidx][1].shape[-1])).cuda())
            shuffle_f[bidx][:len(newx[bidx][1]),:] = newx[bidx][1]

        batch_size_this = len(newx)

        shuffle_c = torch.stack(shuffle_c)
        shuffle_f = torch.stack(shuffle_f)

        # gather all the ddp batches pcds
        c_gather = concat_all_gather(shuffle_c)
        f_gather = concat_all_gather(shuffle_f)

        batch_size_all = c_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        c_this = []
        f_this = []
        batch_id = []

        # after unshuffling we get only the actual information of each tensor
        # :actual_size is the information, actual_size:biggest_size are just ones (ignore)
        for idx in range(len(idx_this)):
            c_this.append(c_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())
            f_this.append(f_gather[idx_this[idx]][:all_size[idx_this[idx]],:].cpu().numpy())

        # final unshuffled coordinates and features, build back the sparse tensor
        c_this = np.array(c_this)
        f_this = np.array(f_this)
        x_this = numpy_to_sparse_tensor(c_this, f_this)

        return x_this

    def forward(self, source_stensor, source_labels, target_stensor, target_pseudo, mixed_tensor, step=None):


        source_labels_filtered = source_labels[source_labels != -1]
        target_pseudo_filtered = target_pseudo[target_pseudo != -1]

        classes = np.intersect1d(source_labels_filtered.cpu().numpy(), target_pseudo_filtered.cpu().numpy())






        if self.config.adaptation.cmmd.query == 'target student' :   
            h_q = self.model_q(target_stensor, is_seg=False)  # queries: NxC

            h_qs, q_labels = list_segments_points(h_q.C, h_q.F, target_pseudo, classes, self.batch_size)   

        z_qs = self.head_q(h_qs)

        
        if self.normalization :
            q_seg = nn.functional.normalize(z_qs, dim=1)
        else :
            q_seg = z_qs




        # compute key features
        with torch.no_grad():  # no gradient to keys

            if (step+1) % self.config.adaptation.momentum.update_every == 0:
                print("updating key encoder")
                self._momentum_update_key_encoder()  # update the key encoder

            if self.config.adaptation.cmmd.key == 'source teacher queue' :

                h_k = self.model_k(source_stensor, is_seg=False)
                h_ks, k_labels = list_segments_points(h_k.C, h_k.F, source_labels, classes, self.batch_size)

            elif self.config.adaptation.cmmd.key == 'target teacher queue' :
                h_k = self.model_k(target_stensor, is_seg=False)
                h_ks = list_segments_points(h_k.C, h_k.F, target_pseudo, self.batch_size)
                k_labels = target_pseudo[target_pseudo != -1]
            
            elif self.config.adaptation.cmmd.key == 'source student queue' :
                h_k = self.model_q(source_stensor, is_seg=False)
                h_ks = list_segments_points(h_k.C, h_k.F, source_labels, self.batch_size)
                k_labels = source_labels[source_labels != -1]
            
            elif self.config.adaptation.cmmd.key == 'target student queue' :
                h_k = self.model_q(target_stensor, is_seg=False)
                h_ks = list_segments_points(h_k.C, h_k.F, target_pseudo, self.batch_size)
                k_labels = target_pseudo[target_pseudo != -1]


            z_ks = self.head_q(h_ks)
            if self.normalization :
                k_seg = nn.functional.normalize(z_ks, dim=1)
            else :
                k_seg = z_ks





        self._dequeue_and_enqueue_seg(k_seg, k_labels)



        s_out = self.model_q(source_stensor)
        t_out = self.model_q(mixed_tensor)

        #return logits_seg, labels_seg
        return q_seg, q_labels, self.queue_seg, self.k_labels, s_out, t_out

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
