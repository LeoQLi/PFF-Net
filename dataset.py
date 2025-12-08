import os
import sys
import math
import random
import numbers
import numpy as np
from tqdm.auto import tqdm
import scipy.spatial as spatial
import torch
from torch.utils.data import Dataset
import heapq
# import networkx as nx


def load_data(filedir, filename, dtype=np.float32, wo=False):
    filepath = os.path.join(filedir, 'npy', filename + '.npy')
    os.makedirs(os.path.join(filedir, 'npy'), exist_ok=True)
    if os.path.exists(filepath):
        if wo:
            return True
        data = np.load(filepath)
    else:
        data = np.loadtxt(os.path.join(filedir, filename), dtype=dtype)
        np.save(filepath, data)
    return data


def spherical_sample(num):
    x = np.random.randn(num, 3)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def limited_dijkstra(G, source, limit, weight='weight'):
    visited = set()
    distances = {source: 0}
    pq = [(0, source)]
    nodes_in_order = []

    while pq and len(visited) < limit:
        current_distance, current_node = heapq.heappop(pq)
        if current_node in visited:
            continue

        visited.add(current_node)
        nodes_in_order.append((current_node, current_distance))

        for neighbor in G.neighbors(current_node):
            distance = G[current_node][neighbor].get(weight, 1)
            new_distance = current_distance + distance
            if neighbor not in visited and (neighbor not in distances or new_distance < distances[neighbor]):
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))

    nodes_in_order.sort(key=lambda x: x[1])

    return np.array([node for node, dist in nodes_in_order])


class RandomRotate(object):
    def __init__(self, degrees=180.0, axis=0):
        # if isinstance(degrees, numbers.Number):
        #     degrees = (-abs(degrees), abs(degrees))
        # assert isinstance(degrees, (tuple, list)) and len(degrees) == 2
        # self.degrees = degrees
        # self.axis = axis
        self.degrees = [0, 180]
        self.axis = [0, 1, 2]

    def __call__(self, data):
        # degree = math.pi * random.uniform(*self.degrees) / 180.0
        degree = random.choice(self.degrees)
        axis = random.choice(self.axis)
        sin, cos = math.sin(degree), math.cos(degree)

        if axis == 0:
            matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
        elif axis == 1:
            matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
        else:
            matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        matrix = torch.tensor(matrix)

        data['pcl_pat'] = torch.matmul(data['pcl_pat'], matrix)
        data['rand_trans'] = matrix

        if 'normal_center' in data:
            data['normal_center'] = torch.matmul(data['normal_center'], matrix)
        if 'normal_pat' in data:
            data['normal_pat'] = torch.matmul(data['normal_pat'], matrix)
        if 'query_vectors' in data:
            data['query_vectors'] = torch.matmul(data['query_vectors'], matrix)
        if 'pcl_sample' in data:
            data['pcl_sample'] = torch.matmul(data['pcl_sample'], matrix)
        if 'normal_sample' in data:
            data['normal_sample'] = torch.matmul(data['normal_sample'], matrix)
        return data


class PCATrans(object):
    def __init__(self):
        super().__init__()

    def pca_trans(self, pts):
        # compute PCA of points in the patch, center the patch around the mean
        pts_mean = pts.mean(0)
        pts = pts - pts_mean

        trans, _, _ = torch.svd(torch.t(pts[:pts.shape[0]//2]))  # (3, 3)   # TODO
        # trans, _, _ = torch.svd(torch.t(pts))  # (3, 3)
        pts = torch.mm(pts, trans)

        # since the patch was originally centered, the original cp was at (0,0,0)
        cp_new = -pts_mean
        cp_new = torch.matmul(cp_new, trans)

        # re-center on original center point
        pts_new = pts - cp_new
        return pts_new, trans

    def __call__(self, data):
        pts = data['pcl_pat']
        pts_new, trans = self.pca_trans(pts)
        data['pcl_pat'] = pts_new
        data['pca_trans'] = trans

        if 'normal_center' in data:
            data['normal_center'] = torch.matmul(data['normal_center'], trans)
        if 'normal_pat' in data:
            data['normal_pat'] = torch.matmul(data['normal_pat'], trans)
        if 'query_vectors' in data:
            data['query_vectors'] = torch.matmul(data['query_vectors'], trans)
        if 'pcl_sample' in data:
            data['pcl_sample'] = torch.matmul(data['pcl_sample'], trans)
        if 'normal_sample' in data:
            data['normal_sample'] = torch.matmul(data['normal_sample'], trans)
        return data


class SequentialPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.total_patch_count = sum(data_source.datasets.shape_patch_count)

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


class RandomPointcloudPatchSampler(torch.utils.data.sampler.Sampler):
    # Randomly get subset data from the whole dataset
    def __init__(self, data_source, patches_per_shape, seed=None, identical_epochs=False):
        self.data_source = data_source
        self.patches_per_shape = patches_per_shape
        self.seed = seed
        self.identical_epochs = identical_epochs

        if self.seed is None:
            self.seed = np.random.random_integers(0, 2**32-1, 1)[0]
        self.rng = np.random.RandomState(self.seed)

        self.total_patch_count = 0
        for shape_ind, _ in enumerate(data_source.datasets.shape_names):
            self.total_patch_count += min(self.patches_per_shape, data_source.datasets.shape_patch_count[shape_ind])

    def __iter__(self):
        # optionally always pick the same permutation (mainly for debugging)
        if self.identical_epochs:
            self.rng.seed(self.seed)

        return iter(self.rng.choice(sum(self.data_source.datasets.shape_patch_count), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count


class AddUniformBallNoise_Normal(object):
    def __init__(self, scale=0.0, num=0):
        super().__init__()
        self.scale = scale
        self.num = num
        self.theta_max = np.pi / 2

    def apply(self, v):
        ### rotation vector
        angle = np.random.normal(loc=0.0, scale=self.scale * self.theta_max, size=(self.num, 1))
        angle = np.clip(angle, a_min=-self.theta_max, a_max=self.theta_max)
        # angle = np.random.uniform(-self.scale * self.theta_max, self.scale * self.theta_max, size=(self.num, 1))

        rot_axis = np.random.randn(self.num, 3)
        rot_axis /= np.linalg.norm(rot_axis, axis=-1, keepdims=True)

        rot = spatial.transform.Rotation.from_rotvec(angle * rot_axis)
        v = rot.apply(v)
        return v

    def __call__(self, v):
        return self.apply(v)


class PointCloudDataset(Dataset):
    def __init__(self, root, mode=None, data_set='', data_list='', sparse_patch=False):
        super().__init__()
        self.mode = mode
        self.data_set = data_set
        self.sparse_patch = sparse_patch
        self.data_dir = os.path.join(root, data_set)

        self.pointclouds = []
        self.shape_names = []
        self.normals = []
        self.pidxs = []
        self.graphs = []
        self.kdtrees = []
        self.kdtrees_clean = []
        self.shape_patch_count = []   # point number of each shape
        assert self.mode in ['train', 'val', 'test']

        # get all shape names
        if len(data_list) > 0:
            cur_sets = []
            with open(os.path.join(root, data_set, 'list', data_list + '.txt')) as f:
                cur_sets = f.readlines()
            cur_sets = [x.strip() for x in cur_sets]
            cur_sets = list(filter(None, cur_sets))
        else:
            raise ValueError('Data list need to be given.')

        print('Current %s dataset:' % self.mode)
        for s in cur_sets:
            print('   ', s)

        self.get_data(cur_sets)
        self.cur_sets = cur_sets

    def __len__(self):
        return len(self.pointclouds)

    def get_data(self, cur_sets):
        for s in tqdm(cur_sets, desc='Loading data'):
            pcl = load_data(filedir=self.data_dir, filename='%s.xyz' % s, dtype=np.float32)[:, :3]

            if os.path.exists(os.path.join(self.data_dir, s + '.normals')):
                nor = load_data(filedir=self.data_dir, filename=s + '.normals', dtype=np.float32)
            else:
                nor = np.zeros_like(pcl)

            self.pointclouds.append(pcl)
            self.normals.append(nor)
            self.shape_names.append(s)

            # KDTree construction may run out of recursions
            sys.setrecursionlimit(int(max(1000, round(pcl.shape[0]/10))))
            kdtree = spatial.cKDTree(pcl, 10)
            self.kdtrees.append(kdtree)

            # Build graph
            # graph = nx.Graph()
            # for i in range(pcl.shape[0]):
            #     distances, indices = kdtree.query(pcl[i, :], k=50)  # include the point itself
            #     for j, dist in zip(indices[1:], distances[1:]):     # skip the first one (the point itself)
            #         graph.add_edge(i, j, weight=dist)
            # self.graphs.append(graph)

            # s_clean = s.split('_noise_')[0]
            # pcl_clean = load_data(filedir=self.data_dir, filename='%s.xyz' % s_clean, dtype=np.float32)[:, :3]
            # kdtree_clean = spatial.cKDTree(pcl_clean, 10)
            # self.kdtrees_clean.append(kdtree_clean)

            if self.sparse_patch:
                pidx = load_data(filedir=self.data_dir, filename='%s.pidx' % s, dtype=np.int32)
                self.pidxs.append(pidx)
                self.shape_patch_count.append(len(pidx))
            else:
                self.shape_patch_count.append(pcl.shape[0])

    def __getitem__(self, idx):
        # kdtree uses a reference, not a copy of these points,
        # so modifying the points would make the kdtree give incorrect results!
        data = {
            'pcl': self.pointclouds[idx].copy(),
            'kdtree': self.kdtrees[idx],
            'kdtree_clean': self.kdtrees_clean[idx] if len(self.kdtrees_clean) > 0 else None,
            # 'graph': self.graphs[idx],
            'normal': self.normals[idx],
            'pidx': self.pidxs[idx] if len(self.pidxs) > 0 else None,
            'name': self.shape_names[idx],
        }
        return data


class PatchDataset(Dataset):
    def __init__(self, datasets, patch_size=1, list_trans=['PCA', 'Random'], sample_size=0, seed=None):
        super().__init__()
        self.datasets = datasets
        self.patch_size = patch_size
        self.trans_pca = None
        self.trans_rand = None
        if 'PCA' in list_trans:
            self.trans_pca = PCATrans()
        if 'Random' in list_trans:
            self.trans_rand = RandomRotate()

        self.sample_size = sample_size
        self.rng_global_sample = np.random.RandomState(seed)

        # self.train_angle = True
        # if self.train_angle:
        #     self.num_sample = 5000
        #     self.num_query = self.num_sample // 10
        #     self.sphere_vectors = spherical_sample(self.num_sample)
        # else:
        #     self.add_noise = AddUniformBallNoise_Normal(scale=0.4, num=4000)

    def __len__(self):
        return sum(self.datasets.shape_patch_count)

    def shape_index(self, index):
        """
            Translate global (dataset-wide) point index to shape index & local (shape-wide) point index
        """
        shape_patch_offset = 0
        shape_ind = None
        for shape_ind, shape_patch_count in enumerate(self.datasets.shape_patch_count):
            if index >= shape_patch_offset and index < shape_patch_offset + shape_patch_count:
                shape_patch_ind = index - shape_patch_offset  # index in shape with ID shape_ind
                break
            shape_patch_offset = shape_patch_offset + shape_patch_count
        return shape_ind, shape_patch_ind

    def make_patch(self, pcl, kdtree, kdtree_clean=None, graph=None, nor=None, query_idx=None, patch_size=1):
        """
        Args:
            pcl: (N, 3)
            kdtree:
            nor: (N, 3)
            query_idx: (P,)
            patch_size: K
        Returns:
            pcl_pat, nor_pat: (P, K, 3)
        """
        seed_pnts = pcl[query_idx, :]
        dists, pat_idx = kdtree.query(seed_pnts, k=patch_size)  # sorted by distance (nearest first)
        dist_max = max(dists)

        if graph is not None:
            idx_crop = limited_dijkstra(graph, query_idx, patch_size)
            if idx_crop.shape[0] == patch_size:
                pat_idx = idx_crop

        pcl_pat = pcl[pat_idx, :]        # (K, 3)
        pcl_pat = pcl_pat - seed_pnts    # center
        pcl_pat = pcl_pat / dist_max     # normlize

        nor_pat = None
        nor_cen = None
        if nor is not None:
            if kdtree_clean is not None:
                _, pat_idx = kdtree_clean.query(pcl[pat_idx, :])
                query_idx = pat_idx[0]

            nor_pat = nor[pat_idx, :]
            nor_cen = nor[query_idx, :]
        return pcl_pat, nor_pat, nor_cen

    def get_subsample(self, pts, query_idx, sample_size, pts_1=None, rng=None, fixed=False, uniform=False):
        """
            pts: (N, 3)
            query_idx: (1,)
            Warning: the query point may not be included in the output point cloud !
        """
        N_pts = pts.shape[0]
        query_point = pts[query_idx, :]

        ### if there are too much points, it is not helpful for orienting normal.
        # N_max = sample_size * 50   # TODO
        # if N_pts > N_max:
        #     point_idx = np.random.choice(N_pts, N_max, replace=False)
        #     # if query_idx not in point_idx:
        #     #     point_idx[0] = query_idx
        #     #     query_idx = 0
        #     pts = pts[point_idx, :]
        #     if pts_1 is not None:
        #         pts_1 = pts_1[point_idx, :]
        #     N_pts = N_max

        pts = pts - query_point
        dist = np.linalg.norm(pts, axis=1)
        dist_max = np.max(dist)
        pts = pts / dist_max

        if pts_1 is not None:
            pts_1 = pts_1 - query_point
            pts_1 = pts_1 / dist_max

        if N_pts >= sample_size:
            if fixed:
                rng.seed(42)

            if uniform:
                sub_ids = rng.randint(low=0, high=N_pts, size=sample_size)
            else:
                dist_normalized = dist / dist_max
                prob = 1.0 - 1.5 * dist_normalized
                prob_clipped = np.clip(prob, 0.05, 1.0)

                ids = rng.choice(N_pts, size=int(sample_size / 1.5), replace=False)
                prob_clipped[ids] = 1.0
                prob = prob_clipped / np.sum(prob_clipped)
                sub_ids = rng.choice(N_pts, size=sample_size, replace=False, p=prob)

            # Let the query point be included
            if query_idx not in sub_ids:
                sub_ids[0] = query_idx
            pts_sub = pts[sub_ids, :]
            # id_new = np.argsort(dist[sub_ids])
            # pts_sub = pts_sub[id_new, :]
        else:
            pts_shuffled = pts[:, :3]
            rng.shuffle(pts_shuffled)
            zeros_padding = np.zeros((sample_size - N_pts, 3), dtype=np.float32)
            pts_sub = np.concatenate((pts_shuffled, zeros_padding), axis=0)

        # pts_sub[0, :] = 0    # TODO
        if pts_1 is not None:
            return pts_sub, pts_1[sub_ids, :]
        return pts_sub, sub_ids

    def compute_angle_offset(self, query_vector, gt_normal, eps=1e-6):
        norm = np.linalg.norm(np.cross(query_vector, gt_normal), axis=1)
        norm[(norm < eps) & (norm > -eps)] = 0.0
        norm[norm > 1.0] = 1.0
        norm[norm < -1.0] = -1.0
        return np.arcsin(norm)

    def __getitem__(self, idx):
        """
            Returns a patch centered at the point with the given global index
            and the ground truth normal of the patch center
        """
        ### find shape that contains the point with given global index
        shape_idx, patch_idx = self.shape_index(idx)
        shape_data = self.datasets[shape_idx]

        ### get the center point
        if shape_data['pidx'] is None:
            query_idx = patch_idx
        else:
            query_idx = shape_data['pidx'][patch_idx]

        pcl_pat, normal_pat, normal_center = self.make_patch(pcl=shape_data['pcl'],
                                                kdtree=shape_data['kdtree'],
                                                kdtree_clean=shape_data['kdtree_clean'],
                                                # graph=shape_data['graph'],
                                                nor=shape_data['normal'],
                                                query_idx=query_idx,
                                                patch_size=self.patch_size,
                                            )
        data = {'name': shape_data['name'],
                'pcl_pat': torch.from_numpy(pcl_pat).float(),
                'normal_pat': torch.from_numpy(normal_pat).float(),
                'normal_center': torch.from_numpy(normal_center).float(),
            }

        if self.sample_size > 0:
            pcl_sample, sample_ids = self.get_subsample(pts=shape_data['pcl'],
                                            query_idx=query_idx,
                                            sample_size=self.sample_size,
                                            rng=self.rng_global_sample,
                                            uniform=False,
                                        )
            data['pcl_sample'] = torch.from_numpy(pcl_sample).float()
            # data['normal_sample'] = torch.from_numpy(shape_data['normal'][sample_ids, :]).float()

        # if self.train_angle:
        #     sample_ind = self.rng_global_sample.choice(self.num_sample, size=self.num_query, replace=False)
        #     query_vectors = self.sphere_vectors[sample_ind, :]
        #     angle_offsets = self.compute_angle_offset(query_vectors, normal_center)
        #     data['query_vectors'] = torch.from_numpy(query_vectors).float()
        #     data['angle_offsets'] = torch.from_numpy(angle_offsets).float()
        # else:
        #     query_vectors = self.add_noise(normal_center)
        #     data['query_vectors'] = torch.from_numpy(query_vectors).float()

        if self.trans_pca is not None:
            data = self.trans_pca(data)
        if self.trans_rand is not None:
            data = self.trans_rand(data)
        return data



