import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json
import open3d as o3d
import h5py
# import pptk

from provider import train_test_split, get_example_list
from sklearn.neighbors import KNeighborsClassifier

transform_labels = {0: 0,
                        1: 1,
                        2: 1,
                        3: 2,
                        4: 3,
                        5: 4,
                        6: 5,
                        7: 6,
                        8: 5,
                        9: 6,
                        10: 7,
                        11: 7,
                        12: 6,
                        13: 8,
                        14: 9,
                        15: 10,
                        16: 11,
                        17 : 12}

def triangle_center_3d(v1, v2, v3):
    # Calculate the centroid of the triangle
    x = (v1[0] + v2[0] + v3[0]) / 3
    y = (v1[1] + v2[1] + v3[1]) / 3
    z = (v1[2] + v2[2] + v3[2]) / 3
    
    return np.asarray([x,y,z])

def scale(point: list, mean: float, maxi: float, mini: float) -> float:
    
    return np.asarray([2 * (point[0] - mean)/(maxi-mini), 2 * (point[1] - mean)/(maxi-mini), 2 * (point[2] - mean)/(maxi-mini)])

def pad_along_axis(array: np.ndarray, target_length: int, axis: int = 0) -> np.ndarray:

    pad_size = target_length - array.shape[axis]

    if pad_size <= 0:
        return array

    npad = [(0, 0)] * array.ndim
    npad[axis] = (0, pad_size)

    return np.pad(array, pad_width=npad, mode='constant', constant_values=0)


class ModelNetDataLoader(Dataset):
    def __init__(self, root, npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints,:]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel


        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)
    
    
########### MANUFACTURING FEATURES ###############
    
class FFMaachiningModels(Dataset):
    def __init__(self, examples: list, datapath = './data/own', num_points = 20000, num_classes = 16, is_normals = True) -> None:
        
        self.datapath = datapath
        self.num_points = num_points
        self.num_classes = num_classes
        self.point_datapath = f"{self.datapath}/pcd"
        self.point_datapath_normalized = f"{self.datapath}/pcd_normalized"
        self.stl_datapath =  f"{self.datapath}/own_stl"
        self.label_path = f"{self.datapath}/own_labels"
        self.examples = examples
        self.is_normals = is_normals
        
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        pcd_unprocessed = o3d.io.read_point_cloud(f"{self.point_datapath}/{self.examples[index]}.pcd")
        pcd_unprocessed = np.asarray(pcd_unprocessed.points)
        pointlabels = []
        
        pcd = o3d.io.read_point_cloud(f"{self.point_datapath_normalized}/{self.examples[index]}.pcd")
        
        with open(f"{self.label_path}/{self.examples[index]}.txt") as f:
            for line in f.readlines():
                # strs = line.split(' ')
                pointlabels.append(transform_labels[int(line)])
                
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
            
        pointlabels = np.array(pointlabels)
        
        glob_mean = np.mean(pcd_unprocessed)
        glob_max = np.max(pcd_unprocessed)
        glob_min = np.min(pcd_unprocessed)

        mesh = o3d.io.read_triangle_mesh(f"{self.stl_datapath}/{self.examples[index]}.stl")
        

        vertices = np.asarray(mesh.vertices)
        loaded_faces = np.asarray(mesh.triangles)
        centroids = [triangle_center_3d(vertices[face[0]], vertices[face[1]], vertices[face[2]]) for face in loaded_faces]
        
        scaled_centroids = [scale(centroid, glob_mean, glob_max, glob_min) for centroid in centroids]
        
        # visualize_point_cloud(points, pointlabels, 25)
        
        pointlabels = np.array(pointlabels)
        
        c1 = KNeighborsClassifier(n_neighbors=3)
        c1.fit(points, pointlabels)
        ground_truth = c1.predict(scaled_centroids)
    
        
        if self.num_points != 20000:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            points = points[choice, :]
            normals = normals[choice, :]
            pointlabels = pointlabels[choice]
            
            # ratio = int(20000/self.num_points)
            # if ratio > 1:
            #     points = points[::ratio]
            #     normals = normals[::ratio]
            #     pointlabels = pointlabels[::ratio]
          
        max_length = 500000
        length = len(ground_truth) 
        pad_length = max_length - length
        scaled_centroids = np.concatenate((np.asarray(scaled_centroids), np.zeros((pad_length, 3))), axis = 0) 
        ground_truth = np.pad(ground_truth, (0,pad_length)) 
             
        if self.is_normals:
            pointdata = np.concatenate((points, normals), axis=1)
            return pointdata, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.stl"
        else:
            return points, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.stl"
        
        
        
class FeaturenetSingle(Dataset):
    def __init__(self, examples: list, datapath = './data/featurenet', num_points = 5000, num_classes = 25, is_normals = True) -> None:
        
        self.datapath = datapath
        self.num_points = num_points
        self.num_classes = num_classes
        self.point_datapath = f"{self.datapath}/featurenet_pcd"
        self.point_datapath_normalized = f"{self.datapath}/featurenet_pcd_normalized"
        self.stl_datapath =  f"{self.datapath}/featurenet_stl_all"
        self.label_path = f"{self.datapath}/featurenet_labels"
        self.examples = examples
        self.is_normals = is_normals
        
       
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        
        # pcd_unprocessed = o3d.io.read_point_cloud(f"{self.point_datapath}/{self.examples[index]}.pcd")
        # pcd_unprocessed = np.asarray(pcd_unprocessed.points)
        pointlabels = []
        
        pcd = o3d.io.read_point_cloud(f"{self.point_datapath_normalized}/{self.examples[index]}.pcd")
        
        with open(f"{self.label_path}/{self.examples[index]}.txt") as f:
            for line in f.readlines():
                # strs = line.split(' ')
                pointlabels.append(int(line) + 1)
          
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
            
        pointlabels = np.array(pointlabels)
        
        # glob_mean = np.mean(pcd_unprocessed)
        # glob_max = np.max(pcd_unprocessed)
        # glob_min = np.min(pcd_unprocessed)

        mesh = o3d.io.read_triangle_mesh(f"{self.stl_datapath}/{self.examples[index]}.STL")
        

        vertices = np.asarray(mesh.vertices)
        loaded_faces = np.asarray(mesh.triangles)
        centroids = [triangle_center_3d(vertices[face[0]], vertices[face[1]], vertices[face[2]]) for face in loaded_faces]
        
        scaled_centroids = [scale(centroid, 5.0, 10.0, 0.0) for centroid in centroids]
        
        # visualize_point_cloud(points, pointlabels, 25)
        
        pointlabels = np.array(pointlabels)
        
        c1 = KNeighborsClassifier(n_neighbors=3)
        c1.fit(points, pointlabels)
        ground_truth = c1.predict(scaled_centroids)
        
        check_labels_p = set(pointlabels)    
        check_labels_t = set(ground_truth)
    
        
        if self.num_points != 5000:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            points = points[choice, :]
            normals = normals[choice, :]
            pointlabels = pointlabels[choice]
            
              
        max_length = 500
        length = len(ground_truth) 
        pad_length = max_length - length
        scaled_centroids = np.concatenate((np.asarray(scaled_centroids), np.zeros((pad_length, 3))), axis = 0) 
        ground_truth = np.pad(ground_truth, (0,pad_length)) 

       
        if self.is_normals:
            pointdata = np.concatenate((points, normals), axis=1)
            return pointdata, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.STL"
        else:
            return points, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.STL"
        
        
    
class FeaturenetMulti(Dataset):
    def __init__(self, examples: list, datapath = './data/multi_featurenet', num_points = 10000, num_classes = 25, is_normals = True) -> None:
        
        self.datapath = datapath
        self.num_points = num_points
        self.num_classes = num_classes
        self.point_datapath = f"{self.datapath}/multi_featurenet_pcd"
        self.point_datapath_normalized = f"{self.datapath}/multi_featurenet_pcd_processed"
        self.stl_datapath =  f"{self.datapath}/multi_featurenet_stl"
        self.label_path = f"{self.datapath}/multi_featurenet_labels"
        self.examples = examples
        self.is_normals = is_normals
        
       
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        
        # pcd_unprocessed = o3d.io.read_point_cloud(f"{self.point_datapath}/{self.examples[index]}.pcd")
        # pcd_unprocessed = np.asarray(pcd_unprocessed.points)
        pointlabels = []
        
        pcd = o3d.io.read_point_cloud(f"{self.point_datapath_normalized}/{self.examples[index]}.pcd")
        
        with open(f"{self.label_path}/{self.examples[index]}.txt") as f:
            for line in f.readlines():
                # strs = line.split(' ')
                pointlabels.append(int(line) + 1)
          
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
            
        pointlabels = np.array(pointlabels)
        
        # glob_mean = np.mean(pcd_unprocessed)
        # glob_max = np.max(pcd_unprocessed)
        # glob_min = np.min(pcd_unprocessed)

        mesh = o3d.io.read_triangle_mesh(f"{self.stl_datapath}/{self.examples[index]}.STL")

        vertices = np.asarray(mesh.vertices)
        loaded_faces = np.asarray(mesh.triangles)
        centroids = [triangle_center_3d(vertices[face[0]], vertices[face[1]], vertices[face[2]]) for face in loaded_faces]
        
        scaled_centroids = [scale(centroid, 5000.0, 10000.0, 0.0) for centroid in centroids]
                        
        
        c1 = KNeighborsClassifier(n_neighbors=1)
        c1.fit(points, pointlabels)
        ground_truth = c1.predict(scaled_centroids)
        
        check_labels_p = set(pointlabels)    
        check_labels_t = set(ground_truth)
        
        if self.num_points != 10000:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            points = points[choice, :]
            normals = normals[choice, :]
            pointlabels = pointlabels[choice]
            
           
        max_length = 1800
        length = len(ground_truth) 
        pad_length = max_length - length
        scaled_centroids = np.concatenate((np.asarray(scaled_centroids), np.zeros((pad_length, 3))), axis = 0) 
        ground_truth = np.pad(ground_truth, (0,pad_length))
        
        
        if self.is_normals:
            pointdata = np.concatenate((points, normals), axis=1)
            return pointdata, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.STL"
        else:
            return points, pointlabels, scaled_centroids, ground_truth, length, f"{self.stl_datapath}/{self.examples[index]}.STL"



    
class FFMachiningModels_hf5(Dataset):
    def __init__(self, examples: list, num_points = 20000, num_classes = 13, is_normals = True) -> None:

        self.num_points = num_points
        self.num_classes = num_classes
        self.examples = examples
        self.is_normals = is_normals
        
        self.datapath = './data/own_data.h5'
        with h5py.File(self.datapath, "r") as f:
      
            self.labels = f['labels'][examples]
            self.pcd_data = f['pcd_data'][examples]
                
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
       
        pcd_datas = self.pcd_data[index]
        pointlabels = self.labels[index]
        
        classes = list(set(pointlabels))
        class_encoded = np.zeros(self.num_classes)
        
        for ind in classes:
            class_encoded[ind] = 1  
            
        pointlabels = np.array(pointlabels)
        pcd_datas = np.array(pcd_datas)
        
        if self.num_points != 20000:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            
            pcd_datas = pcd_datas[choice, :]
            pointlabels = pointlabels[choice]
            
        
        if self.is_normals:
            
            return pcd_datas, class_encoded, pointlabels 
        else:
            return pcd_datas[:,0:3], class_encoded, pointlabels 
        
        
        
class FeaturenetSingle_hf5(Dataset):
    def __init__(self, examples: list, datapath = './data/featurenet', num_points = 5000, num_classes = 25, is_normals = True, isolated = False) -> None:
        
        # there are 24000 models for non-isolated case and 23977 for isolated
        
        self.num_points = num_points
        self.examples = examples
        self.is_normals = is_normals
        self.isolated = isolated
        
        
        if self.isolated:
            self.num_classes = 24
            self.datapath = './data/featurenet_isolated.h5'
            self.num_points = 3000

        else:
            self.num_classes = num_classes
            self.datapath = './data/featurenet_single_class.h5'
        
        
        with h5py.File(self.datapath, "r") as f:
        
            self.labels = f['labels'][examples]
            self.pcd_data = f['pcd_data'][examples]
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        pcd_datas = self.pcd_data[index]
        pointlabels = self.labels[index]
        classes = list(set(pointlabels))
        class_encoded = np.zeros(self.num_classes)
        
             
        for ind in classes:
            if ind == 81:
                class_encoded[9] = 1
                
                for i, label in enumerate(pointlabels):
                    if label == 81:
                        pointlabels[i] = 9
                
            else:
                class_encoded[ind] = 1    
            
        pointlabels = np.array(pointlabels)
        pcd_datas = np.array(pcd_datas)
        
        if self.isolated:
            limit = 3000
        else:
            limit = 5000
        
        if self.num_points != limit:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            
            pcd_datas = pcd_datas[choice, :]
            pointlabels = pointlabels[choice]
            
        
        if self.is_normals:
            
            return pcd_datas, class_encoded, pointlabels 
        else:
            return pcd_datas[:,0:3], class_encoded, pointlabels 
       
       
class FeaturenetMulti_hf5(Dataset):
    def __init__(self, examples: list, datapath = './data/featurenet', num_points = 10000, num_classes = 25, is_normals = True, mod = False, isolated = False) -> None:
        
        
        self.num_points = num_points
        self.num_classes = num_classes
        self.examples = examples
        self.is_normals = is_normals
        self.mod = mod
        self.isolated = isolated
        
        if self.mod:    
            self.datapath = './data/multi_featurenet_mod.h5'
        else:
            self.datapath = './data/multi_featurenet.h5'
            
        if self.isolated:
            self.datapath = './data/multi_featurenet_isolated.h5'
            self.num_classes = 24
            
        with h5py.File(self.datapath, "r") as f:
        
            self.labels = f['labels'][examples]
            self.pcd_data = f['pcd_data'][examples]
            
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        pcd_datas = self.pcd_data[index]
        if self.isolated:
            pointlabels = self.labels[index]
        else:
            pointlabels = self.labels[index] + 1
        classes = list(set(pointlabels))
        class_encoded = np.zeros(self.num_classes)
     
        for ind in classes:
            class_encoded[ind] = 1  
            
        pointlabels = np.array(pointlabels)
        pcd_datas = np.array(pcd_datas)
        
        if self.isolated:
            limit = 3000
        else:
            limit = 10000
        
        if self.num_points != limit:
            
            choice = np.random.choice(len(pointlabels), self.num_points, replace=False)
            # resample
            
            pcd_datas = pcd_datas[choice, :]
            pointlabels = pointlabels[choice]
            
        
        if self.is_normals:
            
            return pcd_datas, class_encoded, pointlabels 
        else:
            return pcd_datas[:,0:3], class_encoded, pointlabels 
       

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
            
    
if __name__ == '__main__':
    
    mesh = o3d.io.read_triangle_mesh("./data/own/own_stl/202.stl")

    
    
    examples_single = get_example_list('./data/featurenet/featurenet_labels', f5=False)
    examples_multi = get_example_list('./data/multi_featurenet/multi_featurenet_labels', f5=False)
    examples_own = get_example_list('./data/own/own_labels', f5=False)
    train_single, test_single = train_test_split(examples_single)
    train_multi, test_multi = train_test_split(examples_multi)
    train_own, test_own = train_test_split(examples_own)    
        
        
    # data = FeaturenetSingle(test_single, num_points=1024)
    # Dataloader = torch.utils.data.DataLoader(data, batch_size=8, shuffle=True)
    
    data = FeaturenetMulti(test_multi, num_points=4096)
    Dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    
    # data = FFMaachiningModels(test_own, num_points=2048)
    # Dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    
    
    for pointdata, pointlabels, centroids, ground_truth, length, verteces, faces, example in Dataloader:
        print(example)
        for truth, l in zip(ground_truth, length):
            print(len(truth[:l]))
        continue
        check = True

    # for _, classes, seg in Dataloader:
        
        # print(points.shape)
        # print(classes)
        # print(seg)
        
        # new_classes = torch.unsqueeze(classes, 1)
        
        # print(new_classes.shape)
        
        # print(torch.unsqueeze(classes, 1).repeat(1, points.shape[1], 1))
        
        
        
        # # v = pptk.viewer(points[0][:,0:3])
        
        # # w = pptk.viewer(points2[0])
        
        # break
        
        
    exit()
    
    
    avg_nonzero = []
    examples = get_example_list('',num_examples = 23977, f5 = True)
    train, test = train_test_split(examples)
    
    # data = FeaturenetMulti_hf5(examples, num_points=3000, isolated=True)
    data = FeaturenetSingle_hf5(examples, num_points=3000, isolated = True)
    # data = FFMachiningModels_hf5(examples, num_classes=18)

    # data = FeaturenetMulti_hf5(test)
    Dataloader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=True)
    count = 0
    
    examples_per_class = [[] for i in range(25)]

    for points, classes, seg in Dataloader:
        
        
        
        for point_labels in seg:
            
            labels = torch.unique(point_labels).tolist()
            
            for label in labels:

                mask = point_labels == label
                num_labels = point_labels[mask]

                examples_per_class[label].append(len(num_labels))
            
    #     print(points)
    #     # print(classes)
    #     print(seg.shape)
        
    #     # for label in seg[0]:
    #     #     if label != 0:
    #     #         count += 1
                
    #     # print(count)
        
        
    #     # new_classes = torch.unsqueeze(classes, 1)
        
    #     # print(new_classes.shape)
        
    #     # print(torch.unsqueeze(classes, 1).repeat(1, points.shape[1], 1))
        
    #     # print('1')
        
    #     # v = pptk.viewer(points[0][:,0:3])
        
    #     # w = pptk.viewer(points2[0])
        
        # print(f'{count}',end='/r')
    #     count+=1
    #     exit()
    for i, class_ex in enumerate(examples_per_class):
        if len(class_ex) != 0:
            print(i, len(class_ex), np.max(class_ex), np.min(class_ex), np.mean(class_ex), np.median(class_ex), np.sum(class_ex))    
    # print(np.max(avg_nonzero), np.min(avg_nonzero), np.mean(avg_nonzero), np.median(avg_nonzero))