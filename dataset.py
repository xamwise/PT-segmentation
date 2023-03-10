import numpy as np
import os
from torch.utils.data import Dataset
import torch
from pointnet_util import farthest_point_sample, pc_normalize
import json
import open3d as o3d
# import pptk

from provider import train_test_split, get_example_list


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
    
    
    
    
class FFMaachiningModels(Dataset):
    def __init__(self, examples: list, datapath = './data', num_points = 20000, num_classes = 16, is_normals = True) -> None:
        
        self.datapath = datapath
        self.num_points = num_points
        self.num_classes = num_classes
        self.point_datapath = f"{self.datapath}/clouds"
        self.stl_datapath =  f"{self.datapath}/all_stl"
        self.label_path = f"{self.datapath}/labels"
        self.examples = examples
        self.is_normals = is_normals
        
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        pcd = o3d.io.read_point_cloud(f"{self.point_datapath}/{self.examples[index]}.pcd")
        pointlabels = []
        
        with open(f"{self.label_path}/{self.examples[index]}.txt") as f:
            for line in f.readlines():
                # strs = line.split(' ')
                pointlabels.append(int(line))
                
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        classes = list(set(pointlabels))
        class_encoded = np.zeros(self.num_classes)
        
        for ind in classes:
            class_encoded[ind] = 1
            
        pointlabels = np.array(pointlabels)
        
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
        
        if self.is_normals:
            pointdata = np.concatenate((points, normals), axis=1)
            return pointdata, class_encoded, pointlabels 
        else:
            return points, class_encoded, pointlabels 
        
        
        
class FeaturenetSingle(Dataset):
    def __init__(self, examples: list, datapath = './data/featurenet', num_points = 5000, num_classes = 25, is_normals = True) -> None:
        
        self.datapath = datapath
        self.num_points = num_points
        self.num_classes = num_classes
        self.point_datapath = f"{self.datapath}/featurenet_pcd_normalized"
        self.stl_datapath =  f"{self.datapath}/featurenet_stl"
        self.label_path = f"{self.datapath}/featurenet_labels"
        self.examples = examples
        self.is_normals = is_normals
        
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        
        pcd = o3d.io.read_point_cloud(f"{self.point_datapath}/{self.examples[index]}.pcd")
        pointlabels = []
        
        with open(f"{self.label_path}/{self.examples[index]}.txt") as f:
            for line in f.readlines():
                # strs = line.split(' ')
                pointlabels.append(int(line) + 1)
                
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        # classes = int(self.examples[index].split('_')[0])
        
        classes = list(set(pointlabels))
        class_encoded = np.zeros(self.num_classes)
        
        for ind in classes:
            if ind != 0:
                class_encoded[ind] = 1
            
            
        pointlabels = np.array(pointlabels)
        
        if self.num_points != 5000:
            
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
        
        if self.is_normals:
            pointdata = np.concatenate((points, normals), axis=1)
            return pointdata, class_encoded, pointlabels 
        else:
            return points, class_encoded, pointlabels 



def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y
            
    
if __name__ == '__main__':
    
    # data = PartNormalDataset()
    # DataLoader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    # for point,label, seg in DataLoader:
    #     print(point.shape)
    #     print(label.shape)
    #     print(seg.shape)
        
    #     print(label)
        
    #     new_label = to_categorical(label, 16).repeat(1, point.shape[1], 1)
        
    #     print(new_label)
        
    #     exit()
    
    
    # examples = get_example_list('./data/labels')
    # train, test = train_test_split(examples)
        
    # data = FFMaachiningModels(train, num_points=512)
    # Dataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)

    # for points, classes, seg in Dataloader:
        
    #     print(points.shape)
    #     print(classes)
    #     print(seg.shape)
        
    #     new_classes = torch.unsqueeze(classes, 1)
        
    #     print(new_classes.shape)
        
    #     print(torch.unsqueeze(classes, 1).repeat(1, points.shape[1], 1))
        
        
        
    #     # v = pptk.viewer(points[0][:,0:3])
        
    #     # w = pptk.viewer(points2[0])
        
    #     break
    
    examples = get_example_list('./data/featurenet/featurenet_labels')
    train, test = train_test_split(examples)
        
    data = FeaturenetSingle(train, num_points=512)
    Dataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)

    for points, classes, seg in Dataloader:
        
        print(points.shape)
        print(classes)
        print(seg.shape)
        
        new_classes = torch.unsqueeze(classes, 1)
        
        print(new_classes.shape)
        
        print(torch.unsqueeze(classes, 1).repeat(1, points.shape[1], 1))
        
        
        
        # v = pptk.viewer(points[0][:,0:3])
        
        # w = pptk.viewer(points2[0])
        
        break