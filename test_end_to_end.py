"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np

from pathlib import Path
from tqdm import tqdm

from dataset import FeaturenetSingle, FeaturenetMulti, FFMaachiningModels

import hydra
import omegaconf

from provider import get_example_list, train_test_split
from metrics import classwise_IoU_single, pointcloud_accuracy, classwise_pointcloud_accuracy, classwise_IoU_triangles, triangle_accuracy, classwise_triangle_accuracy

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from stl import mesh as mesh_stl
import matplotlib.pyplot as plt
import stl

PRED_PATH = './knn_data/predictions'

def gen_colors(n):   
    colors = []
    for i in range(n):
        r, g, b = np.random.rand(3)
        colors.append([r, g, b])
        
    return colors

def visualize_point_cloud(points, labels):
    # if isinstance(labels, list):
    #     labels = np.array(labels)
    # colors = plt.get_cmap("tab10")(labels / (num_classes - 1))[:, :3]
    
    colors = gen_colors(25)
    
    point_colors = [colors[int(j)] for j in labels] 
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.visualization.draw_geometries([pcd])
    
def visualize_mesh(meshes):
    colors = gen_colors(len(meshes))
    # num_colors = np.arange(0, len(meshes))
    # colors = plt.get_cmap("tab10")(num_colors / (len(num colors) - 1))[:, :3]
    # Paint the triangles with the colors
    for z, mesh in enumerate(meshes):
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(colors[z])
      
    o3d.visualization.draw_geometries(meshes)
    
def show_mesh(faces, vertices, predictions):
    features = set(predictions)
    chosen_triangles = predictions != 0
    chosen_faces = faces[chosen_triangles]
    
    
    ref_model = mesh_stl.Mesh(np.zeros(faces.shape[0], dtype=mesh_stl.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            ref_model.vectors[i][j] = vertices[f[j],:]
    
    model = mesh_stl.Mesh(np.zeros(chosen_faces.shape[0], dtype=mesh_stl.Mesh.dtype))
    for i, f in enumerate(chosen_faces):
        for j in range(3):
            model.vectors[i][j] = vertices[f[j],:]
            
    model.save(f'./knn_data/test.stl', mode=stl.Mode.ASCII, update_normals=True)
    
    ref_model.save(f'./knn_data/ref.stl', mode=stl.Mode.ASCII, update_normals=True)
    
    for z, feature in enumerate(features):
        
        # if feature != 0:
    
        chosen_triangles = predictions == feature
        
        faces_parts = faces[chosen_triangles]
        
        model = mesh_stl.Mesh(np.zeros(faces_parts.shape[0], dtype=mesh_stl.Mesh.dtype))
        for i, f in enumerate(faces_parts):
            for j in range(3):
                model.vectors[i][j] = vertices[f[j],:]
                
        model.save(f'./knn_data/predictions/{feature}.stl', mode=stl.Mode.ASCII, update_normals=False)

    pred_meshes = []
    files_stl = os.listdir(PRED_PATH)
    files_stl = [os.path.join(PRED_PATH, file) for file in files_stl]
    
    for stl_file in files_stl:
        loaded_mesh = o3d.io.read_triangle_mesh(stl_file)
        pred_meshes.append(loaded_mesh)
        os.remove(stl_file)
              
    visualize_mesh(pred_meshes)
   





SEG_CLASSES_OWN = {'None': [0], 'Hole': [1], 'Chamfer': [2], 'Fillet': [3], 'Round': [4], 'Slot': [5], 'Pocket': [6],
               'Step': [7], 'Gear': [8], 'Thread': [9], 'Boss': [10], 'Circular_step': [11], 'Ring': [12]}
                # 'Gear': [13], 'Thread': [14], 'Boss': [15]}
SEG_LABEL_TO_CAT_OWN = {}  

for cat in SEG_CLASSES_OWN.keys():
    SEG_LABEL_TO_CAT_OWN[SEG_CLASSES_OWN[cat][0]] = cat



SEG_CLASSES = {'None': [0], 'Ring': [1], 'Through_Hole': [2], 'Blind_Hole': [3], 'Triangular_passage': [4], 
               'Rectangular_passage': [5], 'Circular_through_slot': [6], 'Triangular_through_slot': [7],
               'Rectangular_through_slot': [8], 'Rectangular_blind_slot': [9], 'Triangular_pocket': [10],
               'Rectangular_pocket': [11], 'Circular_end_pocket': [12], 'Triangular_blind_step': [13], 
               'Circular_blind_step': [14],'Rectangular_blind_step': [15], 'Rectangular_through_step': [16],
               '2_sides_through_step': [17], 'slanted_through_step': [18], 'chamfer': [19], 'round': [20],
               'v_circular_end_blind_slot': [21], 'h_circular_end_blind_slot': [22], '6_sides_passage': [23],
               '6_sides_pocket': [24]}
SEG_LABEL_TO_CAT = {}  # {0:Airplane, 1:Airplane, ...49:Table}



for cat in SEG_CLASSES.keys():
    SEG_LABEL_TO_CAT[SEG_CLASSES[cat][0]] = cat


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


@hydra.main(config_path='config', config_name='inference', version_base=None)
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(args)#.pretty())

    
    
    
    
    examples_single = get_example_list('./data/featurenet/featurenet_labels', f5=False)
    examples_multi = get_example_list('./data/multi_featurenet/multi_featurenet_labels', f5=False)
    examples_own = get_example_list('./data/own/own_labels', f5=False)
    train_single, test_single = train_test_split(examples_single, split_ratio=0.2)
    train_multi, test_multi = train_test_split(examples_multi, split_ratio=0.1)
    train_own, test_own = train_test_split(examples_own, split_ratio=0.2)    
        
        
    # data = FeaturenetSingle(test_single, num_points=2048)
    # Dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
    
    # data = FeaturenetMulti(test_multi, num_points=4096)
    # Dataloader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True)
    
    data = FFMaachiningModels(test_own, num_points=2048)
    Dataloader = torch.utils.data.DataLoader(data, batch_size=4, shuffle=True)
  
    

    logger.info('Finished loading DataSet ...')


    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) 
    args.num_class = 13
    num_category = 13
    num_part = args.num_class
    
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    
            

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    
    # checkpoint = torch.load('./best_models/best_model_featurenet_multi_16nn_cubes_4096.pth')
    checkpoint = torch.load('./best_models/best_model_own_data_weighted_ultra_run_2048.pth')
    # checkpoint = torch.load('./best_models/best_model_featurenet_single_cubes_2048.pth')
 
    
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Loaded pretrained model')
   
    # h = classifier.state_dict()
    # print("Model's state_dict:")
    # for param_tensor in classifier.state_dict():
    #     print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())
   
    
    seen_features = 0
    classified = 0
    check_seen = 0

    with torch.no_grad():
        test_metrics = {}
        mean_test_accuracy = []
        mean_f1_score = []
        classwise_iou = []
        mean_iou_per_class = [[] for i in range(args.num_class)]
        mean_accuracy_per_class = [[] for i in range(args.num_class)]
        classwise_accuracies = []
        classes_per_example = []
        
        mean_iou_classwise = []
        mean_accuracy_classwise = []

    

        classifier = classifier.eval()

        for batch_id, (points, target, centroids, triangle_labels, length, pathes) in tqdm(enumerate(Dataloader), total=len(Dataloader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, target, centroids, triangle_labels, length, = points.float().cuda(), target.long().cpu(), centroids.float().cpu(), triangle_labels.long().cpu(), length.long()
            # seg_pred = classifier(torch.cat([points, torch.unsqueeze(label, 1).repeat(1, points.shape[1], 1)], -1))

            seg_pred = classifier(points)
            
            points = points.float().cpu()
            seg_pred = seg_pred.float().cpu()
            
            for predictions, point , centroid, triangle_label, l, path in zip(seg_pred, points, centroids, triangle_labels, length, pathes):
                
                ref_model = o3d.io.read_triangle_mesh(path)
                ref_model.compute_vertex_normals()
                predictions = torch.argmax(predictions, dim=1)
                
                loaded_vertices = np.asarray(ref_model.vertices)
                loaded_faces = np.asarray(ref_model.triangles)
                
                if args.visualize:    
                    o3d.visualization.draw_geometries([ref_model])                    
                    visualize_point_cloud(point[:,:3], np.zeros((len(predictions),)))
                    visualize_point_cloud(point[:,:3], predictions)

                    
                centroid = centroid[:l]
                triangle_label = triangle_label[:l]
                
                knn_classifier = KNeighborsClassifier(n_neighbors=args.nn)
                  
                knn_classifier.fit(point[:,:3], predictions)
                
                triangle_predictions = knn_classifier.predict(centroid)
                
                if args.visualize:
                    show_mesh(loaded_faces, loaded_vertices, triangle_predictions)
                
                triangle_predictions = torch.tensor(triangle_predictions)
                
                
                
                classes_predicted_points = torch.unique(predictions).cpu().tolist()
                classes = torch.unique(triangle_label).cpu().tolist()
                
                seen_features += len(classes) - 1
                
                mean_test_accuracy.append(triangle_accuracy(triangle_predictions, triangle_label).cpu())
 
                classwise_accuracies = classwise_triangle_accuracy(triangle_predictions, triangle_label)
                classwise_iou = classwise_IoU_triangles(triangle_predictions, triangle_label, args.num_class).cpu()
                
                for ind_class in classes:
                    mean_accuracy_per_class[ind_class].append(classwise_accuracies[ind_class])
                    mean_iou_per_class[ind_class].append(classwise_iou[ind_class])
                check = True
                
                
        for i, scores in enumerate(mean_iou_per_class):
            if i != 0:
                for score in scores:
                    check_seen += 1
                    if score > 0.5:
                        classified += 1  
                        
        correctly_classified = classified/check_seen    
                        
            
        test_metrics['accuracy'] = np.mean(mean_test_accuracy)
        
        for classes_acc, classes_iou in zip(mean_accuracy_per_class, mean_iou_per_class):
            mean_accuracy_classwise.append(np.mean(classes_acc))
            mean_iou_classwise.append(np.mean(classes_iou))
            
        test_metrics['classwise_mIoU'] = mean_iou_classwise
        test_metrics['mean_accuracy_per_class'] = mean_accuracy_classwise
        
        test_metrics['average_acc'] = np.mean(mean_accuracy_classwise)
        test_metrics['average_iou'] = np.mean(mean_iou_classwise)
        
        
        for cat in sorted(SEG_LABEL_TO_CAT.keys()):
            name = SEG_LABEL_TO_CAT[cat]
            logger.info('eval of %s mIoU %f mACC %f' % (name + ' ' * (14 - len(name)), mean_iou_classwise[cat], mean_accuracy_classwise[cat]))
        
        
        # for cat in sorted(SEG_LABEL_TO_CAT.keys()):
        #     name = SEG_LABEL_TO_CAT[cat]
        #     logger.info('eval mACC of %s %f' % (name + ' ' * (14- len(name)), mean_accuracy_classwise[cat]))

    logger.info('Test Accuracy: %f  Class avg mIOU: %f   Class avg Accuracy: %f   Correctly Classified: %f' % (
            test_metrics['accuracy'], test_metrics['average_iou'], test_metrics['average_acc'], correctly_classified))
    


if __name__ == '__main__':
    main()