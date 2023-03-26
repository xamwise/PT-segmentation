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

from dataset import FeaturenetSingle
from dataset import FeaturenetSingle_hf5, FeaturenetMulti_hf5, FFMachiningModels_hf5

import hydra
import omegaconf

from losses import FocalLoss, DiceLoss2, JaccardLoss, DiceLoss
from metrics import classwise_IoU_single, f1_score_single, pointcloud_accuracy, classwise_pointcloud_accuracy

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np

def visualize_point_cloud(points, labels, num_classes):
    colors = plt.get_cmap("tab10")(labels / (num_classes - 1))[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])



SEG_CLASSES_OWN = {'None': [0], 'Hole': [1], 'Chamfer': [2], 'Fillet': [3], 'Round': [4], 'Slot': [5], 'Pocket': [6],
               'Step': [7], 'Gear': [8], 'Thread': [9], 'Boss': [10], 'Circular_step': [11], 'Ring': [12],
                'Gear': [13], 'Thread': [14], 'Boss': [15]}
SEG_LABEL_TO_CAT_OWN = {}  

for cat in SEG_CLASSES_OWN.keys():
    SEG_LABEL_TO_CAT_OWN[SEG_CLASSES_OWN[cat][0]] = cat



# SEG_CLASSES = {'None': [0], 'Ring': [1], 'Through_Hole': [2], 'Blind_Hole': [3], 'Triangular_passage': [4], 
#                'Rectangular_passage': [5], 'Circular_through_slot': [6], 'Triangular_through_slot': [7],
#                'Rectangular_through_slot': [8], 'Rectangular_blind_slot': [9], 'Triangular_pocket': [10],
#                'Rectangular_pocket': [11], 'Circular_end_pocket': [12], 'Triangular_blind_step': [13], 
#                'Circular_blind_step': [14],'Rectangular_blind_step': [15], 'Rectangular_through_step': [16],
#                '2_sides_through_step': [17], 'slanted_through_step': [18], 'chamfer': [19], 'round': [20],
#                'v_circular_end_blind_slot': [21], 'h_circular_end_blind_slot': [22], '6_sides_passage': [23],
#                '6_sides_pocket': [24]}
SEG_LABEL_TO_CAT = {}  # {0:Airplane, 1:Airplane, ...49:Table}



SEG_CLASSES = {'Ring': [0], 'Through_Hole': [1], 'Blind_Hole': [2], 'Triangular_passage': [3], 
               'Rectangular_passage': [4], 'Circular_through_slot': [5], 'Triangular_through_slot': [6],
               'Rectangular_through_slot': [7], 'Rectangular_blind_slot': [8], 'Triangular_pocket': [9],
               'Rectangular_pocket': [10], 'Circular_end_pocket': [11], 'Triangular_blind_step': [12], 
               'Circular_blind_step': [13],'Rectangular_blind_step': [14], 'Rectangular_through_step': [15],
               '2_sides_through_step': [16], 'slanted_through_step': [17], 'chamfer': [18], 'round': [19],
               'v_circular_end_blind_slot': [20], 'h_circular_end_blind_slot': [21], '6_sides_passage': [22],
               '6_sides_pocket': [23]}


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

    # examples = provider.get_example_list('',num_examples = 1007, f5 = True)
    
    # # train, test, val = provider.train_test_split(examples, val=True, split_ratio=0.1, val_ratio=0.1)
    
    # TEST_DATA_OWN = FFMachiningModels_hf5(examples, num_points=args.num_point)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATA_OWN, batch_size=args.batch_size, shuffle=True)
    
    examples2 = provider.get_example_list('',num_examples = 1000, f5 = True)
    
    train, test, val = provider.train_test_split(examples2, val=True)
    
    TEST_DATA = FeaturenetMulti_hf5(examples2, num_points=args.num_point, isolated=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATA, batch_size=args.batch_size, shuffle=True)
    
    
    

    logger.info('Finished loading DataSet ...')


    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) 
    args.num_class = 24
    num_category = 24
    num_part = args.num_class
    
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    
            

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    
    checkpoint = torch.load('./best_models/best_model_featurenet_single_1024.pth')
     
    start_epoch = checkpoint['epoch']
    classifier.load_state_dict(checkpoint['model_state_dict'])
    logger.info('Loaded pretrained model')
   
    # h = classifier.state_dict()
    # print("Model's state_dict:")
    # for param_tensor in classifier.state_dict():
    #     print(param_tensor, "\t", classifier.state_dict()[param_tensor].size())
   
    
   

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

        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            # seg_pred = classifier(torch.cat([points, torch.unsqueeze(label, 1).repeat(1, points.shape[1], 1)], -1))

            seg_pred = classifier(points)
            
            
            for predictions, target_labels in zip(seg_pred, target):
                
            
                print(torch.unique(torch.argmax(predictions, dim=1)))    
                
                classes_per_example = torch.unique(target_labels).cpu().tolist()
                mean_test_accuracy.append(pointcloud_accuracy(predictions, target_labels).cpu())
                mean_f1_score.append(f1_score_single(predictions, target_labels).cpu())
                
                classwise_accuracies = classwise_pointcloud_accuracy(predictions, target_labels)
                classwise_iou = classwise_IoU_single(predictions, target_labels, args.num_class).cpu()
                
                for ind_class in classes_per_example:
                    mean_accuracy_per_class[ind_class].append(classwise_accuracies[ind_class])
                    mean_iou_per_class[ind_class].append(classwise_iou[ind_class])
                check = True
                
                        
            
        test_metrics['accuracy'] = np.mean(mean_test_accuracy)
        test_metrics['mean_f1_score'] = np.mean(mean_f1_score)
        
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

    logger.info('Test Accuracy: %f  Class avg mIOU: %f   Class avg Accuracy: %f' % (
            test_metrics['accuracy'], test_metrics['average_iou'], test_metrics['average_acc']))
    


if __name__ == '__main__':
    main()