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
from dataset import FeaturenetSingle_hf5

import hydra
import omegaconf

from losses import FocalLoss, DiceLoss2, JaccardLoss, DiceLoss
from metrics import classwise_IoU_single, f1_score_single, pointcloud_accuracy, classwise_pointcloud_accuracy



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

@hydra.main(config_path='config', config_name='partseg', version_base=None)
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(args)#.pretty())

    examples = provider.get_example_list('',num_examples = 23977, f5 = True)
    
    train, test, val = provider.train_test_split(examples, val=True, split_ratio=0.1, val_ratio=0.1)
    
    TRAIN_DATA = FeaturenetSingle_hf5(train, num_points=args.num_point, isolated=True)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATA, batch_size=args.batch_size, shuffle=True)
    VAL_DATA = FeaturenetSingle_hf5(val, num_points=args.num_point, isolated=True)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATA, batch_size=args.batch_size, shuffle=True)
    TEST_DATA = FeaturenetSingle_hf5(test, num_points=args.num_point, isolated=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATA, batch_size=args.batch_size, shuffle=True)

    logger.info('Finished loading DataSet ...')


    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) 
    args.num_class = 24
    num_category = 24
    num_part = args.num_class
    
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    
            
    ################### LOSS #################

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = FocalLoss(gamma=5.0)
    # criterion = DiceLoss2(num_classes=args.num_class)
    # criterion = DiceLoss(num_classes=args.num_class)


    ##########################################
    try:
        checkpoint = torch.load('./best_models/best_model_featurenet_single.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        logger.info('Use pretrain model')
    except:
        logger.info('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), 
                                    lr=args.learning_rate, 
                                    momentum=0.9, 
                                    weight_decay=args.weight_decay,
                                    nesterov=True
                                    )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_acc = 0
    global_epoch = 0
    best_avg_iou = 0
    best_avg_acc = 0

    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        logger.info('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        classifier = classifier.train()

        '''learning one epoch'''
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points[:, :, :] = provider.rotate_point_cloud_with_normal(points[:, :, :])
            
            points = torch.Tensor(points)

            
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            optimizer.zero_grad()

            # seg_pred = classifier(torch.cat([points, torch.unsqueeze(label, 1).repeat(1, points.shape[1], 1)], -1))
            seg_pred = classifier(points)
            # loss = criterion(seg_pred, target)

            seg_pred = seg_pred.contiguous().view(-1, num_part)
            target = target.view(-1, 1)[:, 0]
            pred_choice = seg_pred.data.max(1)[1]
            # correct = pred_choice.eq(target.data).cpu().sum()
            # mean_correct.append(correct.item() / (args.batch_size * args.num_point))

            mean_correct.append(pointcloud_accuracy(seg_pred, target).cpu())

            
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

        train_instance_acc = np.mean(mean_correct)
        logger.info('Train accuracy is: %.5f' % train_instance_acc)

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

            for batch_id, (points, label, target) in tqdm(enumerate(valDataLoader), total=len(valDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                # seg_pred = classifier(torch.cat([points, torch.unsqueeze(label, 1).repeat(1, points.shape[1], 1)], -1))

                seg_pred = classifier(points)
                
                for predictions, target_labels in zip(seg_pred, target):
                    
                    
                    
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

        logger.info('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Class avg Accuracy: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['average_iou'], test_metrics['average_acc']))
        if (test_metrics['average_iou'] >= best_avg_iou):
            logger.info('Save model...')
            savepath = f'best_models/best_model_featurenet_single_{str(args.num_point)}.pth'
            logger.info('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['average_iou'],
                'class_avg_acc': test_metrics['average_acc'],
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            logger.info('Saving model....')
            best_avg_iou = test_metrics['average_iou']
            
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['average_acc'] > best_avg_acc:
            best_avg_acc = test_metrics['average_acc']
        logger.info('Best accuracy is: %.5f' % best_acc)
        logger.info('Best class avg mIOU is: %.5f' % best_avg_acc)
        logger.info('Best class avg accracy is: %.5f' % best_avg_acc)
        global_epoch += 1


if __name__ == '__main__':
    main()