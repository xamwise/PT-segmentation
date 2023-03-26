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
# from dataset import PartNormalDataset
from dataset import FFMaachiningModels
from dataset import FFMachiningModels_hf5

import hydra
import omegaconf


from losses import FocalLoss, DiceLoss2, JaccardLoss, DiceLoss
from metrics import classwise_IoU_single, f1_score_single, pointcloud_accuracy, classwise_pointcloud_accuracy



CLASS_FREQUENCY = [12111238, 2069265, 457246, 35405, 219526, 704876, 930048, 
                    553817, 1285240, 407205, 564424, 671544, 129159]


# seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
#                'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
#                'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
# seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

SEG_CLASSES = {'None': [0], 'Hole': [1], 'Chamfer': [2], 'Fillet': [3], 'Round': [4], 'Slot': [5], 'Pocket': [6],
               'Step': [7], 'Gear': [8], 'Thread': [9], 'Boss': [10], 'Circular_step': [11], 'Ring': [12]}
            #    'Gear': [13], 'Thread': [14], 'Boss': [15]}
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

def calculate_class_weights(num_points_per_class):
    total_points = np.sum(num_points_per_class)
    num_classes = len(num_points_per_class)
    class_weights = [0] * num_classes

    for i in range(num_classes):
        class_weights[i] = total_points / (num_classes * num_points_per_class[i])

    # Normalize weights so they sum up to one
    class_weights = class_weights / np.sum(class_weights)

    return torch.Tensor(class_weights).cuda()


@hydra.main(config_path='config', config_name='partseg', version_base=None)
def main(args):
    omegaconf.OmegaConf.set_struct(args, False)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    logger = logging.getLogger(__name__)

    print(args)#.pretty())

    # root = hydra.utils.to_absolute_path('data/shapenetcore_partanno_segmentation_benchmark_v0_normal/')

    # TRAIN_DATASET2 = PartNormalDataset(root=root, npoints=args.num_point, split='trainval', normal_channel=args.normal)
    # trainDataLoader2 = torch.utils.data.DataLoader(TRAIN_DATASET2, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.num_point, split='test', normal_channel=args.normal)
    # testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

    examples = provider.get_example_list('',num_examples = 1007, f5 = True)
    
    train, test, val = provider.train_test_split(examples, val=True, split_ratio=0.1, val_ratio=0.1)
    
    TRAIN_DATA = FFMachiningModels_hf5(train, num_points=args.num_point)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATA, batch_size=args.batch_size, shuffle=True)
    VAL_DATA = FFMachiningModels_hf5(val, num_points=args.num_point)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATA, batch_size=args.batch_size, shuffle=True)
    TEST_DATA = FFMachiningModels_hf5(test, num_points=args.num_point)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATA, batch_size=args.batch_size, shuffle=True)

    logger.info('Finished loading DataSet ...')


    '''MODEL LOADING'''
    args.input_dim = (6 if args.normal else 3) 
    args.num_class = 13
    num_category = 13
    num_part = args.num_class
    
    shutil.copy(hydra.utils.to_absolute_path('models/{}/model.py'.format(args.model.name)), '.')
    
    class_weights = calculate_class_weights(CLASS_FREQUENCY)
    
            
    ################### LOSS #################

    classifier = getattr(importlib.import_module('models.{}.model'.format(args.model.name)), 'PointTransformerSeg')(args).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    # criterion = FocalLoss(gamma=5.0)
    # criterion = DiceLoss2(num_classes=args.num_class)
    # criterion = DiceLoss(num_classes=args.num_class)


    ##########################################
    try:
        checkpoint = torch.load('./best_models/best_model_own_data.pth')
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
    
    train_acc = []
    test_acc = []
    iou = []
    class_acc = []

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
            # points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            # points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            # points[:, :, :] = provider.rotate_point_cloud_with_normal(points[:, :, :])
            
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
            
            train_acc.append(test_metrics['accuracy'])
            test_acc.append(test_metrics['accuracy'])
            iou.append(test_metrics['average_iou'])
            class_acc.append(test_metrics['average_acc'])
            
            
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
            savepath = f'best_models/best_model_own_data_pointnet2_{str(args.num_point)}.pth'
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
        logger.info('Best class avg mIOU is: %.5f' % best_avg_iou)
        logger.info('Best class avg accracy is: %.5f' % best_avg_acc)
        global_epoch += 1

    print(f'{train_acc=}')
    print(f'{test_acc=}')
    print(f'{class_acc=}')
    print(f'{iou=}')

    print(args)

if __name__ == '__main__':
    main()