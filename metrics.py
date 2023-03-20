import torch

def classwise_IoU_single(preds, targets, num_classes):
    # preds has shape (num_points, num_classes)
    # targets has shape (num_points,)

    # convert predictions and targets to integers
    preds = torch.argmax(preds, dim=1)
    targets = targets.long()

  

    # initialize confusion matrix
    conf_mat = torch.zeros((num_classes, num_classes))

    # fill the confusion matrix
    for i in range(num_classes):
        for j in range(num_classes):
            conf_mat[i, j] = torch.sum((preds == i) & (targets == j))

    # compute intersection and union for each class
    intersection = torch.diag(conf_mat)
    union = torch.sum(conf_mat, dim=0) + torch.sum(conf_mat, dim=1) - intersection

    # compute IoU for each class
    iou = intersection / (union + 1e-6)

    return iou

def f1_score_single(preds, targets, ignore_index = None):
    # preds has shape (num_points, num_classes)
    # targets has shape (num_points,)
    
    # convert predictions and targets to integers
    preds = torch.argmax(preds, dim=1)
    
    
    # mask out predictions and targets for the ignore class
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]
    targets = targets.long()

    # compute true positives, false positives, and false negatives
    tp = torch.sum((preds == targets) & (targets != -1))
    fp = torch.sum((preds != targets) & (preds != -1) & (targets != -1))
    fn = torch.sum((preds != targets) & (preds == -1) & (targets != -1))
    
    if tp == 0 and fp == 0 and fn == 0:
        return 0.0

    # compute precision, recall, and F1 score
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1_score = 2 * precision * recall / (precision + recall + 1e-6)

    return f1_score


def pointcloud_accuracy(preds, targets, ignore_index=None):
    # preds has shape (num_points, num_classes)
    # targets has shape (num_points,)

    # convert predictions to integers
    preds = torch.argmax(preds, dim=1)

    # mask out predictions and targets for the ignore class
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]

    # compute accuracy
    correct = torch.sum(preds == targets)
    total = targets.numel()
    accuracy = correct.float() / total

    return accuracy

import torch

def classwise_pointcloud_accuracy(preds, targets, ignore_index=None):
    # preds has shape (num_points, num_classes)
    # targets has shape (num_points,)

    # convert predictions to integers
    preds = torch.argmax(preds, dim=1)

    # mask out predictions and targets for the ignore class
    if ignore_index is not None:
        mask = targets != ignore_index
        preds = preds[mask]
        targets = targets[mask]


    # compute class-wise accuracy
    unique_classes = torch.unique(targets)
    class_accuracies = {}
    for class_id in unique_classes:
        mask = targets == class_id
        class_preds = preds[mask]
        class_targets = targets[mask]
        correct = torch.sum(class_preds == class_targets)
        total = class_targets.numel()
        class_accuracy = correct.float() / total
        class_accuracies[class_id.item()] = class_accuracy.item()

    return class_accuracies
