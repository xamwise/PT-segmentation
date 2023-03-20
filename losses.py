"""_summary_

Implementation of focal loss by mathiaszinnen @ https://github.com/mathiaszinnen/focal_loss_torch



"""

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union
import torch.nn.functional as F


class JaccardLoss(nn.Module):
    def __init__(self, num_classes):
        super(JaccardLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        # convert target to one-hot encoding
        target = F.one_hot(target, num_classes=self.num_classes).float()

        # calculate intersection and union
        intersection = torch.sum(input * target, dim=1)
        union = torch.sum(input + target, dim=1) - intersection

        # calculate jaccard coefficient
        jaccard_coefficient = (intersection + 1e-6) / (union + 1e-6)

        # calculate jaccard loss
        jaccard_loss = 1 - jaccard_coefficient.mean()

        return jaccard_loss


class DiceLoss2(nn.Module):
    def __init__(self, num_classes):
        super(DiceLoss2, self).__init__()
        self.num_classes = num_classes

    def forward(self, input, target):
        
        # convert target to one-hot encoding
        target = F.one_hot(target, num_classes=self.num_classes).float()

        # transpose input to shape (batch_size, num_classes, num_points)
        # input = input.permute(0, 2, 1)

        # calculate intersection and union
        intersection = torch.sum(input * target, dim=2)
        cardinality = torch.sum(input + target, dim=2)

        # calculate dice coefficient
        dice_coefficient = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)

        # calculate dice loss
        dice_loss = 1 - dice_coefficient.mean()

        return dice_loss
    

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # inputs has shape (batch_size, num_points, num_classes)
        # targets has shape (batch_size, num_points)

        # compute cross entropy loss
        ce_loss = F.cross_entropy(inputs.view(-1, inputs.size(-1)), targets.view(-1), reduction='none')

        # reshape the ce_loss to match the input shape
        ce_loss = ce_loss.view(inputs.size(0), inputs.size(1))

        # compute softmax probabilities
        probs = F.softmax(inputs, dim=2)

        # select only the probabilities for the true class
        true_class_probs = probs.gather(2, targets.unsqueeze(2)).squeeze(2)

        # compute the focal weight for each example
        focal_weight = (1 - true_class_probs) ** self.gamma

        # apply the class weights if provided
        if self.alpha is not None:
            alpha = self.alpha.to(targets.device)
            class_weight = alpha.gather(0, targets.view(-1))
            class_weight = class_weight.view(inputs.size(0), inputs.size(1))
            focal_weight = class_weight * focal_weight

        # compute the weighted focal loss
        loss = focal_weight * ce_loss

        # return the average loss over the batch
        return loss.mean()



# class FocalLoss(nn.Module):
#     """Computes the focal loss between input and target
#     as described here https://arxiv.org/abs/1708.02002v2
#     Args:
#         gamma (float):  The focal loss focusing parameter.
#         weights (Union[None, Tensor]): Rescaling weight given to each class.
#         If given, has to be a Tensor of size C. optional.
#         reduction (str): Specifies the reduction to apply to the output.
#         it should be one of the following 'none', 'mean', or 'sum'.
#         default 'mean'.
#         ignore_index (int): Specifies a target value that is ignored and
#         does not contribute to the input gradient. optional.
#         eps (float): smoothing to prevent log from returning inf.
#     """
#     def __init__(
#             self,
#             gamma,
#             weights: Union[None, Tensor] = None,
#             reduction: str = 'mean',
#             ignore_index=-100,
#             eps=1e-16
#             ) -> None:
#         super().__init__()
#         if reduction not in ['mean', 'none', 'sum']:
#             raise NotImplementedError(
#                 'Reduction {} not implemented.'.format(reduction)
#                 )
#         assert weights is None or isinstance(weights, Tensor), \
#             'weights should be of type Tensor or None, but {} given'.format(
#                 type(weights))
#         self.reduction = reduction
#         self.gamma = gamma
#         self.ignore_index = ignore_index
#         self.eps = eps
#         self.weights = weights

#     def _get_weights(self, target: Tensor) -> Tensor:
#         if self.weights is None:
#             return torch.ones(target.shape[0])
#         weights = target * self.weights
#         return weights.sum(dim=-1)

#     def _process_target(
#             self, target: Tensor, num_classes: int
#             ) -> Tensor:
#         target = target.view(-1)
#         return one_hot(target, num_classes=num_classes)

#     def _process_preds(self, x: Tensor) -> Tensor:
#         if x.dim() == 1:
#             x = torch.vstack([1 - x, x])
#             x = x.permute(1, 0)
#             return x
#         return x.view(-1, x.shape[-1])

#     def _calc_pt(
#             self, target: Tensor, x: Tensor, mask: Tensor
#             ) -> Tensor:
#         p = target * x
#         p = p.sum(dim=-1)
#         p = p * ~mask
#         return p

#     def forward(self, x: Tensor, target: Tensor) -> Tensor:
#         assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
#             'The predictions values should be between 0 and 1, \
#                 make sure to pass the values to sigmoid for binary \
#                 classification or softmax for multi-class classification'
#         )
#         mask = target == self.ignore_index
#         mask = mask.view(-1)
#         x = self._process_preds(x)
#         num_classes = x.shape[-1]
#         target = self._process_target(target, num_classes)
#         weights = self._get_weights(target).to(x.device)
#         pt = self._calc_pt(target, x, mask)
#         focal = 1 - pt
#         nll = -torch.log(self.eps + pt)
#         nll = nll.masked_fill(mask, 0)
#         loss = weights * (focal ** self.gamma) * nll
#         return self._reduce(loss, mask, weights)

#     def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
#         if self.reduction == 'mean':
#             return x.sum() / (~mask * weights).sum()
#         elif self.reduction == 'sum':
#             return x.sum()
#         else:
#             return x



class FocalLoss2(nn.Module):
    def __init__(self,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss2, self).__init__()
        assert reduction in ('mean', 'sum'), \
            "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        # pred = pred.transpose(0, 1)
        # # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        # pred = pred.reshape(pred.size(0), -1)
        # # [C, N] -> [N, C]
        # pred = pred.transpose(0, 1).contiguous()
        # # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        # target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), \
            "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.

        num_classes = pred.size(1)
        target_one_hot = F.one_hot(target, num_classes=num_classes)


        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target_one_hot + pred_sigmoid * (1 - target_one_hot)
        focal_weight = (alpha * target_one_hot + (1 - alpha) *
                        (1 - target_one_hot)) * one_minus_pt.pow(self.gamma)

        loss = F.cross_entropy(
            pred, target.long(), reduction='none') * focal_weight
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.total()
        return self.loss_weight * loss
    
    
    
class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1,
                 exponent=2,
                 loss_weight=1.0,
                 ignore_index=255):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                **kwargs):

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(0), \
            "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1),
            num_classes=num_classes)

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = torch.sum(pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)) + self.smooth
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss
