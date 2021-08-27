import torch
import torch.nn as nn
import numpy as np
import math
import scipy.spatial
import scipy.ndimage.morphology

"""
True Positive （真正， TP）预测为正的正样本
True Negative（真负 , TN）预测为负的负样本 
False Positive （假正， FP）预测为正的负样本
False Negative（假负 , FN）预测为负的正样本
"""


def metrics(predict, label, out_class):
    """Calculate the required metrics"""
    IOU_list = []
    Dice_list = []
    false_positive_rate_list = []
    false_negtive_rate_list = []
    acc = []
    for i in range(1, out_class):
        N = label.size(0)
        indice = []
        # 根据batch_size筛去全0label，有标签才计算评价指标
        for j in range(N):
            gt_true = torch.sum(label[j, i, :, :])
            if gt_true:
                indice.append(j)

        if indice:
            Dice_list.append(diceCoeffv2(predict[indice, i, :, :], label[indice, i, :, :]))
            IOU_list.append(IOU(predict[indice, i, :, :], label[indice, i, :, :]))
            # FP_FN_rate_list = FP_FN_rate(predict[indice, i, :, :], label[indice, i, :, :])
            # false_positive_rate_list.append(FP_FN_rate_list[0])
            # false_negtive_rate_list.append(FP_FN_rate_list[1])
            # accu = pixel_accuracy(predict[indice, i, :, :], label[indice, i, :, :])
            # if accu > 0.9:
            #     print(f'slice id:{i}, acc:{accu}')
            acc.append(pixel_accuracy(predict[indice, i, :, :], label[indice, i, :, :]))
    return mean(IOU_list), mean(Dice_list), mean(acc)


def mean(list):
    """计算平均值"""
    return sum(list) / len(list)


def batch_pix_accuracy(predict, target):
    """Batch Pixel Accuracy
    Args:
        predict: input 4D tensor
        target: label 3D tensor
    """
    _, predict = torch.max(predict, 1)
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1
    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))
    assert pixel_correct <= pixel_labeled, \
        "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(predict, target, nclass):
    """Batch Intersection of Union
    Args:
        predict: input 4D tensor
        target: label 3D tensor
        nclass: number of categories (int)
    """
    _, predict = torch.max(predict, 1)
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = predict.cpu().numpy() + 1
    target = target.cpu().numpy() + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), \
        "Intersection area should be smaller than Union area"
    return area_inter, area_union


def intersection_and_union(im_pred, im_lab, num_class):
    im_pred = np.asarray(im_pred)
    im_lab = np.asarray(im_lab)
    # Remove classes from unlabeled pixels in gt image.
    im_pred = im_pred * (im_lab > 0)
    # Compute area intersection:
    intersection = im_pred * (im_pred == im_lab)
    area_inter, _ = np.histogram(intersection, bins=num_class - 1,
                                 range=(1, num_class - 1))
    # Compute area union:
    area_pred, _ = np.histogram(im_pred, bins=num_class - 1,
                                range=(1, num_class - 1))
    area_lab, _ = np.histogram(im_lab, bins=num_class - 1,
                               range=(1, num_class - 1))
    area_union = area_pred + area_lab - area_inter
    return area_inter, area_union


def diceCoeff(pred, gt, smooth=1e-5, ):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / |pred| + |gt|
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    score = (2 * intersection + smooth) / (unionset + smooth)

    return score.sum() / N


def diceFlat(pred, gt, smooth=1e-5):
    intersection = ((pred * gt).sum()).item()

    unionset = (pred.sum() + gt.sum()).item()
    score = (2 * intersection + smooth) / (unionset + smooth)
    return score


def diceCoeffv2(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return score.sum() / N


def IOU(pred, gt, eps=1e-5):
    r""" computational formula：
        IOU = pred ∩ gt / pred ∪ gt
        IOU = tp / (tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    score = (tp + eps) / (tp + fp + fn + eps)

    return score.sum() / N


def FP_FN_rate(pred, gt, eps=1e-5):
    r"""computational formula：
        False_Positive_rate = fp / (fp + tn)
        False_Negtive_rate = fn / (fn + tp)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)

    false_positive_rate = fp / (fp + tn + eps)
    false_negtive_rate = fn / (fn + tp + eps)
    return false_positive_rate.sum() / N, false_negtive_rate.sum() / N


def diceCoeffv3(pred, gt, eps=1e-5):
    r""" computational formula：
        dice = (2 * tp) / (2 * tp + fp + fn)
    """

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)
    # 转为float，以防long类型之间相除结果为0
    score = (2 * tp + eps).float() / (2 * tp + fp + fn + eps).float()

    return score.sum() / N


def jaccard(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))

    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score.sum() / N


def jaccardFlat(pred, gt, eps=1e-5):
    pred_flat = pred.squeeze()
    gt_flat = gt.squeeze()
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0))
    score = (tp.float() + eps) / ((tp + fp + fn).float() + eps)
    return score


def jaccardv2(pred, gt, eps=1e-5):
    """TP / (TP + FP + FN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp

    score = (tp + eps).float() / (tp + fp + fn + eps).float()
    return score.sum() / N


def tversky(pred, gt, eps=1e-5, alpha=0.7):
    """TP / (TP + (1-alpha) * FP + alpha * FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum(gt_flat * pred_flat, dim=1)
    fp = torch.sum(pred_flat, dim=1) - tp
    fn = torch.sum(gt_flat, dim=1) - tp
    score = (tp + eps) / (tp + (1 - alpha) * fp + alpha * fn + eps)
    return score.sum() / N


def accuracy(pred, gt, eps=1e-5):
    """(TP + TN) / (TP + FP + FN + TN)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0), dim=1)
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)

    score = ((tp + tn).float() + eps) / ((tp + fp + tn + fn).float() + eps)
    return score.sum() / N


def precision(pred, gt, eps=1e-5):
    """TP / (TP + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0))
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))

    score = (tp.float() + eps) / ((tp + fp).float() + eps)

    return score.sum() / N


def pixel_accuracy(pred, gt, eps=1e-5):
    """TP / (TP + FN)"""
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    tp = torch.sum((pred_flat != 0) * (gt_flat != 0), dim=1)
    fn = torch.sum((pred_flat == 0) * (gt_flat != 0), dim=1)

    score = (tp.float() + eps) / ((tp + fn).float() + eps)
    # if score > 0.9:
    #     print(f'gt_1:{torch.sum(gt_flat, dim=1).item()}, pred_1:{torch.sum(pred_flat, dim=1).item()}')
    #     print(f'tp:{tp.item()}, fn:{fn.item()}')
    return score.sum() / N


def specificity(pred, gt, eps=1e-5):
    """TN / (TN + FP)"""

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
    fp = torch.sum((pred_flat != 0) * (gt_flat == 0))
    tn = torch.sum((pred_flat == 0) * (gt_flat == 0))

    score = (tn.float() + eps) / ((fp + tn).float() + eps)

    return score.sum() / N


def recall(pred, gt, eps=1e-5):
    return sensitivity(pred, gt)


if __name__ == '__main__':
    # shape = torch.Size([2, 3, 4, 4])
    # 模拟batch_size = 2
    '''
    1 0 0= bladder
    0 1 0 = tumor
    0 0 1= background 
    '''
    pred = torch.Tensor([[
        [[0, 1, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 1, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]]
    ])

    gt = torch.Tensor([[
        [[0, 1, 1, 0],
         [1, 0, 0, 1],
         [1, 0, 0, 1],
         [0, 1, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 1, 0],
         [0, 0, 0, 0]],
        [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 1]]]
    ])

    dice1 = diceCoeff(pred[:, 0:1, :], gt[:, 0:1, :])
    dice2 = jaccard(pred[:, 0:1, :], gt[:, 0:1, :])
    dice3 = diceCoeffv3(pred[:, 0:1, :], gt[:, 0:1, :])
    print(dice1, dice2, dice3)
