from torch import nn
from matplotlib import pyplot as plt
import torch.nn
from model import *
import numpy as np

# Initialization
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()   # 指标计算不需要梯度，避免占用显存
def f1_dice_score(preds, true_mask):
    '''
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
    preds should be (B, 25, H, W)
    true_mask should be (B, H, W)
    '''

    f1_batch = []

    for i in range(len(preds)):
        f1_image = []
        img = preds[i].to(DEVICE)
        mask = true_mask[i].to(DEVICE)

        # Change shape of img from [25, H, W] to [H, W]
        img = torch.argmax(img, dim=0)

        for label in range(25):
            if torch.sum(mask == label) != 0:
                area_of_intersect = torch.sum((img == label) * (mask == label))  # 计算预测为该类且真是也为该类的像素数，即交集TP
                area_of_img = torch.sum(img == label)  # 预测为该类的像素总数。
                area_of_label = torch.sum(mask == label)  # 真实为该类的像素总数。
                f1 = 2 * area_of_intersect / (area_of_img + area_of_label)
                f1_image.append(f1)

        # t.item() 把 tensor 标量转成 Python float。因为上面计算F1时，用的是preds (B, 25, H, W)。true_mask  (B, H, W)
        f1_batch.append(np.mean([t.item() for t in f1_image]))
    return np.mean(f1_batch)


# Accuracy
@torch.no_grad()
def accuracy(preds, true_mask):
    '''
    preds should be (B, 25, H, W)
    true_mask should be (B, H, W)
    '''
    accuracy_batch = []

    for i in range(len(preds)):
        img = preds[i].to(DEVICE)
        mask = true_mask[i].to(DEVICE)

        # Change shape of img from [25, H, W] to [H, W]
        img = torch.argmax(img, dim=0)

        # 预测正确的像素数。mask.numel()像素总数
        accuracy_batch.append(torch.sum(img == mask).item() / mask.numel())

    return np.mean(accuracy_batch)


#  评价指标计算函数
@torch.no_grad()
def compute_confusion_matrix(pred, gt, num_classes):
    """
    从单张预测和真实 mask 构建混淆矩阵。
    pred: (H, W) numpy，值 0~num_classes-1
    gt:   (H, W) numpy，值 0~num_classes-1
    返回: (num_classes, num_classes) numpy，行=真实，列=预测
    """
    mask = (gt >= 0) & (gt < num_classes)
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def compute_metrics_from_cm(confusion_matrix):
    """
    从全局混淆矩阵计算 mIoU、Dice/F1、Pixel Acc。
    返回 dict。
    """
    cm = confusion_matrix.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp          # 列和 - 对角线
    fn = cm.sum(axis=1) - tp          # 行和 - 对角线
    total = cm.sum()

    # Pixel Accuracy = 对角线之和 / 总像素
    pixel_acc = tp.sum() / (total + 1e-10)

    # mIoU：逐类 IoU = TP / (TP + FP + FN)，再求均值
    iou = tp / (tp + fp + fn + 1e-10)
    miou = np.nanmean(iou)

    # Dice / F1：逐类 F1 = 2TP / (2TP + FP + FN)，再求均值
    dice = 2 * tp / (2 * tp + fp + fn + 1e-10)
    mean_dice = np.nanmean(dice)

    return {
        'miou':      float(miou),
        'mean_dice': float(mean_dice),
        'pixel_acc': float(pixel_acc),
    }


@torch.no_grad()
def boundary_f1(pred, gt, tolerance=2):
    """
    计算单张图的 Boundary F1。
    pred, gt: (H, W) numpy
    tolerance: 边界容差像素，默认 2
    参考: https://blog.csdn.net/u013172930/article/details/153601469
    """
    def extract_boundary(mask):
        """用形态学腐蚀提取边界像素"""
        # 对每个类别分别做腐蚀，边界 = mask - 腐蚀后的mask
        if mask.max() == 0:
            return np.zeros_like(mask, dtype=bool)
        eroded = ndimage.binary_erosion(mask > 0, iterations=1)
        return (mask > 0) & (~eroded)

    b_pred = extract_boundary(pred)
    b_gt   = extract_boundary(gt)

    if b_pred.sum() == 0 and b_gt.sum() == 0:
        return 1.0
    if b_pred.sum() == 0 or b_gt.sum() == 0:
        return 0.0

    # 计算距离变换：GT 边界到每个像素的距离
    dist_gt = ndimage.distance_transform_edt(~b_gt)
    dist_pred = ndimage.distance_transform_edt(~b_pred)

    # 匹配：预测边界像素在 GT 边界 tolerance 内 → TP
    tp_pred = (b_pred & (dist_gt <= tolerance)).sum()
    tp_gt   = (b_gt & (dist_pred <= tolerance)).sum()

    precision = tp_pred / (b_pred.sum() + 1e-10)
    recall    = tp_gt   / (b_gt.sum()   + 1e-10)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_model_params_m(model):
    """计算模型参数量（单位：M）"""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e6

