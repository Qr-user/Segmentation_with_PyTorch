from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as T
from config import *
import os
import glob
import torch
import torchvision.transforms.functional as TF
import time


train_img_link_list = sorted(glob.glob(os.path.join(DATA_ROOT, f'train_images_{HEIGHT}', '*')))
train_mask_link_list = sorted(glob.glob(os.path.join(DATA_ROOT, f'train_masks_{HEIGHT}', '*')))
val_img_link_list = sorted(glob.glob(os.path.join(DATA_ROOT, f'val_images_{HEIGHT}', '*')))
val_mask_link_list = sorted(glob.glob(os.path.join(DATA_ROOT, f'val_masks_{HEIGHT}', '*')))

# 简单校验，防止路径错/文件为空
assert len(train_img_link_list) == len(train_mask_link_list), \
    f'train 图({len(train_img_link_list)}) != 掩码({len(train_mask_link_list)})'
assert len(val_img_link_list) == len(val_mask_link_list), \
    f'val 图({len(val_img_link_list)}) != 掩码({len(val_mask_link_list)})'
assert len(train_img_link_list) > 0, \
    f'没在 {os.path.join(DATA_ROOT, "train_images_" + str(HEIGHT))} 找到 .pt 文件'


class tensorDataset(Dataset):
    def __init__(self, images: list, masks: list, train: bool):
        self.image_links = images
        self.mask_links = masks
        self.train = train

    def __getitem__(self, index):

        # Select a specific image's link
        img_id = self.image_links[index]
        mask_id = self.mask_links[index]

        # 之前因为保存为.pt,现在就不需要重新resize、TOTensor了，读取更快
        img = torch.load(img_id)
        mask = torch.load(mask_id)

        # 只有训练模型才做数据增强，验证集不增强，保证每次评估结果一致，可比较
        if self.train == True:
            img, mask = self.transform(img, mask)

        # 把 mask 从 (1, H, W) 变成 (H, W)。然后DataLoader转为(B, H, W)。因为损失函数接受的形状为(B,H,W)的mask图
        mask = mask.squeeze(0)

        # 分割任务中mask是类别ID，损失函数要求标签是整数类型
        mask = mask.long()

        return img, mask

    def __len__(self):
        return len(self.image_links)

    def transform(self, img, mask):

        temp_rand = np.random.rand()  # 生成0~1的随机数，决定是否调整亮度和翻转图像
        if temp_rand < 0.3:
            t_darken_image = T.ColorJitter(brightness=[0.6, 0.8])  # 表示亮度乘以0.6到0.8之间的随机值。目的是让模型适应不同关照条件
            img = t_darken_image(img)  # mask不变，因为亮度不影响类别标签？？？

        elif temp_rand > 0.7:
            t_brighten_image = T.ColorJitter(brightness=[1.2, 1.4])
            img = t_brighten_image(img)
            # Do nothing for mask - the colors don't change

        # 翻转卫星图像
        if np.random.rand() < 0.3:
            t_horizonal_flip = T.RandomHorizontalFlip(p=1)  # 水平翻转
            img = t_horizonal_flip(img)
            mask = t_horizonal_flip(mask)

        if np.random.rand() < 0.3:
            t_vertical_flip = T.RandomVerticalFlip(p=1)  # 垂直翻转
            img = t_vertical_flip(img)
            mask = t_vertical_flip(mask)

        # 水机翻转
        if np.random.rand() < 0.3:
            angle = float(np.random.uniform(0, 180))
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

        return img, mask




train_dataset = tensorDataset(train_img_link_list, train_mask_link_list, train=True)
val_dataset = tensorDataset(val_img_link_list, val_mask_link_list, train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

#
# # Test dataloaders
# start = time.time()
# for i, batch in enumerate(train_dataloader):
#     img_batch, img_mask = batch
#     if i == 0:   # [修改] 只看第 0 个 batch，避免打印 100 多次
#         print('train img batch :', img_batch.shape)
#         print('train mask batch:', img_mask.shape)
#     break        # [修改] 快速跳出，节省时间
#
# for i, batch in enumerate(val_dataloader):
#     img_batch, img_mask = batch
#     if i == 0:   # [修改] 同上
#         print('val img batch :', img_batch.shape)
#         print('val mask batch:', img_mask.shape)
#     break
#
# end = time.time()
# print('Time:', end - start)

