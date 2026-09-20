import torch
from torchvision import transforms
from PIL import Image
import os
import glob

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from config import *
from tqdm import tqdm




root = os.getcwd()
train_img_link_list = sorted(glob.glob(root + '/data/train_images/*'))
train_mask_link_list = sorted(glob.glob(root + '/data/train_masks/*'))
test_img_link_list = sorted(glob.glob(root + '/data/test_images/*'))


val_img_list = [6414, 6420, 6433, 6444, 6468, 6556, 6565, 6640, 6715, 7691, 8368, 8447, 8817, 8902, 7233,
                6706, 7256, 7315, 7321, 7336, 7525, 7575, 8917, 8955, 9107, 6820, 6843, 6905, 6966, 6969]
val_img_list = [str(x) for x in val_img_list]

# Transformation pipeline
transform_img = transforms.Compose([
    transforms.Resize([HEIGHT, WIDTH]),
    transforms.ToTensor()
])

# ToTensor()把PIL图像转为tensor类型，形状成H×W×C 变成 C×H×W，像素值从 0~255 变成 0~1 的 float。
transform_mask = transforms.Compose([
    transforms.Resize([HEIGHT, WIDTH], interpolation=Image.NEAREST),  # interpolation是最近邻插值
    transforms.ToTensor()
])


if not os.path.exists(os.path.join(DATA_ROOT, f'train_images_{HEIGHT}')):
    # 在循环前创建目录
    os.makedirs(f'data/train_images_{HEIGHT}', exist_ok=True)
    os.makedirs(f'data/train_masks_{HEIGHT}', exist_ok=True)
    os.makedirs(f'data/val_images_{HEIGHT}', exist_ok=True)
    os.makedirs(f'data/val_masks_{HEIGHT}', exist_ok=True)

    for i in tqdm(range(len(train_img_link_list))):

        img_id = os.path.splitext(os.path.basename(train_img_link_list[i]))[0]  # basename取出文件名，splitext去除扩展名
        mask_id = os.path.splitext(os.path.basename(train_mask_link_list[i]))[0]
        assert img_id == mask_id  # Make sure id's match

        img = Image.open(train_img_link_list[i])
        mask = Image.open(train_mask_link_list[i])

        img = transform_img(img)  # 图像经过resize，to tensor变成[C, HEIGHT, WIDTH] 的 float tensor
        mask = transform_mask(mask) * 255  # tensor数据值为0~1范围，*255恢复原来灰度值
        mask = mask.int()  # 训练时，mask需要的是int或long类别ID

        # 保存为pt数据类型，后续torch.load速度块
        if img_id in val_img_list:
            torch.save(img, f'data/val_images_{HEIGHT}/{img_id}.pt')
            torch.save(mask, f'data/val_masks_{HEIGHT}/{mask_id}.pt')
        else:
            torch.save(img, f'data/train_images_{HEIGHT}/{img_id}.pt')
            torch.save(mask, f'data/train_masks_{HEIGHT}/{mask_id}.pt')

    print('Train and Validation sets created')


    "----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"
    # Iterate through all images in test set, transform them and store corresponging tensors in a new folder
    for i in tqdm(range(len(test_img_link_list))):
        img_id = test_img_link_list[i].split('/')[-1].split('.')[0]

        img = Image.open(test_img_link_list[i])
        img = transform_img(img)
        torch.save(img, f'data/test_images_{HEIGHT}/{img_id}.pt')

    print('Test set created')


    # add images of minority classes
    min_classes = [15, 17, 19, 20, 24]
    count = 0

    # 确保训练集增强样本的保存目录存在
    os.makedirs(f'data/train_images_{HEIGHT}', exist_ok=True)
    os.makedirs(f'data/train_masks_{HEIGHT}', exist_ok=True)

    for i in tqdm(range(len(train_img_link_list))):
        img_id = os.path.splitext(os.path.basename(train_img_link_list[i]))[0]
        mask_id = os.path.splitext(os.path.basename(train_mask_link_list[i]))[0]
        assert img_id == mask_id    # Make sure id's match

        img = Image.open(train_img_link_list[i])
        mask = Image.open(train_mask_link_list[i])

        img = transform_img(img)
        mask = transform_mask(mask) * 255
        mask = mask.int()

        if len(set(min_classes).intersection(set(mask.unique().tolist()))) > 0:  # 判断min_classes和mask中的所有ID类别是否有交集
            cropped_img = T.FiveCrop(size=(HEIGHT // 2, WIDTH // 2))(img)  # 对原图进行五裁剪，裁剪出来的图是原图的一半，然后通过放大，让少数类在局部中的占比更大
            cropped_mask = T.FiveCrop(size=(HEIGHT // 2, WIDTH // 2))(mask)

            for y in range(5):
                # 把第y个裁剪mask展平并转成python列表，检查是否该列表含有少数类别的ID
                list_mask = cropped_mask[y].flatten().tolist()
                if any(item in min_classes for item in list_mask):  # 如果裁剪中至少有一个像素属于少数类，就保留这个增强样本
                    # 把裁剪出来的小图返回原始尺，然后保存。
                    torch.save(TF.resize(cropped_img[y], size=(HEIGHT, WIDTH)),
                               f'data/train_images_{HEIGHT}/PLUS_{img_id}_{y}.pt')
                    torch.save(TF.resize(cropped_mask[y], size=(HEIGHT, WIDTH)),
                               f'data/train_masks_{HEIGHT}/PLUS_{img_id}_{y}.pt')
                    count += 1

    print('Additional', count, ' samples created')

else:
    print('已存在预处理结果，跳过')