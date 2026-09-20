from utils import *
import torch.nn
import os
from model import *
import pandas as pd
import matplotlib.pyplot as plt
from config import *
from dataset import *
import matplotlib
matplotlib.use('Agg')


root = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.join(root, 'models')
SAVE_RESULT = os.path.join(SAVE_DIR, f"{Model}")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_RESULT, exist_ok=True)


total_train_losses = []
total_val_losses = []
total_train_accuracy = []
total_val_accuracy = []
total_train_f1 = []
total_val_f1 = []


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss().to(DEVICE)  # 交叉熵要求输入的是(B, C, H, W)的logits。(B, H, W) 的类别 ID



def train():

    min_val_f1 = -1
    best_model_path = os.path.join(SAVE_DIR, Model, f'{Model}_res101_368_best.pt')

    for epoch in range(STARTING_EPOCH + 1, STARTING_EPOCH + EPOCHS + 1):

        # Train model
        model.train()
        train_losses = []
        train_accuracy = []
        train_f1 = []

        for i, batch in enumerate(train_dataloader):

            img_batch, mask_batch = batch   # img [B,3,H,W], mask[B,H,W]
            img_batch = img_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)

            # Train model
            optimizer.zero_grad()
            output = model(img_batch)  # output: [B, 25, H, W]
            loss = criterion(output, mask_batch)
            loss.backward()
            optimizer.step()

            # Add current loss to temporary list
            f1 = f1_dice_score(output, mask_batch)
            acc = accuracy(output, mask_batch)
            train_losses.append(loss.item())  # 把当前批次的指标加入临时列表，loss.item()把tensor转成python浮点数
            train_accuracy.append(acc)
            train_f1.append(f1)

        # 打印当前 epoch 训练集平均 loss、F1、accuracy
        print(f'TRAIN       Epoch: {epoch} | Epoch metrics | loss: {np.mean(train_losses):.4f}, f1: {np.mean(train_f1):.3f}, accuracy: {np.mean(train_accuracy):.3f}')

        #把当前 epoch 的平均指标加入全局列表，用于画曲线和保存 CSV。
        total_train_losses.append(np.mean(train_losses))
        total_train_accuracy.append(np.mean(train_accuracy))
        total_train_f1.append(np.mean(train_f1))

        # Validate model
        model.eval()
        val_losses = []
        val_accuracy = []
        val_f1 = []

        for i, batch in enumerate(val_dataloader):
            # Extract data, labels
            img_batch, mask_batch = batch
            img_batch = img_batch.to(DEVICE)
            mask_batch = mask_batch.to(DEVICE)

            # 加 torch.no_grad()，避免验证阶段构建计算图，节省显存
            with torch.no_grad(), torch.cuda.amp.autocast():
                output = model(img_batch)
                loss = criterion(output, mask_batch)

            # 计算并记录验证集指标。
            f1 = f1_dice_score(output, mask_batch)
            acc = accuracy(output, mask_batch)
            val_losses.append(loss.item())
            val_accuracy.append(acc)
            val_f1.append(f1)

        # 打印验证集平均指标
        print(f'VALIDATION  Epoch: {epoch} | Epoch metrics | loss: {np.mean(val_losses):.4f}, f1: {np.mean(val_f1):.3f}, accuracy: {np.mean(val_accuracy):.3f}')
        print('---------------------------------------------------------------------------------')
        total_val_losses.append(np.mean(val_losses))
        total_val_accuracy.append(np.mean(val_accuracy))
        total_val_f1.append(np.mean(val_f1))

        # Save the model
        if np.mean(val_f1) > min_val_f1:
            os.makedirs(SAVE_DIR, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            min_val_f1 = np.mean(val_f1)

        # Save the results so far
        temp_df = pd.DataFrame(
            list(zip(total_train_losses, total_val_losses, total_train_f1, total_val_f1,
                     total_train_accuracy, total_val_accuracy)),
            columns=['train_loss', 'val_loss', 'train_f1', 'val_f1',
                     'train_accuracy', 'val_accuracy']
        )
        # 加 .csv 扩展名，路径用 SAVE_DIR
        temp_df.to_csv(os.path.join(SAVE_DIR, Model, f'{Model}_train_val_measures.csv'), index=False)


if __name__ == '__main__':
    train()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    epochs = range(1, len(total_train_losses) + 1)
    plt.plot(epochs, total_train_losses, label='train loss')
    plt.plot(epochs, total_val_losses, label='val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, total_train_f1, label='train f1')
    plt.plot(epochs, total_val_f1, label='val f1')
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()

    plt.tight_layout()  # 自动调整子图布局，避免重叠。

    # plt.show()
    plt.savefig(os.path.join(SAVE_DIR, Model, f'{Model}_train_curve.png'))
    plt.close()




