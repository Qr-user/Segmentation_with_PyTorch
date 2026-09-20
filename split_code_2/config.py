


# Hyperparameters
HEIGHT = 368
WIDTH = 368

DATA_ROOT = "data"  # 用来存放增强后的图片，一般放在和训练数据集的同一个文件下

# 设置训练参数
STARTING_EPOCH = 0       #  若从头训练，请改成 0
EPOCHS = 300
LR = 1e-3
BATCH_SIZE = 24

# 采用什么模型
Model = "DeepLabV3Plus"
