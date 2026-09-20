## Credits

This project is based on [Semantic-segmentation-with-PyTorch-Satellite-Imagery
]([https://github.com/原作者/原仓库](https://github.com/JenAlchimowicz/Semantic-segmentation-with-PyTorch-Satellite-Imagery)) by [Jen Alchimowicz](https://github.com/JenAlchimowicz).


数据集链接：https://figshare.com/collections/semantic_segmentation_satellite_imagery/6026765

1.代码运行流程：​ (一).dataset_splitter数据预处理​ (二).config里修改必要的参数​ (三).运行trian.py训练模型​ (四).predict.py进行推理​ (五).tabulate_data.py对比不同模型推理后的评价指标

2.文件解释​ (一).datase_splitter.py加载原始图片和掩码路径，定义预处理变化。划分训练和验证集。采用五裁剪方法对少 数类增强​ (二).dataset.py 定义DataLoader,并对训练数据集进行图片增强(亮度调整、水平/垂直翻转、旋转)​ (三).model.py 定义训练模型、预训练权重、类别等参数​ (四).config.py 一些必要的参数​ (五).utils.py 定义评价指标的函数​ (六).train.py 训练模型部分，并记录训练日志​ (七).predict.py 推理部分​ (八).tabulate_data.py 对比不同模型推理后的评价指标​ (九).代码运行后结果保存​ (十).我的思考记录

DeepLabV3Plus在边界质量（Boundary_F1: 0.7142）上最优，但三个模型都面临小目标丢失和类别不均衡的问题。增加训练轮次后，各类模型均出现不同程度的过拟合（小目标漏检加剧）。未来工作应考虑引入 Dice Loss + CE 混合损失，并增加小目标样本的过采样
