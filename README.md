# Satellite Imagery Semantic Segmentation with PyTorch

> 基于 PyTorch 的卫星图像语义分割项目，包含数据预处理、模型训练、推理以及多模型评价指标对比流程。

## 目录

- [项目简介](#项目简介)
- [致谢与来源](#致谢与来源)
- [数据集](#数据集)
- [运行流程](#运行流程)
- [文件说明](#文件说明)
- [实验结果与思考](#实验结果与思考)
- [未来改进](#未来改进)
- [结果保存与思考记录](#结果保存与思考记录)

## 项目简介

本项目使用 PyTorch 对卫星图像进行语义分割，覆盖从数据预处理、训练、推理到多模型指标对比的完整流程。项目支持训练日志记录，并重点关注边界分割质量、小目标丢失以及类别不均衡等问题。

## 致谢与来源

This project is based on [Semantic-segmentation-with-PyTorch-Satellite-Imagery](https://github.com/JenAlchimowicz/Semantic-segmentation-with-PyTorch-Satellite-Imagery) by [Jen Alchimowicz](https://github.com/JenAlchimowicz).

感谢原作者的开源工作。

## 数据集

数据集链接：

https://figshare.com/collections/semantic_segmentation_satellite_imagery/6026765

## 运行流程

1. **数据预处理**：运行 `dataset_splitter.py`
2. **修改配置**：在 `config.py` 中修改必要参数
3. **训练模型**：运行 `train.py`
4. **模型推理**：运行 `predict.py`
5. **指标对比**：运行 `tabulate_data.py`，对比不同模型推理后的评价指标

```bash
# 1. 数据预处理
python dataset_splitter.py

# 2. 修改 config.py 中的参数

# 3. 训练模型
python train.py

# 4. 推理
python predict.py

# 5. 对比不同模型推理后的评价指标
python tabulate_data.py
```

> 注意：运行前请根据本地环境修改 `config.py` 中的数据集路径、类别数、模型参数等配置。

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `dataset_splitter.py` | 加载原始图片和掩码路径，定义预处理变换，划分训练集和验证集；采用五裁剪方法对少数类进行增强 |
| `dataset.py` | 定义 `DataLoader`，并对训练数据集进行图片增强，包括亮度调整、水平/垂直翻转、旋转 |
| `model.py` | 定义训练模型、预训练权重、类别等参数 |
| `config.py` | 存放项目运行所需的必要参数 |
| `utils.py` | 定义评价指标函数 |
| `train.py` | 训练模型，并记录训练日志 |
| `predict.py` | 模型推理部分 |
| `tabulate_data.py` | 对比不同模型推理后的评价指标 |
| 运行输出 | 代码运行后保存的结果、日志或预测文件 |
| 思考记录 | 项目实验过程中的思考与总结 |

## 实验结果与思考

### 结果摘要

- **DeepLabV3Plus** 在边界质量上表现最优：`Boundary_F1: 0.7142`
- 三个模型都面临**小目标丢失**和**类别不均衡**问题
- 增加训练轮次后，各类模型均出现不同程度的**过拟合**
- 过拟合后，小目标漏检问题进一步加剧

### 未来改进

- 引入 **Dice Loss + CE 混合损失**
- 增加小目标样本的**过采样**
- 进一步缓解类别不均衡问题
- 尝试更多针对小目标的训练策略与后处理方法

## 结果保存与思考记录

- 代码运行后，训练日志、推理结果和模型指标对比结果会保存到对应输出目录中。
- “我的思考记录”用于记录实验过程中的观察、问题和后续改进方向。

## TODO

- [ ] 引入 Dice Loss + CE 混合损失
- [ ] 增加小目标样本过采样
- [ ] 缓解过拟合，提升小目标召回
- [ ] 进一步完善多模型评价指标对比

---

如果这个项目对你有帮助，欢迎 Star / Fork。
