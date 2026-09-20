from utils import *
import os
import time
import numpy as np
import glob
from PIL import Image
from tqdm import tqdm
from scipy import ndimage
from model import *


root = os.getcwd()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = os.path.join(root, 'models', Model)
best_model_path = os.path.join(SAVE_DIR, f'{Model}_res101_368_best.pt')

model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print('Best model loaded from', best_model_path)


#  统一调色板：所有模型共用，类别颜色一致，方便对比
NUM_CLASSES = 25
rng = np.random.default_rng(42)  # 用随机种子每次运行生成相同的颜色，方便对比
PALETTE = np.zeros((256 * 3,), dtype=np.uint8)
for c in range(NUM_CLASSES):
    PALETTE[c*3 : c*3+3] = rng.integers(0, 255, size=3)
PALETTE = PALETTE.tolist()

#  真实 mask 目录
MASK_DIR = os.path.join(DATA_ROOT, 'test_mask_368')

# 5) 对测试集逐张推理并保存 PNG
test_img_link_list = sorted(glob.glob(os.path.join(DATA_ROOT, f'test_images_{HEIGHT}', '*')))

PRED_DIR_RAW = os.path.join(root, "models", Model)
PRED_DIR_VIS = os.path.join(root, 'models', Model)
os.makedirs(PRED_DIR_RAW, exist_ok=True)
os.makedirs(PRED_DIR_VIS, exist_ok=True)

model.eval()

# 全局混淆矩阵（累加所有图片）
global_cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
# 边界 F1 列表
boundary_f1_list = []
# 推理时间列表（ms）
infer_times = []

for i in tqdm(range(len(test_img_link_list))):
    img_id = os.path.splitext(os.path.basename(test_img_link_list[i]))[0]
    img = torch.load(test_img_link_list[i], map_location=DEVICE).unsqueeze(0)
    orig_tensor = img.squeeze(0).detach().cpu()

    # 计时开始（预热后稳定计时）
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        output = model(img)

    # 计时结束
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    infer_times.append((t_end - t_start) * 1000)   # 转 ms

    output = torch.argmax(output, dim=1).squeeze(0)   # (H, W)
    output = output.byte().cpu().numpy()              # uint8

    # 1) 原始类别图
    Image.fromarray(output, mode='L').save(
        os.path.join(PRED_DIR_RAW, f'{img_id}.png')
    )

    # 2) 原图 -> PIL RGB
    orig = orig_tensor
    if orig.shape[0] == 3:
        orig = orig.permute(1, 2, 0)
    orig = orig.numpy()

    # 如果训练时用了 ImageNet 归一化，取消下面两行注释
    # mean = np.array([0.485, 0.456, 0.406])
    # std  = np.array([0.229, 0.224, 0.225])
    # orig = orig * std + mean

    orig = np.clip(orig * 255, 0, 255).astype(np.uint8)
    orig_pil = Image.fromarray(orig, mode='RGB')

    # 3) 彩色可视化图
    vis = Image.fromarray(output, mode='L')
    vis.putpalette(PALETTE)
    vis_rgb = vis.convert('RGB')

    # 4) 读 GT mask（.pt 文件）
    mask_path = os.path.join(MASK_DIR, f'{img_id}.pt')
    gt = None
    if os.path.exists(mask_path):
        gt = torch.load(mask_path, map_location='cpu')
        if isinstance(gt, torch.Tensor):
            gt = gt.numpy()
        gt = gt.astype(np.uint8)
        if gt.ndim == 3:
            gt = gt.squeeze()
        gt_pil = Image.fromarray(gt, mode='L')
        gt_pil.putpalette(PALETTE)
        gt_rgb = gt_pil.convert('RGB')
    else:
        gt_rgb = Image.new('RGB', orig_pil.size, (0, 0, 0))

    #  累加指标
    if gt is not None:
        # 混淆矩阵
        global_cm += compute_confusion_matrix(output, gt, NUM_CLASSES)
        # Boundary F1
        boundary_f1_list.append(boundary_f1(output, gt, tolerance=2))

    # 6) 三栏并排
    GAP = 20
    w, h = orig_pil.size
    canvas = Image.new('RGB', (w * 3 + GAP * 2, h), (255, 255, 255))
    canvas.paste(orig_pil, (0, 0))
    canvas.paste(gt_rgb,   (w + GAP, 0))
    canvas.paste(vis_rgb,  (w * 2 + GAP * 2, 0))
    canvas.save(os.path.join(PRED_DIR_VIS, f'{Model}_{img_id}.png'))

print('Raw preds :', PRED_DIR_RAW)
print('Vis  preds:', PRED_DIR_VIS)


# ============================================================
# 汇总所有指标并打印
# ============================================================
print('\n' + '=' * 60)
print(f'评估指标汇总  (Model: {Model})')
print('=' * 60)

metrics = compute_metrics_from_cm(global_cm)
params_m = get_model_params_m(model)
mean_infer_ms = np.mean(infer_times) if infer_times else 0.0
std_infer_ms = np.std(infer_times) if infer_times else 0.0
mean_bf1 = np.mean(boundary_f1_list) if boundary_f1_list else 0.0

print(f'mIoU              : {metrics["miou"]:.4f}')
print(f'Dice / F1         : {metrics["mean_dice"]:.4f}')
print(f'Pixel Acc         : {metrics["pixel_acc"]:.4f}')
print(f'Boundary F1       : {mean_bf1:.4f}')
print(f'参数量 (M)        : {params_m:.2f} M')
print(f'推理时间 (ms)     : {mean_infer_ms:.2f} ± {std_infer_ms:.2f} ms')
print('=' * 60)

#保存指标到 CSV
import pandas as pd
results_df = pd.DataFrame([{
    'Model':          Model,
    'mIoU':           round(metrics['miou'], 4),
    'Dice_F1':        round(metrics['mean_dice'], 4),
    'Pixel_Acc':      round(metrics['pixel_acc'], 4),
    'Boundary_F1':    round(mean_bf1, 4),
    'Params_M':       round(params_m, 2),
    'Infer_ms':       round(mean_infer_ms, 2),
}])
csv_path = os.path.join(SAVE_DIR, f'{Model}_metrics.csv')
results_df.to_csv(csv_path, index=False)
print(f'\n指标已保存到: {csv_path}')

# CHECK: how many predicted images in folder
temp = sorted(glob.glob(os.path.join(PRED_DIR_RAW, '*')))
print('Predicted files:', len(temp))

























