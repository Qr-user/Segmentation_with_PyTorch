
import os
import glob
import pandas as pd
import numpy as np

root = os.getcwd()

MODELS = ["PSPNet", "Unet", "DeepLabV3Plus"]

models_dir = os.path.join(root, "models")

DataFrame = []
for m in MODELS:
    csv_path = os.path.join(models_dir, m, f"{m}_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"找不到{csv_path}")
        continue
    df = pd.read_csv(csv_path)
    DataFrame.append(df)
    print(f"读取{m}--成功")

summary = pd.concat(DataFrame, ignore_index=True)
summary = summary.sort_values("mIoU", ascending=False)

output_dir = os.path.join(root, "models", "metrics_summary.csv")
summary.to_csv(output_dir, index=False)
print("保存陈功")