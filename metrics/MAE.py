import numpy as np
import argparse
import torch
from glob import glob
from ntpath import basename
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--data-path', default=r'C:\Users\28584\Desktop\our\v1+v2' , help='Path to ground truth data', type=str)
    parser.add_argument('--output-path', default=r'C:\Users\28584\Desktop\lolv1v2\our\best+v1v2project+result\result', help='Path to output data', type=str)
    parser.add_argument('--debug', default=0, help='Debug', type=int)
    args = parser.parse_args()
    return args

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

args = parse_args()
for arg in vars(args):
    print('[%s] =' % arg, getattr(args, arg))

path_true = args.data_path
path_pred = args.output_path

mae = []
names = []

files = list(glob(path_true + '/*/*.jpg')) + list(glob(path_true + '/*/*.png')) + list(glob(path_true + '/*.jpg')) + list(glob(path_true + '/*.png'))

for fn in sorted(files):
    name = basename(str(fn))
    names.append(name)
    
    img_gt = (cv2.imread(str(fn), cv2.IMREAD_COLOR) / 255.0).astype(np.float32)
    img_pred = (cv2.imread(path_pred + '/' + basename(str(fn)), cv2.IMREAD_COLOR) / 255.0).astype(np.float32)

     # 如果预测的图像尺寸与真实图像不同，则将其调整为相同尺寸
    if img_pred.shape[:2] != img_gt.shape[:2]:
        img_pred = cv2.resize(img_pred, (img_gt.shape[1], img_gt.shape[0]))

    mae.append(compare_mae(img_gt, img_pred))

print(
    "MAE: %.4f" % round(np.mean(mae), 4),
)
