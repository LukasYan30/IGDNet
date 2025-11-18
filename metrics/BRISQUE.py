import os
import pyiqa
import torch
import imageio
import numpy as np
from tqdm import tqdm
import cv2

def imread(f):
    if f.endswith('png'):
        return imageio.imread(f, format="PNG-PIL", ignoregamma=True)
    else:
        return imageio.imread(f)

def load_imgs(path, target_shape=(512, 512)):
    path = os.path.expanduser(path)
    imgfiles = [os.path.join(path, f) for f in sorted(os.listdir(path)) if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]
    
    imgs = []
    for f in imgfiles:
        img = imread(f)[..., :3] / 255.
        # 检查图像形状
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.resize(img, target_shape)  # resize 图像
            imgs.append(img)

    if len(imgs) == 0:
        raise ValueError("No valid images found in the directory")

    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    imgs = imgs.astype(np.float32)
    imgs = torch.tensor(imgs).cuda()
    return imgs

def evaluate(img_path):
    imgs = load_imgs(img_path)
    imgs = imgs.permute(0, 3, 1, 2)
    brisque = pyiqa.create_metric('brisque')
    brisque_score = brisque(imgs)

    mean_brisque = torch.mean(brisque_score)
    print('brisque:low ||', mean_brisque)

if __name__ == '__main__':
    img_path = r"C:\Users\28584\Desktop\unpair_dataset\RRDresult"
    evaluate(img_path)