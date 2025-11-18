import math
import cv2
import numpy as np
import os

#### PSNR
def img_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    # compute psnr
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(1 / math.sqrt(mse))
    return psnr

#### SSIM
def img_ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)  # 返回SSIM指标的平均值

#### LOE
def img_loe(ipic, epic, window_size=7):

    def U_feature(image):
        image = cv2.resize(image, (500,500))
        image = np.max(image, axis=2)
        w_half = window_size // 2
        padded_arr = np.pad(image, ((w_half, w_half), (w_half, w_half)), mode='constant')

        local_windows = np.lib.stride_tricks.sliding_window_view(padded_arr, (window_size, window_size))
        local_windows = local_windows.reshape(-1, window_size * window_size)
        relationship = local_windows[:,:,None] > local_windows[:,None,:]
        return relationship.flatten()

    ipic = U_feature(ipic)
    epic = U_feature(epic)

    return np.mean(ipic!=epic)

def metric(gt_image, pred_image):
    #psnr = img_psnr(gt_image, pred_image)
    #ssim = img_ssim(gt_image, pred_image)
    loe = img_loe(gt_image, pred_image)
    #return psnr, ssim, loe
    return loe

gt_folder_path = r'C:\Users\28584\Desktop\our\v1+v2'
pred_folder_path = r'C:\Users\28584\Desktop\lolv1v2\our\best+v1v2project+result\result'

#psnr_sum = 0
#ssim_sum = 0
loe_sum = 0
num_images = 0


# 遍历文件夹中的图片
for filename in os.listdir(gt_folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        gt_image_path = os.path.join(gt_folder_path, filename)
        pred_image_path = os.path.join(pred_folder_path, filename)
        
        # 读取图片
        gt_image = cv2.imread(gt_image_path) / 255.
        pred_image = cv2.imread(pred_image_path) / 255.
        pred_image = cv2.resize(pred_image, (gt_image.shape[1], gt_image.shape[0]))

        # 计算评估指标
       # psnr, ssim, loe = metric(gt_image, pred_image)
        loe = metric(gt_image, pred_image)

        #psnr_sum += psnr
        #ssim_sum += ssim
        loe_sum += loe
        num_images += 1

#avg_psnr = psnr_sum / num_images
#avg_ssim = ssim_sum / num_images
avg_loe = loe_sum / num_images

# 打印结果
#print("Average PSNR:", avg_psnr)
#print("Average SSIM:", avg_ssim)
print("Average LOE:", avg_loe)

