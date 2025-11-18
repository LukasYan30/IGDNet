import os
os.environ["CUDA_VISIBLE_DEVICES"] = "8"
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from model.RRDNet import RRDNet
from loss.loss_functions import reconstruction_loss, illumination_smooth_loss, reflectance_smooth_loss, noise_loss, normalize01
import conf
import torch
import os
from model.ZS_NSN import Image_denoise
import matplotlib.pyplot as plt


def pipline_retinex(net, img):

    img_tensor = transforms.ToTensor()(img)  # [c, h, w] 
    img_tensor = img_tensor.to(conf.device)
    img_tensor = img_tensor.unsqueeze(0)     # [1, c, h, w] 
    optimizer = optim.Adam(net.parameters(), lr=conf.lr) 

    # iterations
    for i in range(conf.iterations+1): 
       
        illumination, reflectance, noise = net(img_tensor) 

        adjust_illu = torch.pow(illumination, conf.gamma)  
        #res_image = adjust_illu*((img_tensor-noise)/illumination)
        #res_image = torch.clamp(res_image, min=0, max=1) 

       
        loss_recons = reconstruction_loss(img_tensor, illumination, reflectance, noise)
        loss_illu = illumination_smooth_loss(img_tensor, illumination) 
        loss_reflect = reflectance_smooth_loss(img_tensor, illumination, reflectance) 
        loss_noise = noise_loss(img_tensor, illumination, reflectance, noise) 

        loss = loss_recons + conf.illu_factor*loss_illu + conf.reflect_factor*loss_reflect + conf.noise_factor * loss_noise 

        # backward
        net.zero_grad()  

        loss.backward()  
        optimizer.step() 

        # log
        if i%100 == 0:
           print("iter:", i, '  reconstruction loss:', float(loss_recons.data), '  illumination loss:', float(loss_illu.data), '  reflectance loss:', float(loss_reflect.data), '  noise loss:', float(loss_noise.data))



    adjust_illu = torch.pow(illumination, conf.gamma)  
    res_image = adjust_illu*((img_tensor-noise)/illumination)
    res_image = torch.clamp(res_image, min=0, max=1) 


    if conf.device != 'cpu':
        res_image = res_image.cpu()
        illumination = illumination.cpu()
        adjust_illu = adjust_illu.cpu()
        reflectance = reflectance.cpu()
        noise = noise.cpu()

    res_img = transforms.ToPILImage()(res_image.squeeze(0)) 
    illum_img = transforms.ToPILImage()(illumination.squeeze(0))
    adjust_illu_img = transforms.ToPILImage()(adjust_illu.squeeze(0))
    reflect_img = transforms.ToPILImage()(reflectance.squeeze(0))
    noise_img = transforms.ToPILImage()(normalize01(noise.squeeze(0)))

    return res_img, illum_img, adjust_illu_img, reflect_img, noise_img

def calc_gamma_param(pixel_value, x):
    if pixel_value > 0.5:
       gamma = x + (2**( pixel_value- x )-1)
       #gamma=pixel_value
    else:
      gamma = x
    return gamma

def adaptive_gamma_correction(illumination_img, target_img, x):
    illumination_normalized = np.array(illumination_img).astype(float) / 255.0
    target_img = np.array(target_img).astype(float) / 255.0
    target = np.array(target_img)  
  
    corrected_img = np.zeros_like(target)

    # 对每个像素进行伽马校正
    for i in range(target.shape[0]):
      for j in range(target.shape[1]):  
        pixel_value = illumination_normalized[i, j]
            # 计算伽马参数
        gamma = calc_gamma_param(pixel_value, x)
            
            # 进行伽马校正
        corrected_img[i, j] = np.power(target[i, j].astype(np.float32), gamma)

    temp = corrected_img - np.min(corrected_img)  
    res_img = (temp/np.max(temp))*255   
    res_img = Image.fromarray(res_img.astype(np.uint8)) 
    return res_img


if __name__ == '__main__':

    net = RRDNet()
    net = net.to(conf.device)
    

    results_folder = './test'
    result_folders = ['result', 'illumination', 'adjust_illumination', 'reflectance', 'noise_map']
    for folder in result_folders:
        folder_path = os.path.join(results_folder, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_list = os.listdir(conf.test_image_path) 
    for f in file_list:
        file_path = os.path.join(conf.test_image_path, f)
        img = Image.open(file_path)
        #img_low = cv2.imread(file_path)
        #img_low = np.array(img) / 255.0
        #img = 1 - img_low

        res_img, illum_img, adjust_illu_img, reflect_img, noise_img = pipline_retinex(net, img)

        illum_img.save(os.path.join(results_folder, 'illumination', f))
        adjust_illu_img.save(os.path.join(results_folder, 'adjust_illumination', f))
        reflect_img.save(os.path.join(results_folder, 'reflectance', f))
        noise_img.save(os.path.join(results_folder, 'noise_map', f))
        res_img.save(os.path.join(results_folder, 'RRDresult', f))


        x = 0.5 # 自适应参数
        result_img = adaptive_gamma_correction(illum_img, res_img, x)
        
        res_img = Image_denoise(result_img)
        plt.imsave(os.path.join(results_folder, 'result', f), res_img)
    
        
