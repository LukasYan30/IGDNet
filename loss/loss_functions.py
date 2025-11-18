import torch
import torch.nn as nn
import torch.nn.functional as F
import conf
import numpy as np

def reconstruction_loss(image, illumination, reflectance, noise): # image是输入低照度图像，ref是反射分量，ill是光照分量
    reconstructed_image = illumination * reflectance + noise 
    return torch.norm(image - reconstructed_image, 1)

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
    gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
    gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
    gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
    gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
    gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
    return gradient_h*gradient2_h, gradient_w*gradient2_w


def normalize01(img):
    minv = img.min()
    maxv = img.max()
    return (img-minv)/(maxv-minv)


def gaussianblur3(input):
    slice1 = F.conv2d(input[:,0,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    slice2 = F.conv2d(input[:,1,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    slice3 = F.conv2d(input[:,2,:,:].unsqueeze(1), weight=conf.gaussian_kernel, padding=conf.g_padding)
    x = torch.cat([slice1,slice2, slice3], dim=1)
    return x


def illumination_smooth_loss(image, illumination):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:]
    max_rgb, _ = torch.max(image, 1)
    max_rgb = max_rgb.unsqueeze(1)
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    weight_h = 1/(F.conv2d(gradient_gray_h, weight=conf.gaussian_kernel, padding=conf.g_padding)+0.0001)
    weight_w = 1/(F.conv2d(gradient_gray_w, weight=conf.gaussian_kernel, padding=conf.g_padding)+0.0001)
    weight_h.detach()
    weight_w.detach()
    loss_h = weight_h * gradient_illu_h
    loss_w = weight_w * gradient_illu_w
    max_rgb.detach()
    return loss_h.sum() + loss_w.sum() + torch.norm(illumination-max_rgb, 1)


def reflectance_smooth_loss(image, illumination, reflectance):
    gray_tensor = 0.299*image[0,0,:,:] + 0.587*image[0,1,:,:] + 0.114*image[0,2,:,:] #这一行代码将输入的彩色图像 image 转换为灰度图像，使用了RGB通道加权求和的方式得到灰度值。
    gradient_gray_h, gradient_gray_w = gradient(gray_tensor.unsqueeze(0).unsqueeze(0))#这里计算了灰度图像 gray_tensor 在水平和垂直方向上的梯度，gradient 函数可能是用来计算梯度的子函数。
    gradient_reflect_h, gradient_reflect_w = gradient(reflectance) #同样地，计算了反射率 reflectance 在水平和垂直方向上的梯度。
    weight = 1/(illumination*gradient_gray_h*gradient_gray_w+0.0001) #根据灰度图像梯度、光照值和常数项计算了权重，用于平衡反射率梯度的影响。
    weight = normalize01(weight) #对权重进行了归一化处理，使其范围在 [0, 1] 之间。
    weight.detach() #将权重张量从计算图中分离，使其在反向传播时不被更新。
    loss_h = weight * gradient_reflect_h #计算了基于权重的反射率在水平和垂直方向上的平滑损失。
    loss_w = weight * gradient_reflect_w 
    refrence_reflect = image/illumination #计算了参考的反射率，这里将彩色图像除以光照值得到。
    refrence_reflect.detach() #同样地，将参考的反射率张量从计算图中分离
    return loss_h.sum() + loss_w.sum() + conf.reffac*torch.norm(refrence_reflect - reflectance, 1)  


def noise_loss(image, illumination, reflectance, noise):
    weight_illu = illumination
    weight_illu.detach()
    loss = weight_illu*noise
    return torch.norm(loss, 2)


class L_exp(nn.Module):

    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        # self.mean_val = mean_val
    def forward(self, x, mean_val ):

        #b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        meanTensor = torch.FloatTensor([mean_val] ).to(conf.device)

        d = torch.mean(torch.pow(mean- meanTensor,2))
        return d
