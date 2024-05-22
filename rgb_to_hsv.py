from torchvision.utils import save_image
from PIL import Image
from torchvision import transforms
import torch
import numpy as np

img_cover = Image.open('/data/tl/code/PrintImage/Project1/data/template/raw/im1314.jpg').convert('RGB')
img_totensor=transforms.ToTensor()
img_toresize=transforms.Resize((400,400),interpolation=Image.BILINEAR)
img_cover=img_toresize(img_cover)
img_cover=img_totensor(img_cover)
def rgb_hsv(image):
    B,C,H,W=image.shape
    hue = torch.zeros(image.shape[0], image.shape[2], image.shape[3]).to(image.device)
    eps=1e-7
    mask_h_r=(image[:,0,:,:]==image.max(1)[0]).int()
    hue+=((60 * ((image[:,0,:,:]-image[:,1,:,:])/( image.max(1)[0] - image.min(1)[0] +eps)) + 360) % 360)*mask_h_r
    mask_h_g=(image[:,1,:,:]==image.max(1)[0]).int()
    hue+=((60 * ((image[:,0,:,:]-image[:,1,:,:])/( image.max(1)[0] - image.min(1)[0] +eps)) + 120) % 360)*mask_h_g
    mask_h_b=(image[:,2,:,:]==image.max(1)[0]).int()
    hue+=((60 * ((image[:,0,:,:]-image[:,1,:,:])/( image.max(1)[0] - image.min(1)[0] +eps)) + 240) % 360)*mask_h_b
    
    saturation = ( image.max(1)[0] - image.min(1)[0] ) / ( image.max(1)[0] + eps )
    mask_s=(image.max(1)[0]==0).int()
    saturation=saturation*(1-mask_s)
    value = image.max(1)[0]
    
    return hue,saturation,value
##img_cover:C,H,W
hue,saturation,value=rgb_hsv(img_cover.unsqueeze(0).repeat(2,1,1,1))
