import torch
from torch.nn import functional as F

from Processing_Image_Interface.utilities import utils
#import utils

#from utils import CONFIG
from Processing_Image_Interface.utilities.utils import CONFIG

#import networks
from Processing_Image_Interface.utilities import networks

import numpy as np
import cv2

def single_inference(model, image_dict, use_gpu=False, post_process=False):

    with torch.no_grad():
        image, mask = image_dict['image'], image_dict['mask']
        alpha_shape = image_dict['alpha_shape']
        if use_gpu:
            image = image.cuda()
            mask = mask.cuda()
        pred = model(image, mask)
        alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']

        ### refinement
        alpha_pred = alpha_pred_os8.clone().detach()
        weight_os4 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width1, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = utils.get_unknown_tensor_from_pred(alpha_pred, rand_width=CONFIG.model.self_refine_width2, train_mode=False)
        alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]


        h, w = alpha_shape
        alpha_pred = alpha_pred[0, 0, ...].data.cpu().numpy()
        if post_process:
            alpha_pred = utils.postprocess(alpha_pred)
        alpha_pred = alpha_pred * 255
        alpha_pred = alpha_pred.astype(np.uint8)
        alpha_pred = alpha_pred[32:h+32, 32:w+32]

        return alpha_pred



def generator_tensor_dict(image_path, mask_path, guidance_thres=128):
    # read images
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, 0)

    mask = (mask >= guidance_thres).astype(np.float32) ### only keep FG part of trimap
    
    #mask = mask.astype(np.float32) / 255.0 ### soft trimap

    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, mask = sample['image'][:,:,::-1], sample['mask']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    mask = np.expand_dims(mask.astype(np.float32), axis=0)

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # add first channel
    sample['image'], sample['mask'] = sample['image'][None, ...], sample['mask'][None, ...]

    return sample



def generator_tensor_dict_matrix(image, mask, guidance_thres=128):
    mask = (mask >= guidance_thres).astype(np.float32) ### only keep FG part of trimap
    
    #mask = mask.astype(np.float32) / 255.0 ### soft trimap

    sample = {'image': image, 'mask': mask, 'alpha_shape': mask.shape}

    # reshape
    h, w = sample["alpha_shape"]
    
    if h % 32 == 0 and w % 32 == 0:
        padded_image = np.pad(sample['image'], ((32,32), (32, 32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,32), (32, 32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask
    else:
        target_h = 32 * ((h - 1) // 32 + 1)
        target_w = 32 * ((w - 1) // 32 + 1)
        pad_h = target_h - h
        pad_w = target_w - w
        padded_image = np.pad(sample['image'], ((32,pad_h+32), (32, pad_w+32), (0,0)), mode="reflect")
        padded_mask = np.pad(sample['mask'], ((32,pad_h+32), (32, pad_w+32)), mode="reflect")
        sample['image'] = padded_image
        sample['mask'] = padded_mask

    # ImageNet mean & std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    # convert GBR images to RGB
    image, mask = sample['image'][:,:,::-1], sample['mask']
    # swap color axis
    image = image.transpose((2, 0, 1)).astype(np.float32)

    mask = np.expand_dims(mask.astype(np.float32), axis=0)

    # normalize image
    image /= 255.

    # to tensor
    sample['image'], sample['mask'] = torch.from_numpy(image), torch.from_numpy(mask)
    sample['image'] = sample['image'].sub_(mean).div_(std)

    # add first channel
    sample['image'], sample['mask'] = sample['image'][None, ...], sample['mask'][None, ...]

    return sample