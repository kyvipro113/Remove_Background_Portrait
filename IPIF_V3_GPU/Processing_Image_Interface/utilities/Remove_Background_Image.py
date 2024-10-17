import openvino as ov
import openvino.properties as properties
import openvino.properties.device as device
import openvino.properties.hint as hints
import openvino.properties.streams as streams
import openvino.properties.intel_auto as intel_auto
import openvino.runtime as ov_runtime

import torch
import cv2
import numpy as np
import time
import os
from PIL import Image
from scipy.special import softmax

from Processing_Image_Interface.utilities.MGMatting import *
from Processing_Image_Interface.utilities.base64_convert import *
from Processing_Image_Interface.utilities.equal_hist_img import *
from Processing_Image_Interface.utilities.preprocessing_img import *

from Processing_Image_Interface.logger.Logger import *

class Class_Config(object):
    model_name_seg: str
    model_path_seg: str
    tensor_format: str
    H: int | str
    W: int | str
    image_flex: bool
    use_padding: bool
    padding_value: int
    scale: int
    mean: list
    stdev: list
    model_path_al: str
    al_use_gpu: bool
    num_threads_pytorch: int
    equal_hist: bool = False
    gray_scale: bool = False
    use_adapt_contrast: bool = False
    min_thresh: int
    max_thresh: int
    ct_alpha: int
    ct_beta: int

def load_settings_image_processing(model_name_seg: str, model_path_seg: str, tensor_format: str, H: int, W: int, 
                                   use_padding: bool, padding_value: int, scale: int, mean: list, stdev: list,
                                   model_path_al: str, al_use_gpu: bool, equal_hist: bool, gray_scale: bool,
                                   use_adapt_contrast: bool, min_thresh: int, max_thresh: int, ct_alpha: int, ct_beta: int):
    Class_Config.model_name_seg = model_name_seg
    Class_Config.model_path_seg = model_path_seg
    Class_Config.tensor_format = tensor_format
    Class_Config.H = H
    Class_Config.W = W
    if H == "Flex" or W == "Flex":
        Class_Config.image_flex = True
    else:
        Class_Config.image_flex = False
    Class_Config.use_padding = use_padding
    Class_Config.padding_value = padding_value
    Class_Config.scale = scale
    Class_Config.mean = mean
    Class_Config.stdev = stdev
    Class_Config.model_path_al = model_path_al
    Class_Config.al_use_gpu = al_use_gpu
    Class_Config.equal_hist = equal_hist
    Class_Config.gray_scale = gray_scale
    Class_Config.use_adapt_contrast = use_adapt_contrast
    Class_Config.min_thresh = min_thresh
    Class_Config.max_thresh = max_thresh
    Class_Config.ct_alpha = ct_alpha
    Class_Config.ct_beta = ct_beta


def normalize_channel(channel, mean_target=128, stddev_target=64):
    # Calculate mean and standard deviation
    mean, stddev = cv2.meanStdDev(channel)
    mean = mean[0][0]
    stddev = stddev[0][0]
    # Adjust 
    normalized_channel = (channel - mean) / stddev * stddev_target + mean_target
    return np.clip(normalized_channel, 0, 255).astype(np.uint8)

def process_remove_background_image(data_path: str, img_size_inp: str, lImage: list, data_out_path: str, name_logger_sub_process: str):
    print(f"PID: {os.getpid()}")
    _logger = Logger(folder_name="log_Image_Processing_IF", file_name=name_logger_sub_process, name_logger=name_logger_sub_process, file_mode="a")
    if not os.path.exists(data_out_path):
        os.mkdir(data_out_path)
    equal_hist = Class_Config.equal_hist
    gray_scale = Class_Config.gray_scale
    model_name_seg: str = Class_Config.model_name_seg
    model_path_seg: str = Class_Config.model_path_seg
    tensor_format = Class_Config.tensor_format
    H, W = Class_Config.H, Class_Config.W
    image_flex = Class_Config.image_flex
    use_padding = Class_Config.use_padding
    padding_value = Class_Config.padding_value
    scale = Class_Config.scale
    _mean = Class_Config.mean
    stdev = Class_Config.stdev
    model_path_al = Class_Config.model_path_al
    al_use_gpu = Class_Config.al_use_gpu
    use_adapt_contrast = Class_Config.use_adapt_contrast
    min_thresh = Class_Config.min_thresh
    max_thresh = Class_Config.max_thresh
    alpha = Class_Config.ct_alpha
    beta = Class_Config.ct_beta

    # Load Segmentation Model (OPENVINO)
    core = ov.Core()
    model_ov = core.read_model(model_path_seg)
    compiled_model = core.compile_model(model=model_ov, device_name='CPU')

    # Load Alpha Matting Model (PYTORCH)
    model = networks.get_generator(encoder=CONFIG.model.arch.encoder, decoder=CONFIG.model.arch.decoder)
    if not al_use_gpu:
        checkpoint = torch.load(model_path_al, map_location=torch.device("cpu"))    ## USE CPU
        model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)    
    else:
        model.cuda()
        checkpoint = torch.load(model_path_al)
        model.load_state_dict(utils.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)

    model = model.eval()    ## Evaluate mode

    for img_file in lImage:
        image: cv2.Mat
        img_pil = None
        x: cv2.Mat
        w: int
        h: int
        bbx: None
        if(img_size_inp == "Large"):
            img_pil = Image.open(data_path + "/" + img_file)
            if img_pil.mode != "RGB":
                img_pil = img_pil.convert("RGB")
            w, h = img_pil.size
            if(image_flex):
                if Class_Config.H == "Flex":
                    H = h * W // w
                if Class_Config.W == "Flex":
                    W = w * H // h
            img_pil_x = img_pil.copy()
            x_pil = img_pil_x.resize((W, H), Image.BILINEAR)
            x = np.array(x_pil)
            x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
            image = np.array(img_pil)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if model_name_seg == "portrait":
                x = image.copy()
        else:
            image = cv2.imread(data_path + "/" + img_file , cv2.IMREAD_COLOR)
            h, w, c = image.shape
            if(image_flex):
                if Class_Config.H == "Flex":
                    H = h * W // w
                if Class_Config.W == "Flex":
                    W = w * H // h
            x = cv2.resize(image, (W, H))
            if model_name_seg == "portrait":
                x = image.copy()
        
        # Normalization 
        if model_name_seg != "portrait":
            channels = cv2.split(x)
            normalized_channels = [normalize_channel(ch) for ch in channels]
            x = cv2.merge(normalized_channels)
            x = x/255.0
            x = x.astype(np.float32)
        else:
            if use_padding:
                x, bbx = resize_padding(image=x, dstshape=[H, W], padValue=padding_value)
            x = normalize_img(imgOri=x, scale=scale, mean=_mean, val=stdev)
            print(f"Portrait shape: {x.shape}")

        x = np.expand_dims(x, axis=0)
        if tensor_format != "NHWC":
            x = np.transpose(x, (0, 3, 1, 2))
            print(f"NCHW shape: {x.shape}")

        start_time = time.time()
        # Inference
        ## Inference segmentation
        output_data = compiled_model([x])
        output_np = output_data[next(iter(output_data))]
        if tensor_format != "NHWC":
            output_np = np.transpose(output_np, (0, 2, 3, 1))
        mask: cv2.Mat
        if model_name_seg != "portrait":
            y = output_np[0]
            # y[y > 0.8] = 1
            # y[y < 0.5] = 0
            y = cv2.resize(y, (w, h), cv2.INTER_LINEAR)
            
            mask = np.squeeze(y)
            mask = cv2.convertScaleAbs(mask*255)
        else:
            prob = softmax(output_np, axis=3)
            y = prob[0]
            y = y[:, :, 1]
            if use_padding:
                y = y[bbx[1]:bbx[3], bbx[0]:bbx[2]]
            y [y < 0.6] = 0     ## Test threshsold
            y = cv2.resize(y, (w, h), cv2.INTER_LINEAR)
            mask = cv2.convertScaleAbs(y*255)
            # cv2.imwrite(data_out_path + "/" + img_file.split(".")[0] +"_mask.png", mask)

        ## Concatenate image with mask
        image_dict = generator_tensor_dict_matrix(image=image, mask=mask)
        ## Inference alpha matting
        alpha_pred: np.array = single_inference(model, image_dict, use_gpu=al_use_gpu, post_process=False)

        end_time = time.time()
        infer_time = end_time - start_time
        print(f"Inference Time: {infer_time}s")

        if equal_hist:
            image = equal_hist_img_color(img=image)
        if use_adapt_contrast:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean = np.mean(img_gray)
            if(mean < min_thresh):
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            if(mean > max_thresh):
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=-beta)
        if gray_scale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        I_pil = Image.fromarray(image_rgb)
        mask_refine_pil = Image.fromarray(alpha_pred)

        I_pil.putalpha(mask_refine_pil)

        image_remove_bk = np.array(I_pil)
        image_remove_bk = cv2.cvtColor(image_remove_bk, cv2.COLOR_RGBA2BGRA)

        cv2.imwrite(data_out_path + "/" + img_file.split(".")[0] +".png", image_remove_bk)
        _logger.trace(f"{img_file} file processing successful. Inference time: {infer_time}s")