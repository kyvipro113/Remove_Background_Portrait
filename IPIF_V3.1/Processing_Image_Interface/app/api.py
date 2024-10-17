from fastapi import FastAPI, HTTPException, Request, Response, Depends, Path, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Optional
from datetime import datetime, timedelta
import yaml
import json
import os
from pydantic import BaseModel
import hashlib
from multiprocessing import *
import time

from Processing_Image_Interface.utilities.Remove_Background_Image import load_settings_image_processing
from Processing_Image_Interface.app.service.Process_Image import *


app = FastAPI(debug=True)

origins = ["*"]  # Replace "*" with the allowed origins, e.g., ["http://localhost", "http://localhost:4200"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configure
with open("config.yaml", "r") as file:
    data = yaml.load(stream=file, Loader=yaml.Loader)

image_processing_config = data["IMAGE_PROCESSING_CONFIG"]
NUM_PROCESSES = image_processing_config["NUM_PROCESSES"]
IMG_SIZE_INP = image_processing_config["IMG_SIZE_INP"]
DATA_IN = image_processing_config["DATA_IN"]
DATA_OUT = image_processing_config["DATA_OUT"]
EQUAL_HIST = image_processing_config["EQUAL_HIST"]
GRAY_SCALE = image_processing_config["GRAY_SCALE"]
USE_OVAL_MASK = image_processing_config["USE_OVAL_MASK"]
PERCENT_PADDING_X = image_processing_config["PERCENT_PADDING_X"]
PERCENT_PADDING_Y = image_processing_config["PERCENT_PADDING_Y"]

segmentation_model_config = image_processing_config["SEGMENTATION_MODEL"]
MODEL_NAME_SEG = segmentation_model_config["MODEL_NAME_SEG"]
MODEL_PATH_SEG = segmentation_model_config["MODEL_PATH_SEG"]
TENSOR_FORMAT = segmentation_model_config["TENSOR_FORMAT"]
H = segmentation_model_config["HEIGHT"]
W = segmentation_model_config["WIDTH"]
USE_PADDING = None
PADDING_VALUE = None
SCALE = None
MEAN = None
STDEV = None
if MODEL_NAME_SEG == "portrait":
    USE_PADDING = segmentation_model_config["USE_PADDING"]
    PADDING_VALUE = segmentation_model_config["PADDING_VALUE"]
    SCALE = segmentation_model_config["SCALE"]
    MEAN = segmentation_model_config["MEAN"]
    STDEV = segmentation_model_config["STDEV"]

alpha_matting_model_config = image_processing_config["ALPHA_MATTING_MODEL"]
MODEL_NAME_AL = alpha_matting_model_config["MODEL_NAME_AL"]
MODEL_PATH_AL = alpha_matting_model_config["MODEL_PATH_AL"]
AL_USE_GPU = alpha_matting_model_config["USE_GPU"]
OMP_NUM_THREADS = alpha_matting_model_config["OMP_NUM_THREADS"]

contrast_adapt_config = image_processing_config["CONTRAST"]
USE_ADAPT = contrast_adapt_config["USE_ADAPT"]
MIN_THRESH = contrast_adapt_config["MIN_THRESH"]
MAX_THRESH = contrast_adapt_config["MAX_THRESH"]
ALPHA = contrast_adapt_config["ALPHA"]
BETA = contrast_adapt_config["BETA"]

load_settings_image_processing(model_name_seg=MODEL_NAME_SEG, model_path_seg=MODEL_PATH_SEG, tensor_format=TENSOR_FORMAT, H=H, W=W, 
                               use_padding=USE_PADDING, padding_value=PADDING_VALUE, scale=SCALE, mean=MEAN, stdev=STDEV,
                               model_path_al=MODEL_PATH_AL, al_use_gpu=AL_USE_GPU, equal_hist=EQUAL_HIST, gray_scale=GRAY_SCALE, 
                               use_oval_mask=USE_OVAL_MASK, percent_padding_x=PERCENT_PADDING_X, percent_padding_y=PERCENT_PADDING_Y,
                               use_adapt_contrast=USE_ADAPT, min_thresh=MIN_THRESH, max_thresh=MAX_THRESH, ct_alpha=ALPHA, ct_beta=BETA)
load_img_config(img_size_inp=IMG_SIZE_INP, omp_num_threads=OMP_NUM_THREADS, data_in=DATA_IN, data_out=DATA_OUT)

@app.post("/process-image")
async def process_image(data_if: Data_Info_Model):
    start_time = time.time()
    res_data = processImage(data_if=data_if, num_processes=NUM_PROCESSES)
    end_time = time.time()
    time_execute_all = end_time - start_time
    print(f"Execute time: {time_execute_all}s. Start time: {start_time}, end time: {end_time}")
    return res_data