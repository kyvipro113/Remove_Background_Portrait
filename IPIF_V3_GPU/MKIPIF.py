from multiprocessing import freeze_support
import time
import yaml
from uvicorn import run
import ssl
import re

from Processing_Image_Interface.app.api import *

# if __name__ == "__main__":
#     freeze_support()
#     with open("config.yaml", "r") as file:
#         data = yaml.load(stream=file, Loader=yaml.Loader)

#     config_IPIF_Service = data["IPIF_SERVICE"]
#     IPIF_HOST = config_IPIF_Service["IPIF_HOST"]
#     IPIF_PORT = config_IPIF_Service["IPIF_PORT"]
#     SECURE = config_IPIF_Service["SECURE"]

#     if(SECURE):
#         # SSL
#         config_SSL = data["SSL"]
#         KEY = config_SSL["KEY"]
#         CERT = config_SSL["CERT"]
#         CA_CERT = config_SSL["CA_CERT"]
#         run(app="Processing_Image_Interface.app.api:app", host=IPIF_HOST, port=IPIF_PORT, workers=1, ssl_certfile=CERT, ssl_keyfile=KEY, ssl_ca_certs=CA_CERT)
#     else:
#         run(app="Processing_Image_Interface.app.api:app", host=IPIF_HOST, port=IPIF_PORT, workers=1)


def is_valid_ip(ip):
    pattern = re.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    return pattern.match(ip) is not None

def is_boolean(value):
    return isinstance(value, bool)

def check_file_exists(file_path):
    return os.path.isfile(file_path)

def check_H_W(H, W):
    if(type(H) == int and type(W) == int):
        return True
    if(type(H) == str and type(W) == int):
        if H == "Flex":
            return True
        print("HEIGHT must be Flex or int")
        return False
    if(type(H) == int and type(W) == str):
        if W == "Flex":
            return True
        print("WIDTH must be Flex or int")
        return False
    print("HEIGHT and WIDTH must be integer or str (Flex). If use deeplabv3 H = W = 512, If use u2net H = W <= 256 or Flex")
    return False

def check_config(data):
    try:
        image_processing_config = data["IMAGE_PROCESSING_CONFIG"]
        NUM_PROCESSES = image_processing_config["NUM_PROCESSES"]
        if type(NUM_PROCESSES) != int:
            print("NUM_PROCESSES must be integer")
            return False
        IMG_SIZE_INP = image_processing_config["IMG_SIZE_INP"]
        if type(IMG_SIZE_INP) != str:
            print("IMG_SIZE_INP must be string (Large or Small)")
            return False
        segmentation_model_config = image_processing_config["SEGMENTATION_MODEL"]
        MODEL_NAME_SEG = segmentation_model_config["MODEL_NAME_SEG"]
        if(MODEL_NAME_SEG != "deeplabv3" and MODEL_NAME_SEG != "u2net" and MODEL_NAME_SEG != "portrait"):
            print("MODEL_NAME_SEG only support for (deeplabv3, u2net or portrait)")
            return False
        MODEL_PATH_SEG = segmentation_model_config["MODEL_PATH_SEG"]
        if not check_file_exists(MODEL_PATH_SEG):
            print("MODEL_PATH_SEG is not found")
            return False
        TENSOR_FORMAT = segmentation_model_config["TENSOR_FORMAT"]
        if MODEL_NAME_SEG == "deeplabv3":
            if TENSOR_FORMAT != "NHWC":
                print(f"TENSOR_FORMAT not match with model {MODEL_NAME_SEG}. It must be use channel last (NHWC)")
                return False
        if MODEL_NAME_SEG == "u2net" or MODEL_NAME_SEG == "portrait":
            if TENSOR_FORMAT != "NCHW":
                print(f"TENSOR_FORMAT not match with model {MODEL_NAME_SEG}. It must be use channel first (NCHW)")
                return False
        H = segmentation_model_config["HEIGHT"]
        W = segmentation_model_config["WIDTH"]
        if not check_H_W(H=H, W=W):
            return False
        alpha_matting_model_config = image_processing_config["ALPHA_MATTING_MODEL"]
        MODEL_NAME_AL = alpha_matting_model_config["MODEL_NAME_AL"]
        if(MODEL_NAME_AL != "mgmatting"):
            print("MODEL_NAME_AL only support for (mgmatting)")
            return False
        MODEL_PATH_AL = alpha_matting_model_config["MODEL_PATH_AL"]
        if not check_file_exists(MODEL_PATH_AL):
            print("MODEL_PATH_AL is not found")
            return False
        AL_USE_GPU = alpha_matting_model_config["USE_GPU"]
        if not is_boolean(AL_USE_GPU):
            print("USE_GPU must be bool (True or False)")
            return False
        OMP_NUM_THREADS = alpha_matting_model_config["OMP_NUM_THREADS"]
        if type(OMP_NUM_THREADS) != int:
            if OMP_NUM_THREADS != "INF":
                print("OMP_NUM_THREADS must be integer or 'INF'")
                return False
        EQUAL_HIST = image_processing_config["EQUAL_HIST"]
        if not is_boolean(EQUAL_HIST):
            print("EQUAL_HIST must be bool (True or False)")
            return False
        GRAY_SCALE = image_processing_config["GRAY_SCALE"]
        if not is_boolean(GRAY_SCALE):
            print("GRAY_SCALE must be bool (True or False)")
            return False
        contrast_adapt_config = image_processing_config["CONTRAST"]
        USE_ADAPT = contrast_adapt_config["USE_ADAPT"]
        if not is_boolean(USE_ADAPT):
            print("USE_ADAPT must be bool (True or False)")
            return False
        MIN_THRESH = contrast_adapt_config["MIN_THRESH"]
        MAX_THRESH = contrast_adapt_config["MAX_THRESH"]
        if type(MIN_THRESH) != int or type(MAX_THRESH) != int:
            print("MIN_THRESH and MAX_THRESH must be integer. Recommend 60 <= MIN_THRESH <= 100, 120 < MAX_THRESH < 210")
            return False
        ALPHA = contrast_adapt_config["ALPHA"]
        BETA = contrast_adapt_config["BETA"]
        if type(ALPHA) != int or type(BETA) != int:
            print("ALPHA and BETA must be integer. Recommend 1 <= ALPHA <=3. -127 <= BETA <= 127")
            return False
    except Exception as e:
        print(f"Error validation config. {str(e)}")
        return False
    return True

if __name__ == "__main__":
    freeze_support()
    is_check = True
    with open("config.yaml", "r") as file:
        data = yaml.load(stream=file, Loader=yaml.Loader)

    config_IPIF_Service = data["IPIF_SERVICE"]
    IPIF_HOST = config_IPIF_Service["IPIF_HOST"]
    IPIF_PORT = config_IPIF_Service["IPIF_PORT"]
    SECURE = config_IPIF_Service["SECURE"]
    KEY: str
    CERT: str
    CA_CERT: str

    if not is_valid_ip(IPIF_HOST):
        print("Validattion IP Address Error")
        is_check = False
    if not type(IPIF_PORT) == int:
        print("Validation Port Error")
        is_check = False
    if not is_boolean(SECURE):
        print("Validation Secure Error")
        is_check = False    

    if(SECURE):
        config_SSL = data["SSL"]
        KEY = config_SSL["KEY"]
        CERT = config_SSL["CERT"]
        CA_CERT = config_SSL["CA_CERT"]
        if not (check_file_exists(KEY) and check_file_exists(CERT) and check_file_exists(CA_CERT)):
            print("KEY or CERT or CA_CERT not found")
            is_check = False

    if not check_config(data=data):
        is_check = False

    if(is_check):
        if(SECURE):
            # SSL
            config_SSL = data["SSL"]
            KEY = config_SSL["KEY"]
            CERT = config_SSL["CERT"]
            CA_CERT = config_SSL["CA_CERT"]
            run(app="Processing_Image_Interface.app.api:app", host=IPIF_HOST, port=IPIF_PORT, workers=1, ssl_certfile=CERT, ssl_keyfile=KEY, ssl_ca_certs=CA_CERT)
        else:
            run(app="Processing_Image_Interface.app.api:app", host=IPIF_HOST, port=IPIF_PORT, workers=1)
    else:
        print("Config Error!")


