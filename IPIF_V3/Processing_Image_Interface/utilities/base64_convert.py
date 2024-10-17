import cv2
import base64
import numpy as np

def convertBase64ToByte(data: str)->bytes:
    return base64.b64decode(data)

def convertMat2Base64(image: cv2.Mat)->str:
    _, buffer = cv2.imencode('.png', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return base64_string

def convertBase64ToMat(data: str)->cv2.Mat:
    data_byte = base64.b64decode(data)
    I_array = np.frombuffer(data_byte, np.uint8)
    image = cv2.imdecode(I_array, cv2.IMREAD_COLOR)
    return image
