import cv2
from PIL import Image, ImageDraw
import numpy as np
from typing import Any

def append_oval_mask(detector: Any, raw_image: Image.Image, remove_bk_image: Image.Image, percent_padding_x: float, percent_padding_y: float):
    img = np.array(raw_image)
    faces = detector(img)
    if len(faces) > 0:
        max_area = 0
        largest_face = None
        for face in faces:
            face_area = face.width() * face.height()
            if face_area > max_area:
                max_area = face_area
                largest_face = face

        left, top, right, bottom = largest_face.left(), largest_face.top(), largest_face.right(), largest_face.bottom()
        mask = Image.new('L', remove_bk_image.size, 0)
        draw = ImageDraw.Draw(mask)
        padding_x = int((right - left) * percent_padding_x)
        padding_y = int((bottom - top) * percent_padding_y)

        draw.ellipse((left - padding_x, top - padding_y, right + padding_x, bottom + padding_y), fill=255)
        oval_image = Image.composite(remove_bk_image, Image.new("RGBA", remove_bk_image.size), mask)
        bbox = mask.getbbox()
        cropped_image = oval_image.crop(bbox)
        return cropped_image
    else:
        return remove_bk_image
    
def append_oval_mask(remove_bk_image: Image.Image):
    w, h = remove_bk_image.size
    left, top, right, bottom = 0, 0, w, h
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((left, top, right, bottom), fill=255)
    oval_image = Image.composite(remove_bk_image, Image.new("RGBA", remove_bk_image.size), mask)
    bbox = mask.getbbox()
    cropped_image = oval_image.crop(bbox)
    return cropped_image

    

