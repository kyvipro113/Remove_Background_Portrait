import cv2

def equal_hist_img_color(img: cv2.Mat):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    img_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    return img_eq