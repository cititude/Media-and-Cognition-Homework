import torch 
import json
from utils.utils import *
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def main():
    if not os.path.exists("imgs"):
        os.mkdir("imgs")
    jsonfile=json.load(open("filter_unique_pred17.json"))
    imgs=jsonfile["imgs"]
    for idx in imgs:
        img=imgs[idx]
        bboxes=img["objects"]
        output=[]

        mosaic = np.full((2048,2048, 3), 255, dtype=np.uint8)

        # Fix class - colour map
        prop_cycle = plt.rcParams['axes.prop_cycle']

        hex2rgb = lambda h: tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
        color_lut = [hex2rgb(h) for h in prop_cycle.by_key()['color']]
        img=np.array(Image.open(os.path.join("../image_exp/Detection/test",idx+".jpg")).convert('RGB'))
        mosaic = img
        if(len(bboxes)>8):
            print(idx)
        else:
            continue
        for bbox in bboxes:
            box=np.array([bbox["bbox"]["xmin"],bbox["bbox"]["ymin"],bbox["bbox"]["xmax"],bbox["bbox"]["ymax"]])
            class_=bbox["category"]
            conf=bbox["score"]
            color = color_lut[hash(class_) % len(color_lut)]
            plot_one_box(box, mosaic, label=class_, color=color, line_thickness=2)



        # Image border
        cv2.rectangle(mosaic, (1, 1), (2047, 2047), (255, 255, 255), thickness=3)
        cv2.imwrite("imgs/{}.jpg".format(idx), cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB),)

main()