import torch
import torchvision
import argparse
import itertools
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import os
import numpy as np
import random
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from torch.autograd import Variable
import matplotlib.cm as mpl_color_map
import matplotlib.pyplot as plt
import copy

def save_class_activation_images(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('./heatmaps'):
        os.makedirs('./heatmaps')
    # Grayscale activation map
    heatmap, heatmap_on_image = apply_colormap_on_image(org_img, activation_map, 'hsv')
    # Save colored heatmap
    path_to_file = os.path.join('./heatmaps', file_name+'_Cam_Heatmap.png')
    save_image(heatmap, path_to_file)
    # Save heatmap on iamge
    path_to_file = os.path.join('./heatmaps', file_name+'_Cam_On_Image.png')
    save_image(heatmap_on_image, path_to_file)
    # SAve grayscale heatmap
    path_to_file = os.path.join('./heatmaps', file_name+'_Cam_Grayscale.png')
    save_image(activation_map, path_to_file)
    path_to_file = os.path.join('./heatmaps', file_name+'_origin.png')
    org_img.save(path_to_file)


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    if type(org_im) is np.ndarray:
        org_im=Image.fromarray(org_im)
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image

def format_np_output(np_arr):
    """
        This is a (kind of) bandaid fix to streamline saving procedure.
        It converts all the outputs to the same format which is 3xWxH
        with using sucecssive if clauses.
    Args:
        im_as_arr (Numpy array): Matrix of shape 1xWxH or WxH or 3xWxH
    """
    # Phase/Case 1: The np arr only has 2 dimensions
    # Result: Add a dimension at the beginning
    if len(np_arr.shape) == 2:
        np_arr = np.expand_dims(np_arr, axis=0)
    # Phase/Case 2: Np arr has only 1 channel (assuming first dim is channel)
    # Result: Repeat first channel and convert 1xWxH to 3xWxH
    if np_arr.shape[0] == 1:
        np_arr = np.repeat(np_arr, 3, axis=0)
    # Phase/Case 3: Np arr is of shape 3xWxH
    # Result: Convert it to WxHx3 in order to make it saveable by PIL
    if np_arr.shape[0] == 3:
        np_arr = np_arr.transpose(1, 2, 0)
    # Phase/Case 4: NP arr is normalized between 0-1
    # Result: Multiply with 255 and change type to make it saveable by PIL
    if np.max(np_arr) <= 1:
        np_arr = (np_arr*255).astype(np.uint8)
    return np_arr

    
class Cam():
    """
        Extracts cam features from the model
    """
    def __init__(self, model,target_class=None):
        self.model = model.module if hasattr(model,"module") else model
        self.gradients = None
        self.target_class=target_class

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        for i,(module_pos, module) in enumerate(self.model._modules.items()):
            if module_pos =="maxpool":
                continue
            if module_pos == "fc":
                continue
            if module_pos=="classifier":
                continue
            x = module(x)  # Forward
        return x

    def forward_pass(self, x,filename):
        input_image=x
        filename=filename[0]
        x = self.forward_pass_on_convolutions(x)
        out= F.avg_pool2d(x, kernel_size=7, stride=1).flatten()
        out = self.model.classifier(out)
        if self.target_class is None:
            _,top1=out.topk(1,dim=0,largest=True)
            self.target_class=top1.item()
        _,nc,w,h=x.shape  # must have batch 1
        feature_map=x.squeeze(0).permute(1,2,0)
        cam=torch.matmul(feature_map,self.model.classifier.weight.t())[:,:,self.target_class].to("cpu").detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255.0
        input_image=Image.open(os.path.join("../image_exp/Classification/Data/Test",filename)).resize((224,224),Image.ANTIALIAS)
        save_class_activation_images(input_image,cam,filename)

def save_image(im, path):
    """
        Saves a numpy matrix or PIL image as an image
    Args:
        im_as_arr (Numpy array): Matrix of shape DxWxH
        path (str): Path to the image
    """
    if isinstance(im, (np.ndarray, np.generic)):
        im = format_np_output(im)
        im = Image.fromarray(im)
    im.save(path)


