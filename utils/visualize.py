import cv2
import json
import torch
import numpy as np


# This GradCAM code was supported by CHATGPT 
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.target_layer = model.model.features[8]
        self.model.eval()
        self.feature = None
        self.gradient = None
        self.forward_hook = None
        self.backward_hook = None

    def save_gradient(self, grad):
        self.gradient = grad

    def forward_hook_fn(self, output):
        self.feature = output

    def backward_hook_fn(self, grad_output):
        self.gradient = grad_output[0]

    def __call__(self, x, index=None):
        self.forward_hook = self.target_layer.register_forward_hook(self.forward_hook_fn)
        self.backward_hook = self.target_layer.register_backward_hook(self.backward_hook_fn)

        features = self.model(x)

        if index is None:
            index = torch.argmax(features)

        one_hot = torch.zeros_like(features)
        if len(one_hot.shape) > 2:
            one_hot[0][index] = 1
        else:
            one_hot[0, index] = 1

        self.model.zero_grad()
        features.backward(gradient=one_hot, retain_graph=True)

        grads_val = self.gradient
        target = self.feature

        if len(grads_val.shape) > 2:
            alpha = grads_val.mean(dim=(2, 3), keepdim=True)
            weights = torch.relu((alpha * target).sum(dim=1, keepdim=True))
            weights /= weights.sum(dim=(2, 3), keepdim=True)
            cam = (weights * target).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
        else:
            weights = grads_val.mean(dim=0, keepdim=True)
            cam = (weights * target).sum()

        self.forward_hook.remove()
        self.backward_hook.remove()
        self.feature = None
        self.gradient = None

        return cam



def get_heatmap(cam, image_size=512, delta=20):
    cam = cam.detach().cpu().numpy()[0][0]
    cam = np.uint8(255 * cam)
    cam = cam - np.min(cam)
    # gaussian blur
    cam = cv2.GaussianBlur(cam, (5, 5), 0)
    cam = cv2.resize(cam, (image_size, image_size))
    
    # 75% value
    
    value = np.percentile(cam, 95)
    cam[cam < value] = 0
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    cam = cv2.GaussianBlur(cam, (15, 15), 3)
    cam = cam / np.max(cam)
    
    return 1-cam



def get_heatmap_on_image(heatmap, image):
    return 0.25*heatmap + 0.75*image



def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def resize_and_padding(img, size):
    h, w, c = img.shape
    if h>w:
        new_h = size
        new_w = int(w*(size/h))
    else:
        new_w = size
        new_h = int(h*(size/w))
    img = cv2.resize(img, (new_w, new_h))
    if new_h>new_w:
        pad = (size-new_w)//2
        img = np.pad(img, ((0,0),(pad,pad),(0,0)), 'constant', constant_values=(0,0))
    else:
        pad = (size-new_h)//2
        img = np.pad(img, ((pad,pad),(0,0),(0,0)), 'constant', constant_values=(0,0))
    
    img = cv2.resize(img, (size, size))
    return img