import sys
sys.path.insert(0, './')
import os
import glob

import cv2
import torch
import pnnx

import mmcv

from mmseg.apis import init_segmentor
# from mmengine.runner import load_checkpoint
# from mmseg.models import build_segmentor
# from mmcv.runner import load_state_dict


import torch
import torch.nn as nn
import numpy as np


class WrappedSegmentor(nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor.eval()

    def forward(self, x):
        B, C, H, W = x.shape
        img_metas = [{
            'ori_shape': (H, W, 3),
            'img_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
            'flip': False,
        } for _ in range(B)]
        output = self.segmentor.forward([x], img_metas, return_loss=False)
        return output[0]  # 返回第一个样本的结果（单图导出）
    
class ExportableSegmentor(nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor.eval()

    def forward(self, x):
        B, C, H, W = x.shape
        img_metas = [{
            'ori_shape': (H, W, 3),
            'img_shape': (H, W, 3),
            'pad_shape': (H, W, 3),
            'scale_factor': 1.0,
            'flip': False,
        } for _ in range(B)]
        return self.segmentor.forward([x], img_metas, return_loss=False)[0]


class ExportableSegmentor2(nn.Module):
    def __init__(self, segmentor):
        super().__init__()
        self.segmentor = segmentor.eval()

    def forward(self, x):
        # 不使用 img_metas，直接调用 encode_decode，确保 trace 成功
        return self.segmentor.encode_decode(x, self.segmentor.test_cfg)
    
def preprocess_image(img_path, input_size=(512, 512)):
    # 图像读取
    img = cv2.imread(img_path)[:, :, ::-1]  # BGR -> RGB
    ori_shape = img.shape[:2]

    # AlignResize: 保持比例 resize + pad 到 32 的倍数
    h, w = input_size
    scale = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

    pad_img = np.zeros((h, w, 3), dtype=np.uint8)
    pad_img[:resized.shape[0], :resized.shape[1], :] = resized

    # Normalize（基于 config 的 mean/std）
    mean = np.array([114.66601625, 119.63588384, 124.47265218])
    std = np.array([62.54581098, 62.51420716, 63.6411008])
    norm_img = (pad_img.astype(np.float32) - mean) / std

    # 转为 tensor
    # img_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0)  # BCHW
    img_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).unsqueeze(0).float()  # BCHW

    return img_tensor

def main():
    config_path = "configs/repvit/fpn_repvit_m1_1_human.py"
    set_checkpoint = 'log/fpn_repvit_m1_1_human_stage1/iter_1000.pth'
    output = 'tools/pnnx/fpn_repvit.pth'

    input_size = (512, 512)
    
    checkpoint = set_checkpoint
        
    save_path =  'pnnx/human'
    
    img_path = "tools/0_00420.png"
    mean = 119.63588384
    std = 62.5458109
    
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    device = "cpu"
    model = init_segmentor(
        config_path,
        checkpoint,
        device=device)
    
    wrapped_model = ExportableSegmentor2(model)
    # img_tensor = preprocess_image("tools/0_00420.png", input_size).to(device)
    img_tensor = preprocess_image("tools/0_00420.png", input_size).to(device).float()


    wrapped_model.eval()

    # 然后用 pnnx 导出
    # torch.save(wrapped_model, output)  # 保存完整模型
    opt_model = pnnx.export(wrapped_model, output, inputs=img_tensor)



if __name__ == '__main__':
    main()
