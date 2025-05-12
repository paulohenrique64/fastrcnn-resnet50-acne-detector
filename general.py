import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

import logging
import os
import pandas as pd

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

plt.style.use('ggplot')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def create_model(num_classes):
    # carrega modelo Faster RCNN pré-treinado
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # número de input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # definir uma nova cabeça para o detetor com o número necessário de classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

# Esta classe registra os valores das perdas de treino ajudando a obter a média de cada época
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

# Classe para guardar o melhor modelo durante o treino
class SaveBestModel:
    def __init__(self, best_valid_map=float(0)):
        self.best_valid_map = best_valid_map

    def update(self, model, current_valid_map, epoch, OUT_DIR, config):
        if current_valid_map > self.best_valid_map:
            self.best_valid_map = current_valid_map
            print(f"\nBEST VALIDATION mAP: {self.best_valid_map}")
            print(f"\nSAVING BEST MODEL FOR EPOCH: {epoch+1}\n")
            print(f"{OUT_DIR}/best_model.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'config': config,
                }, f"{OUT_DIR}/best_model.pth")

def set_log(log_dir):
    logging.basicConfig(format='%(message)s', filename=f"{log_dir}/train.log", filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def log(content, *args):
    for arg in args:
        content += str(arg)
    logger.info(content)

def coco_log(log_dir, stats):
    log_dict_keys = [
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]',
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ]',
    ]

    with open(f"{log_dir}/train.log", 'a+') as f:
        f.writelines('\n')
        for i, key in enumerate(log_dict_keys):
            out_str = f"{key} = {stats[i]}"
            logger.debug(out_str) 
        logger.debug('\n'*2) 

