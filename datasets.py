import albumentations as A
from matplotlib import transforms
import torch
import cv2
import numpy as np
import os
import glob as glob

import torchvision.transforms.functional as F
from xml.etree import ElementTree as et
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class CustomDataset(Dataset):
    def __init__(self, images_path, labels_path, width, height, classes, train=False):
        self.images_path = images_path
        self.labels_path = labels_path
        self.height = height
        self.width = width
        self.classes = classes
        self.train = train
        self.all_image_paths = []
        self.all_annot_paths = []

        # obter os caminhos das imagens
        for file_name in os.listdir(images_path):
            if file_name.endswith('.jpg'):
                image_path = os.path.join(images_path, file_name)
                self.all_image_paths.append(image_path)

        # obter os caminhos das annotations
        for file_name in os.listdir(labels_path):
            if file_name.endswith('.xml'):

                label_path = os.path.join(labels_path, file_name)
                self.all_annot_paths.append(label_path)

    def load_image_and_labels(self, index):
        image_path = self.all_image_paths[index]
        image = cv2.imread(image_path)

        # converter BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized = image_resized.astype(np.float32) / 255.0 # normalização

        # capturar o arquivo xml correspondente para acessar as anotações
        annotation_path = image_path.replace('.jpg', '.xml')
        annotation_path = annotation_path.replace('images', 'xmls')

        boxes = []
        orig_boxes = []
        labels = []
        tree = et.parse(annotation_path)
        root = tree.getroot()
        
        # captura a altura e a largura da imagem
        image_width = image.shape[1]
        image_height = image.shape[0]
                
        # Coordenadas da caixa no XML são extraídas e corrigidas conforme o tamanho da imagem
        for member in root.findall('object'):
            # Mapeia o nome do objeto atual para a lista de classes e obtém o índice correspondente
            labels.append(self.classes.index(member.find('name').text))
            
            xmin = round(float(member.find('bndbox').find('xmin').text))
            xmax = round(float(member.find('bndbox').find('xmax').text))
            ymin = round(float(member.find('bndbox').find('ymin').text))
            ymax = round(float(member.find('bndbox').find('ymax').text))

            orig_boxes.append([xmin, ymin, xmax, ymax])
            
            # Redimensiona as caixas conforme o novo tamanho desejado
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Converte bounding boxes para tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # Área das bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # Sem instâncias crowd (valor 0)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # Converte rótulos para tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return image_resized, boxes, labels, area, iscrowd

    
    def __getitem__(self, index):
        image, boxes, labels, area, iscrowd = self.load_image_and_labels(index)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([index])
        }

        transform = get_augmentations(train=self.train)
        sample = transform(image=image, bboxes=boxes.tolist(), labels=labels.tolist())
        image = sample['image']
        target['boxes'] = torch.tensor(sample['bboxes'], dtype=torch.float32)
        target['labels'] = torch.tensor(sample['labels'], dtype=torch.int64)

        return image, target

    def __len__(self):
        return len(self.all_image_paths)

def get_augmentations(train):
    transforms = []
    
    if train:
        transforms.extend([
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=0.4),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.97, 1.03), translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, rotate=(-5, 5), p=0.3),
        ])
    
    transforms.append(ToTensorV2(p=1.0))

    return A.Compose(transforms, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# Métodos auxiliares para criação dos datasets de treino e validação
def create_train_dataset(train_dir_images, train_dir_labels, resize_width, resize_height, classes):
    train_dataset = CustomDataset(
        train_dir_images, 
        train_dir_labels,
        resize_width, 
        resize_height, 
        classes, 
        train=True
    )
    return train_dataset

def create_valid_dataset(valid_dir_images, valid_dir_labels, resize_width, resize_height, classes):
    valid_dataset = CustomDataset(
        valid_dir_images, 
        valid_dir_labels, 
        resize_width, 
        resize_height, 
        classes, 
        train=False
    )
    return valid_dataset

# Métodos auxiliares para criação dos dataloaders
def collate_fn(batch):
    return tuple(zip(*batch))

def create_train_loader(train_dataset, batch_size, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, batch_size, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn 
    )
    return valid_loader