import torch
import yaml
import numpy as np

from torch_utils.engine import (train_one_epoch, evaluate)
from dataset import (create_train_dataset, create_valid_dataset, create_train_loader, create_valid_loader)
from torch_utils.utils import Averager, SaveBestModel, coco_log, create_model, save_loss_plot, save_mAP, set_log

np.random.seed(42)

def main():
    # Hiperparâmetros
    NUM_WORKERS = 4
    NUM_EPOCHS = 50
    BATCH_SIZE = 2
    IMAGE_WIDTH = 640
    IMAGE_HEIGHT = 640

    # Carrega as configurações do dataset a partir de um arquivo YAML
    with open('acne.yaml') as file:
        data_configs = yaml.safe_load(file)
    
    # Diretórios de treino e validação
    TRAIN_DIR_IMAGES = data_configs['TRAIN_DIR_IMAGES']
    TRAIN_DIR_LABELS = data_configs['TRAIN_DIR_LABELS']
    VALID_DIR_IMAGES = data_configs['VALID_DIR_IMAGES']
    VALID_DIR_LABELS = data_configs['VALID_DIR_LABELS']
    CLASSES = data_configs['CLASSES']
    NUM_CLASSES = data_configs['NC']

    # Cria diretório de saída para salvar os resultados
    OUT_DIR = './results/training'

    # Cores para cada classe (RGB normalizado)
    COLORS = np.random.uniform(0, 1, size=(len(CLASSES), 3))

    # Define o dispositivo de computação (GPU se disponível)
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Inicia o arquivo de log
    set_log(OUT_DIR)

    # Cria datasets de treino e validação com transformações
    train_dataset = create_train_dataset(
        TRAIN_DIR_IMAGES, TRAIN_DIR_LABELS,
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES,
    )

    valid_dataset = create_valid_dataset(
        VALID_DIR_IMAGES, VALID_DIR_LABELS, 
        IMAGE_WIDTH, IMAGE_HEIGHT, CLASSES
    )

    # Cria data loaders
    train_loader = create_train_loader(train_dataset, BATCH_SIZE, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, BATCH_SIZE, NUM_WORKERS)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Inicializa acumuladores de perdas
    train_loss_hist = Averager()
    train_loss_list = []
    loss_cls_list = []
    loss_box_reg_list = []
    loss_objectness_list = []
    loss_rpn_list = []
    train_loss_list_epoch = []
    val_map_05 = []
    val_map = []
    start_epochs = 0

    # Criação do modelo
    model = create_model(num_classes=NUM_CLASSES)
    print(model)

    # Move modelo para GPU ou CPU
    model = model.to(DEVICE)

    # Define otimizador
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)

    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)

    # Inicializa controle de melhor modelo
    save_best_model = SaveBestModel()

    # Loop de treinamento por época
    for epoch in range(start_epochs, NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list, \
             batch_loss_cls_list, \
             batch_loss_box_reg_list, \
             batch_loss_objectness_list, \
             batch_loss_rpn_list = train_one_epoch(
            model, 
            optimizer, 
            train_loader, 
            DEVICE, 
            epoch, 
            train_loss_hist,
            print_freq=100,
        )

        coco_evaluator, stats = evaluate(
            model, 
            valid_loader, 
            device=DEVICE,
        )

        # Armazena perdas e métricas
        train_loss_list.extend(batch_loss_list)
        loss_cls_list.extend(batch_loss_cls_list)
        loss_box_reg_list.extend(batch_loss_box_reg_list)
        loss_objectness_list.extend(batch_loss_objectness_list)
        loss_rpn_list.extend(batch_loss_rpn_list)
        train_loss_list_epoch.append(train_loss_hist.value())
        val_map_05.append(stats[1])
        val_map.append(stats[0])

        # Salva logs de validação
        coco_log(OUT_DIR, stats)

        # Armazena perdas e métricas
        save_loss_plot(
            OUT_DIR, 
            train_loss_list_epoch,
            'epochs',
            'train loss',
            save_name='train_loss_epoch' 
        )

        # Armazena perdas e métricas
        save_mAP(OUT_DIR, val_map_05, val_map)

        # Salva melhor modelo (baseado no mAP médio)
        save_best_model.update(
            model, 
            val_map[-1], 
            epoch, 
            OUT_DIR,
            data_configs,
        )

if __name__ == '__main__':
    main()