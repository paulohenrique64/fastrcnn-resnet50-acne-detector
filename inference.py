import numpy as np
import cv2
import torch
import glob as glob
import os
import yaml

from dataset import create_valid_dataset
from torch_utils.utils import create_model

np.random.seed(42)

def main():
    TEST_IMAGES_DIR = './data/test_images'
    TEST_XMLS_DIR = './data/test_xmls'
    OUTPUT_DIR = './results/inference'
    WEIGHTS_PATH = './results/training/best_model.pth'
    THRESHOLD = 0.6

    with open('acne.yaml') as file:
        data_configs = yaml.safe_load(file)
        num_classes = data_configs['NC']
        classes = data_configs['CLASSES']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # carrega os weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    test_images = []
    if os.path.isdir(TEST_IMAGES_DIR):
        image_file_types = ['.jpg', '.jpeg', '.png']

        for file_type in image_file_types:
            for file_name in os.listdir(TEST_IMAGES_DIR):
                if file_name.endswith(file_type):
                    test_images.append(os.path.join(TEST_IMAGES_DIR, file_name))
    else:
        test_images.append(TEST_IMAGES_DIR)

    images_width = 640
    images_heigth = 640

    for i in range(len(test_images)):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])

        images_heigth, images_width = image.shape[:2]

        # BGR to RGB
        image = cv2.resize(image, (640, 640))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_normalized = image_rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        tensor = torch.unsqueeze(tensor, 0)

        with torch.no_grad():
            outputs = model(tensor.to(device))

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            detections = {
                'labels': outputs[0]['labels'],
                'boxes': outputs[0]['boxes'],
                'scores': outputs[0]['scores']
            }

            image = draw_annotations(image_normalized, images_width, images_heigth, detections, classes, colors, inference=True, threshold=THRESHOLD)
    
        cv2.imwrite(f'{OUTPUT_DIR}/{image_name}.jpg', image)
        print(f'Finished inference for image {image_name}.jpg')

    # montar as anotações criadas manualmente na imagem para comparação
    dataset = create_valid_dataset(TEST_IMAGES_DIR, TEST_XMLS_DIR, 640, 640, classes)

    for i in range(dataset.__len__()):
        image, boxes, labels, _, _, image_path = dataset.load_image_and_labels(i)

        image_name = image_path.split('/')[-1]

        detections = {
            'labels': labels,
            'boxes': boxes,
        }

        image = draw_annotations(image, images_width, images_heigth, detections, classes, colors)
        cv2.imwrite(f'{OUTPUT_DIR}/EXPECTED_{image_name}', image)

    cv2.destroyAllWindows()

# desenhar as anotações na imagem
def draw_annotations(image, image_height, image_width, detections, classes, colors, inference=False, threshold=None):
    import numpy as np
    import cv2

    # garante que a imagem está no formato esperado para o OpenCV
    image = np.ascontiguousarray(image, dtype=np.float32)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (image_height, image_width)) 

    labels = detections['labels']
    boxes = detections['boxes']

    if inference:
        scores = detections['scores']
        filtered_boxes = []
        filtered_labels = []

        for i in range(len(boxes)):
            score = scores[i].detach().cpu().numpy().astype(np.float32)
            if score >= threshold:
                filtered_boxes.append(boxes[i])
                filtered_labels.append(labels[i])

        boxes = filtered_boxes
        labels = filtered_labels

    # escala proporcional ao tamanho da imagem (usado para fonte e espessura da caixa)
    base_dim = min(image_height, image_width)
    thickness = max(1, int(base_dim * 0.002))       
    font_scale = base_dim * 0.0010               

    for i in range(len(labels)):
        label = labels[i].detach().cpu().numpy().astype(np.int32)
        box = boxes[i].detach().cpu().numpy().astype(np.float32)  

        x_scale = image_height / 640
        y_scale = image_width / 640

        x_min = int(box[0] * x_scale)
        y_min = int(box[1] * y_scale)
        x_max = int(box[2] * x_scale)
        y_max = int(box[3] * y_scale)

        class_name = classes[label]
        color = colors[label]

        # desenha a box
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            image,
            class_name,
            (x_min, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 0, 255),
            thickness=max(1, thickness),
            lineType=cv2.LINE_AA
        )

    return (image * 255).clip(0, 255).astype(np.uint8)


if __name__ == '__main__':
    main()