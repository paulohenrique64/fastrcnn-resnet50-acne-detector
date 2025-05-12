import numpy as np
import cv2
import torch
import glob as glob
import os
import yaml

from torch_utils.utils import create_model

np.random.seed(42)

def main():
    INPUT_DIR = 'data/inference_data'
    OUTPUT_DIR = 'results/inference'
    WEIGHTS_PATH = 'results/training/best_model.pth'
    THRESHOLD = 0.35   

    with open('acne.yaml') as file:
        data_configs = yaml.safe_load(file)
        num_classes = data_configs['NC']
        classes = data_configs['classes']

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # carrega os weights
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)

    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    test_images = []
    if os.path.isdir(INPUT_DIR):
        image_file_types = ['.jpg', '.jpeg', '.png']

        for file_type in image_file_types:
            for file_name in os.listdir(INPUT_DIR):
                if file_name.endswith(file_type):
                    test_images.append(os.path.join(INPUT_DIR, file_name))
    else:
        test_images.append(INPUT_DIR)

    for i in range(len(test_images)):
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])

        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.unsqueeze(image, 0)

        with torch.no_grad():
            outputs = model(image.to(device))

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) != 0:
            image = draw_inference_annotations(
                outputs, THRESHOLD, classes,
                colors, image
            )

            cv2.imshow(image)
            cv2.waitKey(1)

        cv2.imwrite(f"{OUTPUT_DIR}/{image_name}.jpg", image)
        print(f"Finished inference for image {i+1}")
    cv2.destroyAllWindows()

# desenhar as anotações na imagem respeitando o threshold
def draw_inference_annotations(outputs, threshold, classes, colors, image):
    boxes = outputs[0]['boxes'].data.numpy()
    scores = outputs[0]['scores'].data.numpy()

    # filtrar as caixas de acordo com o limite de detecção
    boxes = boxes[scores >= threshold].astype(np.int32)
    draw_boxes = boxes.copy()

    pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

    line_width = 2 # largura da linha
    font_thickness= 1 # espessura da letra
    
    # desenhar as caixas delimitadoras e escrever o nome da classe em cima
    for j, box in enumerate(draw_boxes):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        class_name = pred_classes[j]
        color = colors[classes.index(class_name)]

        cv2.rectangle(
            image,
            p1, p2,
            color=color, 
            thickness=line_width,
            lineType=cv2.LINE_AA
        )

        w, h = cv2.getTextSize(
            class_name, 
            0, 
            fontScale=line_width / 3, 
            thickness=font_thickness
        )[0]  # largura e altura do texto

        cv2.rectangle(
            image, 
            p1, 
            p2, 
            color=color, 
            thickness=-1, 
            lineType=cv2.LINE_AA
        )  

        cv2.putText(
            image, 
            class_name, 
            p1[1] + h + 2,
            cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=line_width / 3.8, 
            color=(255, 255, 255), 
            thickness=font_thickness, 
            lineType=cv2.LINE_AA
        )

    return image

if __name__ == '__main__':
    main()