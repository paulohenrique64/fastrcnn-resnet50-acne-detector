import os
import shutil

VALID_SPLIT = 0.15
TEST_SPLIT = 0.15

IMAGES_FOLDER = os.path.join('data', 'images')
XML_FOLDER = os.path.join('data', 'xmls')
TRAIN_IMAGES_DEST = os.path.join('data', 'train_images')
TRAIN_XML_DEST = os.path.join('data', 'train_xmls')
VALID_IMAGES_DEST = os.path.join('data', 'valid_images')
VALID_XMLS_DEST = os.path.join('data', 'valid_xmls')
TEST_IMAGES_DEST = os.path.join('data', 'test_images')
TEST_XMLS_DEST = os.path.join('data', 'test_xmls')

os.makedirs(TRAIN_IMAGES_DEST, exist_ok=True)
os.makedirs(TRAIN_XML_DEST, exist_ok=True)
os.makedirs(VALID_IMAGES_DEST, exist_ok=True)
os.makedirs(VALID_XMLS_DEST, exist_ok=True)
os.makedirs(TEST_IMAGES_DEST, exist_ok=True)
os.makedirs(TEST_XMLS_DEST, exist_ok=True)

# Adicionando sorted()
all_src_images = sorted(os.listdir(IMAGES_FOLDER))
all_src_xmls = sorted(os.listdir(XML_FOLDER))

temp_images = all_src_images
temp_xmls = all_src_xmls

num_training_images = int(len(temp_images)*(1-VALID_SPLIT-TEST_SPLIT))
num_valid_images = int(len(temp_images)*VALID_SPLIT)
num_test_images = int(len(temp_images)*TEST_SPLIT)

print(num_training_images, num_valid_images, num_test_images)

train_images = temp_images[:num_training_images]
train_xmls = temp_xmls[:num_training_images]

valid_images = temp_images[num_training_images:num_training_images + num_valid_images]
valid_xmls = temp_xmls[num_training_images:num_training_images + num_valid_images]

test_images = temp_images[num_training_images + num_valid_images:]
test_xmls = temp_xmls[num_training_images + num_valid_images:]

for i in range(len(train_images)):
    shutil.copy(os.path.join(IMAGES_FOLDER, train_images[i]), os.path.join(TRAIN_IMAGES_DEST, train_images[i]))
    shutil.copy(os.path.join(XML_FOLDER, train_xmls[i]), os.path.join(TRAIN_XML_DEST, train_xmls[i]))

for i in range(len(valid_images)):
    shutil.copy(os.path.join(IMAGES_FOLDER, valid_images[i]), os.path.join(VALID_IMAGES_DEST, valid_images[i]))
    shutil.copy(os.path.join(XML_FOLDER, valid_xmls[i]), os.path.join(VALID_XMLS_DEST, valid_xmls[i]))

for i in range(len(test_images)):
    shutil.copy(os.path.join(IMAGES_FOLDER, test_images[i]), os.path.join(TEST_IMAGES_DEST, test_images[i]))
    shutil.copy(os.path.join(XML_FOLDER, test_xmls[i]), os.path.join(TEST_XMLS_DEST, test_xmls[i]))
