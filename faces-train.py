import pickle
import cv2
import os
from PIL import Image
import numpy as np

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(' ', '-').lower()
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            # print(f'label: {label}\n path: {path}')
            pil_image = Image.open(path).convert("L")
            #size = (550, 550)
            # final_image = pil_image.resize(size, Image.ANTIALIAS) # resizing effected badly for me, so I commented it
            image_array = np.array(pil_image, "uint8")
            # print(f'pil_image: {pil_image}\n image_array: {image_array}')
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
print(label_ids)
with open('pickles/labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('recognizers/face-trainner.yml')