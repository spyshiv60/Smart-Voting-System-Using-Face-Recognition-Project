import cv2
import numpy as np
import os
import time

size = 6
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
visited_folder = 'visited'

# Create the 'visited' folder if it doesn't exist
if not os.path.exists(visited_folder):
    os.makedirs(visited_folder)

# Check if the visitor has visited before
def check_visitor(name):
    visited_path = os.path.join(visited_folder, name)
    if os.path.exists(visited_path):
        return False  # Already visited
    else:
        # Create a folder for the visitor
        os.makedirs(visited_path)
        return True  # First visit

print('Training...')
(images, labels, names, id) = ([], [], {}, 0)

# Training data preparation
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1

(width, height) = (130, 100)
(images, labels) = [np.array(lis) for lis in [images, labels]]

# Model training
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

last_detection_time = time.time()

while True:
    current_time = time.time()
    if current_time - last_detection_time < 10:
        continue
    last_detection_time = current_time

    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 255, 0), 4)
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))

        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 255, 0), 4)

        if prediction[1] < 800:
            name = names[prediction[0]]
            if check_visitor(name):
                print("Allowed to vote:", name)
            else:
                print("Already visited:", name)
               
            # Save two images of the person in their folder inside the 'visited' folder
            person_folder = os.path.join(visited_folder, name)
            for i in range(2):
                image_name = f"{name}_{i+1}.jpg"
                image_path = os.path.join(person_folder, image_name)
                cv2.imwrite(image_path, im)

        else:
            cv2.putText(im, 'Unknown', (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 4)

    cv2.imshow('OpenCV', im)
    key = cv2.waitKey(5)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()
