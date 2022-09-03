import numpy as np
import cv2
import torch
import glob
from torchvision import transforms as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('model/autopilot_model.pt', map_location=device)

dataset_dir = "driving_dataset"


def getImagesList():
    images = glob.glob("./driving_dataset/*.jpg")
    return images


def model_predict(model, image):
    image = preprocess_image(image)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5))])
    image = transform(image).unsqueeze(0)
    # print(image.numpy().shape)
    steering_angle = float(model(image))
    steering_angle = steering_angle * 60
    return steering_angle


def preprocess_image(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    return img


steer_image = cv2.imread('file/steering_wheel_image.jpg', 0)
rows, cols = steer_image.shape
smoothed_angle = 0

i=0

while(cv2.waitKey(10) != ord('q')):
    img = cv2.imread(f"driving_dataset/{i}.jpg", cv2.COLOR_BGR2RGB)
    gray = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    steering_angle = model_predict(model, gray)
    print(steering_angle)
    cv2.imshow('frame', img)

    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
            steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer_image, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

    i += 1

cv2.destroyAllWindows()