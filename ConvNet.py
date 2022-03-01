import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
# from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
# from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from GradCam.gradcam import GradCam
import cv2
import os

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # model = VGG16()
    model = ResNet101()
    # layer_name = "block5_conv3"
    original_image = cv2.imread("docs/bird.jpg")
    image = load_img("docs/bird.jpg", target_size=(224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model.summary()
    gradcam = GradCam(model)
    gradcam.apply_cls_grad(original_image, image)
