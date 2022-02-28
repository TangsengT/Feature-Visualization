import tensorflow as tf
import numpy as np
# from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from GradCam.gradcam import GradCam
import cv2
import os

if __name__ == '__main__':
    model = tf.keras.applications.xception.Xception(weights="imagenet")
    # model = ResNet101()
    # layer_name = "conv5_block3_add"
    original_image = cv2.imread("docs/bird.jpg")
    image = load_img("docs/bird.jpg", target_size=(299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model.summary()
    gradcam = GradCam(model)
    gradcam.apply_cls_grad(original_image, image)
