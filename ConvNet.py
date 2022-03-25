import tensorflow as tf
import numpy as np
# from tensorflow.keras.applications.resnet import ResNet101, preprocess_input, decode_predictions
from tensorflow.keras.applications.xception import Xception, preprocess_input, decode_predictions
# from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from GradCam.gradcam import GradCam
import cv2
import os

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # model = VGG16()
    path = "docs/bird.jpg"
    # model = ResNet101()
    model = Xception()
    # layer_name = "conv5_block3_add"
    original_image = cv2.imread(path)
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    model.summary()
    preds = model.predict(image)
    print(decode_predictions(preds, top=1)[0])
    gradcam = GradCam(model, "data/classes/imagenet.names", dataset="imagenet", classIdx=None, save_layer_name="all")
    gradcam.show_all_grad(original_image, image, step=1)
    # gradcam.apply_cls_grad(original_image,image)
