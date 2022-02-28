from tensorflow.keras.models import Model
import numpy as np
from core.utils import read_class_names
from core.config import cfg
import tensorflow as tf
import cv2
import os


class GradCam:
    def __init__(self, model, classIdx=None, layer_name=None, save_layer_name=None):
        self.model = model
        self.classIdx = classIdx
        self.layer_name = layer_name
        self.save_layer_name = save_layer_name
        if layer_name is None:
            self.layer_name = self.find_last_conv_layer()
            self.save_layer_name = self.layer_name
        self._init_file()

    def _init_file(self):
        if not os.path.exists("grad_result"):
            os.mkdir("grad_result")
        if not os.path.exists(f"grad_result/{self.model.name}"):
            os.mkdir(f"grad_result/{self.model.name}")
        if not os.path.exists(f"grad_result/{self.model.name}/{self.save_layer_name}"):
            os.mkdir(f"grad_result/{self.model.name}/{self.save_layer_name}")

    def find_last_conv_layer(self):
        for l in reversed(self.model.layers):
            if len(l.output_shape) == 4:
                return l.name
        raise ValueError("Couldn't find the 4 dimensions layer output.")

    def apply_cls_grad(self, original_image, image):
        grad_model = tf.keras.Model(inputs=self.model.inputs,
                                    outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            last_conv_output, pred = grad_model(image)
            if self.classIdx is None:
                self.classIdx = tf.argmax(pred[0])
            score = pred[..., self.classIdx]
        grads = tape.gradient(score, last_conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
        last_conv_output = last_conv_output.numpy()[0]
        heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        self._save_grad_result(heatmap, original_image, "bird")

    def apply_yolo_grad(self, original_image, image):
        classes = read_class_names(cfg.YOLO.CLASSES)
        for classIndex in self.classIdx:
            print(f"--------------Grad-Cam {classes[classIndex]}----------------------")
            grad_model = tf.keras.Model(inputs=self.model.inputs,
                                        outputs=[self.model.get_layer(self.layer_name).output, self.model.output])
            heatmaps, maxVals, normalized_heatmaps = [], [], []
            for i in range(3):
                with tf.GradientTape() as tape:
                    last_conv_output, pred_bbox = grad_model(image)
                    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
                    score = pred_bbox[i][..., 5 + classIndex]
                grads = tape.gradient(score, last_conv_output)
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
                last_conv_output = last_conv_output.numpy()[0]
                for j in range(pooled_grads.shape[-1]):
                    last_conv_output[:, :, j] *= pooled_grads[j]
                heatmap = np.mean(last_conv_output, axis=-1)
                heatmap = np.maximum(heatmap, 0)
                heatmaps.append(heatmap)
                maxVals.append(np.max(heatmap))
            for i, heatmap in enumerate(heatmaps):
                heatmap = self._normalize_heatmap(heatmap, maxVals)
                normalized_heatmaps.append(heatmap)
                self._save_grad_result(heatmap, original_image, classes[classIndex], str(i))
            heatmap = np.maximum.reduce(normalized_heatmaps)
            self._save_grad_result(heatmap, original_image, classes[classIndex], "combined")
            print(f"--------------Grad-Cam {classes[classIndex]} has generated.----------------------")

    def _normalize_heatmap(self, heatmap, maxVals):
        heatmap = heatmap / np.max(maxVals)
        return heatmap

    def _save_grad_result(self, heatmap, original_image, save_class_name, save_index="0"):
        heatmap = np.clip(heatmap, 0, 1)
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        colormap = cv2.COLORMAP_JET
        heatmap = cv2.applyColorMap(heatmap, colormap)
        grad_img = original_image * 0.9 + heatmap
        cv2.imwrite(f"./grad_result/{self.model.name}/{self.save_layer_name}/output_{save_class_name}_{save_index}.jpg",
                    grad_img)
