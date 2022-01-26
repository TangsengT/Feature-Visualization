import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from GradCam.gradcam import GradCam

input_size = 416
image_path = "./docs/image.jpg"

input_layer = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
model.summary()
layer_names = {"scale_13": "tf_op_layer_AddV2_22", "scale_26": "tf_op_layer_AddV2_18",
               "scale_52": "tf_op_layer_AddV2_10"}
pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.4)
bboxes = utils.nms(bboxes, 0.45, method='nms')


def find_high_prob_classes(bboxes):
    top_classes_idx = []
    for i in bboxes:
        if int(i[5]) not in top_classes_idx:
            top_classes_idx.append(int(i[5]))
    return top_classes_idx


classIdx = find_high_prob_classes(bboxes)
for layer_name in layer_names:
    print(f"--------------The Grad-Cam {layer_name} starts.----------------------")
    grad_cam = GradCam(model, classIdx, layer_names[layer_name], layer_name)
    grad_cam.apply_yolo_grad(original_image, tf.constant(image_data, dtype=tf.float32))
print(f"--------------All Grad-Cam results have generated.----------------------")
image = utils.draw_bbox(original_image, bboxes)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imwrite(f"./grad_result/output.jpg", image)
