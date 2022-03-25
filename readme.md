### Feature Visualization in ConvNet & YOLOv3

This is a graduation design completed in 2022. If you want to use this interface, I provide two different framework tensorflow 2.0 & pytorch 1.10.

##### Notice: The pytorch version didn't work in YOLOv3, please use tensorflow version if you want to visualize YOLOv3 !

#### Tensorflow Version

If you want to use this version, you can declare a GradCam instance.

```python
def __init__(self, model, classes_path, classIdx, dataset="default", layer_name=None, save_layer_name=None)
```

model -- a tensorflow version model

classes_path -- where the labels store in

classIdx -- the specific class you want to visualize

dataset -- use the default if you don't use the imagenet pretrained models

layer_name -- the specific layer you want to visualize

save_layer_name -- the path name

This is the example:

```python
from GradCam.gradcam import GradCam
gradcam = GradCam(model, "data/classes/imagenet.names", dataset="imagenet", classIdx=None, save_layer_name="all")

```

Use the code below to show all the feature maps in heatmap mode.

original_image -- the raw image

image -- the preprocessed image which is predicted in the model

step -- apply the visualization every specific step(when it is 1, it would show all layers' feature maps)

```python
gradcam.show_all_grad(original_image, image, step=1)
```

One specific class visualization:

```python
gradcam.apply_cls_grad(original_image, image)
```

YOLOv3 visualization:

```python
grad_cam.apply_yolo_grad(original_image, tf.constant(image_data, dtype=tf.float32))
```

