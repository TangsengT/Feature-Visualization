import os
import torch
from pytorch.GradCam.utils import read_class_names, read_imagenet_names, get_all_layers
import torch.nn.functional as F
import numpy as np
import cv2


class GradCam:
    def __init__(self, model, classes_path, dataset="default", layer_name=None, classIdx=None, save_layer_name=None):
        if model is None:
            raise ValueError("Model could not be none.")
        if classes_path is None:
            raise ValueError("Classes path could not be none.")
        self.model = model
        self.classIdx = classIdx
        self.layer_name = layer_name
        self.save_layer_name = save_layer_name
        self.classes = read_imagenet_names(classes_path) if dataset == "imagenet" else read_class_names(classes_path)
        self.all_layers = get_all_layers(self.model)
        self.layer_output = []
        self.grad_output = []
        if layer_name is None:
            self.layer_name = self._find_last_conv_layer()
        if save_layer_name is None:
            self.save_layer_name = self.layer_name
        self._init_file()

    def _init_file(self):
        if not os.path.exists("grad_result"):
            os.mkdir("grad_result")
        if not os.path.exists(f"grad_result/{self.model.__class__.__name__}"):
            os.mkdir(f"grad_result/{self.model.__class__.__name__}")
        if not os.path.exists(f"grad_result/{self.model.__class__.__name__}/{self.save_layer_name}"):
            os.mkdir(f"grad_result/{self.model.__class__.__name__}/{self.save_layer_name}")

    def _find_last_conv_layer(self):
        for name in reversed(list(self.all_layers.keys())):
            if "conv" in name:
                return name
        raise ValueError("Couldn't find a convolutional layer.")

    def _forward_hook(self, model, inputs, outputs):
        if len(outputs.squeeze().shape) == 4 or len(outputs.squeeze().shape) == 3:
            self.layer_output.append(outputs)
        else:
            pass

    def _backward_hook(self, model, inputs, outputs):
        if len(outputs[0].squeeze().shape) == 4 or len(outputs[0].squeeze().shape) == 3:
            self.grad_output.insert(0, outputs[0])
        else:
            pass

    def apply_all_grad(self, original_image, image):
        for layer in self.all_layers.values():
            layer.register_forward_hook(self._forward_hook)
            layer.register_backward_hook(self._backward_hook)
        logit = self.model(image)
        if self.classIdx is None:
            self.classIdx = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit.squeeze()[self.classIdx]
        self.model.zero_grad()
        score.backward(retain_graph=False)
        for i, (layer_output, grad_output) in enumerate(zip(self.layer_output, self.grad_output)):
            grad = grad_output.squeeze().permute(1, 2, 0)
            output = layer_output.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            grad = grad.mean((0, 1)).cpu().numpy()
            heatmap = output @ grad
            heatmap = F.relu(torch.from_numpy(heatmap))
            heatmap = (heatmap - heatmap.min()).div(heatmap.max() - heatmap.min()).numpy()
            self._save_grad_result(heatmap, original_image, self.classes[int(self.classIdx.cpu().numpy())],
                                   save_index=str(i))

    def apply_cls_grad(self, original_image, image):
        layer = self.all_layers[self.layer_name]
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)
        logit = self.model(image)
        if self.classIdx is None:
            self.classIdx = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit.squeeze()[self.classIdx]
        self.model.zero_grad()
        score.backward(retain_graph=False)
        for i, (layer_output, grad_output) in enumerate(zip(self.layer_output, self.grad_output)):
            grad = grad_output.squeeze().permute(1, 2, 0)
            output = layer_output.squeeze().permute(1, 2, 0).cpu().detach().numpy()
            grad = grad.mean((0, 1)).cpu().numpy()
            heatmap = output @ grad
            heatmap = F.relu(torch.from_numpy(heatmap))
            heatmap = (heatmap - heatmap.min()).div(heatmap.max() - heatmap.min()).numpy()
            self._save_grad_result(heatmap, original_image, self.classes[int(self.classIdx.cpu().numpy())],
                                   save_index=str(i))

    def _save_grad_result(self, heatmap, original_image, save_class_name, save_index="0"):
        heatmap = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        colormap = cv2.COLORMAP_JET
        heatmap = cv2.applyColorMap(heatmap, colormap)
        if len(original_image.shape) == 2:
            original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        grad_img = original_image * 0.9 + heatmap
        cv2.imwrite(
            f"./grad_result/{self.model.__class__.__name__}/{self.save_layer_name}/output_{save_class_name}_{save_index}.jpg",
            grad_img)
