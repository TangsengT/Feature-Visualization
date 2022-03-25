import torchvision.models as models
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch.GradCam.gradcam import GradCam

if __name__ == '__main__':
    resnet = models.resnet101(pretrained=True)
    resnet.to("cuda")
    resnet.eval()
    image = cv2.imread("bird.jpg")
    origin_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(origin_image).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    image = F.upsample(image, (224, 224), mode="bilinear", align_corners=False)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    image = normalize(image)
    gradcam = GradCam(resnet, "./imagenet.names", dataset="imagenet", save_layer_name="all")
    gradcam.apply_all_grad(origin_image, image)
