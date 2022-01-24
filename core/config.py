from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.YOLO = edict()

__C.YOLO.CLASSES = "./data/classes/coco.names"
__C.YOLO.ANCHORS = "./data/anchors/basline_anchors.txt"
__C.YOLO.STRIDES = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE = 3
__C.YOLO.IOU_LOSS_THRESH = 0.5
