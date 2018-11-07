"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import glob as glob
import numpy as np
import skimage.draw
import pydensecrf.densecrf as dcrf

from imgaug import augmenters as iaa
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class IDCardConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "idcard"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

    BACKBONE = "resnet50"

    MAX_GT_INSTANCES = 1
    USE_MINI_MASK = False
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    LEARNING_RATE = 0.001
    MASK_SHAPE = [56, 56]
    RPN_TRAIN_ANCHORS_PER_IMAGE = 128 # 256
    TRAIN_ROIS_PER_IMAGE = 100 #200
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.2
    }


############################################################
#  Dataset
############################################################

class IDCardDataset(utils.Dataset):

    def ptcross4(self, xy):
        (x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8) = xy
        fx=x7*x1*y8-x8*x1*y7-x7*x2*y8+x8*x2*y7-x7*x1*y2+x7*x2*y1+x8*x1*y2-x8*x2*y1
        fy=x7*y8*y1-x8*y7*y1-x7*y8*y2+x8*y7*y2-x1*y7*y2+x2*y7*y1+x1*y8*y2-x2*y8*y1
        ff=x7*y1-x1*y7-x7*y2-x8*y1+x1*y8+x2*y7+x8*y2-x2*y8
        rx1=float(fx)/float(ff)
        ry1=float(fy)/float(ff)
        px1=int(rx1)
        py1=int(ry1)
        fx=x1*x3*y2-x2*x3*y1-x1*x4*y2+x2*x4*y1-x1*x3*y4+x1*x4*y3+x2*x3*y4-x2*x4*y3
        fy=x1*y2*y3-x2*y1*y3-x1*y2*y4+x2*y1*y4-x3*y1*y4+x4*y1*y3+x3*y2*y4-x4*y2*y3
        ff=x1*y3-x3*y1-x1*y4-x2*y3+x3*y2+x4*y1+x2*y4-x4*y2
        rx2=float(fx)/float(ff)
        ry2=float(fy)/float(ff)
        px2=int(rx2)
        py2=int(ry2)
        fx=x3*x5*y4-x4*x5*y3-x3*x6*y4+x4*x6*y3-x3*x5*y6+x3*x6*y5+x4*x5*y6-x4*x6*y5
        fy=x3*y4*y5-x4*y3*y5-x3*y4*y6+x4*y3*y6-x5*y3*y6+x6*y3*y5+x5*y4*y6-x6*y4*y5
        ff=x3*y5-x5*y3-x3*y6-x4*y5+x5*y4+x6*y3+x4*y6-x6*y4
        rx3=float(fx)/float(ff)
        ry3=float(fy)/float(ff)
        px3=int(rx3)
        py3=int(ry3)
        fx=x5*x7*y6-x6*x7*y5-x5*x8*y6+x6*x8*y5-x5*x7*y8+x5*x8*y7+x6*x7*y8-x6*x8*y7
        fy=x5*y6*y7-x6*y5*y7-x5*y6*y8+x6*y5*y8-x7*y5*y8+x8*y5*y7+x7*y6*y8-x8*y6*y7
        ff=x5*y7-x7*y5-x5*y8-x6*y7+x7*y6+x8*y5+x6*y8-x8*y6
        rx4=float(fx)/float(ff)
        ry4=float(fy)/float(ff)
        px4=int(rx4)
        py4=int(ry4)
        return px1,py1,px2,py2,px3,py3,px4,py4

    def load_idcard(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("idcard_back", 1, "idcard_back")
        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = []
        images = glob.glob(os.path.join(dataset_dir, 'images/*.jpg'))
        for _image in images:
            annotations.append(os.path.join(dataset_dir, 'labels/{}.json'.format(os.path.splitext(os.path.basename(_image))[0])))
        # Add images
        for image_path, json_path in zip(images, annotations):
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            bname = os.path.basename(image_path)
            annotation = json.load(open(json_path))
            polygons = [{'name': 'polygon',
                        'all_points_x': [],
                        'all_points_y': []}]
            if not annotation['labeled']:
                continue
            try:
                height, width = int(annotation['size']['height']), int(annotation['size']['width'])
                object_list = annotation['outputs']['object']
                for d_obj in object_list:
                    if 'polygon' in d_obj.keys():
                        xys = []
                        for i in range(1, 9):
                            xys.append(int(d_obj['polygon']['x{}'.format(i)]))
                            xys.append(int(d_obj['polygon']['y{}'.format(i)]))
                        px1, py1, px2, py2, px3, py3, px4, py4 = self.ptcross4(xys)
                        polygons[0]['all_points_x'].extend([px1, px2, px3, px4])
                        polygons[0]['all_points_y'].extend([py1, py2, py3, py4])
            except KeyError:
                print('error annotation', json_path)
                continue
            if not polygons[0]['all_points_x'] \
                or not polygons[0]['all_points_x'] \
                or len(polygons[0]['all_points_x']) != 4 \
                or len(polygons[0]['all_points_y']) != 4:
                print('no polygon annotation', json_path)
                continue
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            #image = skimage.io.imread(image_path)
            height, width = int(annotation['size']['height']), int(annotation['size']['width'])
            self.add_image(
                "idcard_back",
                image_id=bname,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "idcard_back":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            #usedy = list(map(lambda x: info["height"] - x, p['all_points_y']))
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            try:
                mask[rr, cc, i] = 1
            except IndexError:
                for _r in range(rr.shape[0]):
                    if rr[_r] < 0:
                        rr[_r] = 0
                    elif rr[_r] >= info["height"]:
                        rr[_r] = info["height"] - 1
                for _c in range(cc.shape[0]):
                    if cc[_c] < 0:
                        cc[_c] = 0
                    elif cc[_c] >= info["width"]:
                        cc[_c] = info["width"] - 1
                mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "idcard_back":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = IDCardDataset()
    dataset_train.load_idcard(args.dataset, "train")
    dataset_train.prepare()
    # Validation dataset
    dataset_val = IDCardDataset()
    dataset_val.load_idcard(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    augmentation = iaa.SomeOf((0, 2), [
        # iaa.Fliplr(0.5),
        # iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0))
    ])
    layers = '3+'
    print("Training network {}".format(layers))
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                augmentation=augmentation,
                layers=layers)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 0
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect idcard.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/idcard/dataset/",
                        help='Directory of the idcard dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--gpu', required=True,
                        metavar="gpu ids",
                        help='Choose gpus to train/val')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    # Configurations
    if args.command == "train":
        config = IDCardConfig()
        config.GPU_COUNT = len(args.gpu.split(','))
    else:
        class InferenceConfig(IDCardConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
