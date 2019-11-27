import numpy as np
import cv2
import colorsys
from skimage.measure import find_contours
from skimage import morphology
#from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import math

def random_colors(N):
    #np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    #random.shuffle(colors)
    return colors

def random_colors1(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    #np.random.seed(1)
    brightness = 1 if bright else 0.7
    hsv = [(i/N, brightness, 255) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #random.shuffle(colors)
    return colors

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for n,c in enumerate(color):
        image[:, :, n] = np.where(mask == 1,
        image[:, :, n] * (1 - alpha) + alpha * c,
        image[:, :, n])
    return image

'''def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                 image[:, :, c])
    mask_px = np.where(mask)
    for c in range(3):
        image[mask_px[0], mask_px[1], c] = (1 - alpha)*image[mask_px[0], mask_px[1], c] + alpha * color[c] * 255
    return image'''

#class_names = ['BG','leaf','root']
#class_names = ['BG','butterfly','Plaque']
#colors = random_colors(len(class_names))
#class_dict = {name: color for name, color in zip(class_names, colors)}

def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return abs(a)

def display_instance(image,boxes,masks,ids,names,scores):

    n_instance = boxes.shape[0]
    colors = random_colors1(n_instance)
    if not n_instance:
        print("no instance to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    #colors = random_colors(n_instance)
    leaf_num = 0
    root_num = 0
    root_green_num = 0
    root_white_num = 0
    leaf_yellow_num = 0
    leaf_green_num = 0
    total_leaf_length = 0
    total_leaf_area = 0
    total_leaf_width = 0
    total_leafgreen_area = 0
    total_leafyellow_area = 0
    total_root_length = 0
    total_rootgreen_length = 0
    total_rootwhite_length = 0
    avg_leaf_area = 0
    avg_leaf_width = 0
    avg_leaf_length = 0
    avg_root_length = 0
    avg_rootgreen_length = 0
    avg_rootwhite_length = 0
    avg_leafyellow_area = 0
    avg_leafgreen_area = 0
    height,width = image.shape[:2]
    for i in range(n_instance):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1,x1,y2,x2 = boxes[i]
        mask = masks[:,:,i]
        label = names[ids[i]]
        color = class_dict[label]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = apply_mask(image,mask,color)
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        binary_1,contours_1,hierarchy_1 = cv2.findContours(padded_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(image,contours_1,-1,color,1)
        score = scores[i] if scores is not None else None
        caption = '{}:{:.2f}'.format(label,score) if score else label
        #image = cv2.putText(image,caption,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
        if label == 'leaf':
            skeleton = morphology.skeletonize(padded_mask)
            binary_2, contours_2, hierarchy_2 = cv2.findContours(skeleton.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #plt.imshow(skeleton,cmap="gray")
            #plt.contour(skeleton,cmap=plt.get_cmap("Spectral"))
            #plt.show()
            #print(np.array(contours_2).shape)
            if len(np.array(contours_2).shape) == 1:
                leaflength = 0
                for i in range(np.array(contours_2).shape[0]):
                    arr = np.array(contours_2[i]).squeeze()
                    if len(np.array(arr).shape) ==1:
                        leaflength += 7.5
                    else:
                        #length += cv2.arcLength(np.unique(np.array(arr),axis=0), False)
                        leaflength += cv2.arcLength(arr, True)/2 + 7.5
            else:
                contours_2 = np.array(contours_2).squeeze()
                #contours_2 = np.unique(np.array(contours_2),axis=0)
                if len(np.array(contours_2).shape) == 1:
                    leaflength = 7.5
                else:
                    leaflength = cv2.arcLength(contours_2, True)/2 + 7.5
            #print(leaflength)
            #cv2.putText(image, "length:{}cm".format(round(leaflength/57.5,2)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            total_leaf_length += leaflength
            leaf_num+=1
            #leafarea = 0
            '''for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                #p = Polygon(verts, facecolor="none", edgecolor=color)
                leafarea += math.ceil(area(verts))'''
            leafarea = np.sum([math.ceil(area(np.fliplr(verts) - 1)) for verts in contours])
            #cv2.putText(image,"Area:{}cm2".format(round(leafarea/3150,2)),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            #print(leafarea)
            total_leaf_area += leafarea
            cnt = np.array(contours_1[0])
            # print(cnt)
            rect = cv2.minAreaRect(cnt)
            leafwidth = np.min(rect[1])
            #box = np.int0(cv2.boxPoints(rect))
            #cv2.drawContours(image, [box], 0, (255, 0, 0), 2)
            #print(leafwidth)
            cv2.putText(image, "width:{}cm".format(round(leafwidth/57.5,2)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            total_leaf_width += leafwidth
        elif label == 'root':
            skeleton = morphology.skeletonize_3d(padded_mask)
            binary_2, contours_2, hierarchy_2 = cv2.findContours(skeleton.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #plt.imshow(skeleton,cmap="gray")
            #plt.contour(skeleton,cmap=plt.get_cmap("Spectral"))
            #plt.show()
            #print(np.array(contours_2).squeeze()[0].shape)
            if len(np.array(contours_2).shape) == 1:
                rootlength = 0
                for i in range(np.array(contours_2).shape[0]):
                    arr = np.array(contours_2[i]).squeeze()
                    if len(np.array(arr).shape) == 1:
                        rootlength += 7.5
                    else:
                        # rootlength += cv2.arcLength(np.unique(np.array(arr),axis=0), False)
                        rootlength += cv2.arcLength(arr, True) / 2 + 7.5
            else:
                contours_2 = np.array(contours_2).squeeze()
                # contours_2 = np.unique(np.array(contours_2),axis=0)
                if len(np.array(contours_2).shape) == 1:
                    rootlength = 7.5
                elif len(np.array(contours_2).shape) == 3:
                    rootlength = np.sum([cv2.arcLength(contours_2[index], True) / 2 + 7.5 for index in range(contours_2.shape[0])])
                else:
                    rootlength = cv2.arcLength(contours_2, True) / 2 + 7.5
            #print(rootlength)
            #cv2.putText(image, "length:{}cm".format(round(rootlength/57.5,2)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            total_root_length += rootlength
            root_num += 1
        elif label == 'rgreen':
            skeleton = morphology.skeletonize_3d(padded_mask)
            binary_2, contours_2, hierarchy_2 = cv2.findContours(skeleton.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #plt.imshow(skeleton,cmap="gray")
            #plt.contour(skeleton,cmap=plt.get_cmap("Spectral"))
            #plt.show()
            # print(np.array(contours_2).shape)
            if len(np.array(contours_2).shape) == 1:
                rootgreenlength = 0
                for i in range(np.array(contours_2).shape[0]):
                    arr = np.array(contours_2[i]).squeeze()
                    if len(np.array(arr).shape) == 1:
                        rootgreenlength += 7.5
                    else:
                        # length += cv2.arcLength(np.unique(np.array(arr),axis=0), False)
                        rootgreenlength += cv2.arcLength(arr, True) / 2 + 7.5
            else:
                contours_2 = np.array(contours_2).squeeze()
                # contours_2 = np.unique(np.array(contours_2),axis=0)
                if len(np.array(contours_2).shape) == 1:
                    rootgreenlength = 7.5
                else:
                    rootgreenlength = cv2.arcLength(contours_2, True) / 2 + 7.5
            #print(rootgreenlength)
            #cv2.putText(image, "length:{}cm".format(round(rootgreenlength/57.5,2)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            root_green_num += 1
            total_rootgreen_length += rootgreenlength
        elif label == 'rwhite':
            skeleton = morphology.skeletonize_3d(padded_mask)
            binary_2, contours_2, hierarchy_2 = cv2.findContours(skeleton.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            #plt.imshow(skeleton,cmap="gray")
            #plt.contour(skeleton,cmap=plt.get_cmap("Spectral"))
            #plt.show()
            # print(np.array(contours_2).shape)
            if len(np.array(contours_2).shape) == 1:
                rootwhitelength = 0
                for i in range(np.array(contours_2).shape[0]):
                    arr = np.array(contours_2[i]).squeeze()
                    if len(np.array(arr).shape) == 1:
                        rootwhitelength += 7.5
                    else:
                        # length += cv2.arcLength(np.unique(np.array(arr),axis=0), False)
                        rootwhitelength += cv2.arcLength(arr, True) / 2 + 7.5
            else:
                contours_2 = np.array(contours_2).squeeze()
                # contours_2 = np.unique(np.array(contours_2),axis=0)
                if len(np.array(contours_2).shape) == 1:
                    rootwhitelength = 7.5
                else:
                    rootwhitelength = cv2.arcLength(contours_2, True) / 2 + 7.5
            # print(rootwhitelength)
            #cv2.putText(image, "length:{}cm".format(round(rootwhitelength,2)),(x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            root_white_num += 1
            total_rootwhite_length += rootwhitelength
        elif label == 'lyellow':
            leaf_yellow_num += 1
            #leafyellowarea = 0
            '''for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                # p = Polygon(verts, facecolor="none", edgecolor=color)
                leafyellowarea += math.ceil(area(verts))'''
            leafyellowarea = np.sum([math.ceil(area(np.fliplr(verts) - 1)) for verts in contours])
            #cv2.putText(image,"Area:{}cm2".format(leafyellowarea/3150),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            #print(leafyellowarea)
            total_leafyellow_area += leafyellowarea
        elif label == 'lgreen':
            leaf_green_num += 1
            #leafgreenarea = 0
            '''for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                # p = Polygon(verts, facecolor="none", edgecolor=color)
                leafgreenarea += math.ceil(area(verts))'''
            leafgreenarea = np.sum([math.ceil(area(np.fliplr(verts) - 1)) for verts in contours])
            #cv2.putText(image,"Area:{}cm2".format(leafgreenarea/3150),(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
            #print(leafgreenarea)
            total_leafgreen_area += leafgreenarea
    if leaf_num == 0:
        total_leaf_area = 0
        total_leaf_width = 0
        total_leaf_length = 0
        avg_leaf_area = 0
        avg_leaf_width = 0
        avg_leaf_length = 0
    elif leaf_num != 0:
        avg_leaf_area += total_leaf_area/leaf_num
        avg_leaf_length += total_leaf_length/leaf_num
        avg_leaf_width += total_leaf_width/leaf_num
    if root_num == 0:
        total_root_length = 0
    elif root_num != 0:
        avg_root_length += (total_root_length + total_rootwhite_length + total_rootwhite_length)/root_num
    if root_green_num == 0:
        total_rootgreen_length = 0
    elif root_green_num != 0:
        avg_rootgreen_length += total_rootgreen_length/root_green_num
    if root_white_num == 0:
        total_rootwhite_length = 0
    elif root_white_num != 0:
        avg_rootwhite_length += total_rootwhite_length/root_white_num
    if leaf_yellow_num == 0:
        total_leafyellow_area = 0
    elif leaf_yellow_num != 0:
        avg_leafyellow_area += total_leafyellow_area/leaf_yellow_num
    if leaf_green_num == 0:
        total_leafgreen_area = 0
    elif leaf_green_num != 0:
        avg_leafgreen_area += total_leafgreen_area/leaf_green_num

    cv2.putText(image,'leaf_number:{}'.format(leaf_num),(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)
    #cv2.putText(image, 'Avg_leaf_area:{}cm2'.format(round(avg_leaf_area/3150,2)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    #cv2.putText(image, 'Avg_leaf_length:{}cm'.format(round(avg_leaf_length/57.5,2)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    #cv2.putText(image, 'Avg_leaf_width:{}cm'.format(round(avg_leaf_width/57.5, 2)), (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    #cv2.putText(image, 'Avg_root_length:{}cm'.format(round((total_root_length+total_rootgreen_length+total_rootwhite_length)/root_num/57.5, 2)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), 2)
    #cv2.putText(image, 'Avg_leafgreen_area:{}cm2'.format(round(avg_leafgreen_area/3150,2)), (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255, 0, 0), 2)
    #cv2.putText(image, 'Avg_leafyellow_area:{}cm2'.format(round(avg_leafyellow_area/3150,2)), (10, 85),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    #cv2.putText(image, 'Avg_rootgreen_length:{}cm'.format(round(avg_rootgreen_length/57.5,2)), (10, 85), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 0, 0), 2)
    #cv2.putText(image,'Avg_rootwhite_length:{}cm'.format(round(avg_rootwhite_length/57.5,2)), (10, 115), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 0, 0), 2)
    cv2.putText(image, 'root_number:{}'.format(root_num), (10,55), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2)
    cv2.putText(image, 'rootgreen_number:{}'.format(root_green_num), (10, 85), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2)
    cv2.putText(image, 'rootwhite_number:{}'.format(root_white_num), (10, 115), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2)
    cv2.putText(image, 'leafgreen_number:{}'.format(leaf_green_num), (10, 145), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2)
    cv2.putText(image, 'leafyellow_number:{}'.format(leaf_yellow_num), (10, 175), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255, 0, 0), 2)
    return image

'''def display_instance1(image,boxes,masks,ids,names,scores):

    n_instance = boxes.shape[0]
    colors = random_colors1(n_instance)
    if not n_instance:
        print("no instance to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    #colors = random_colors(n_instance)
    height,width = image.shape[:2]
    for i in range(n_instance):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1,x1,y2,x2 = boxes[i]
        mask = masks[:,:,i]
        label = names[ids[i]]
        color = class_dict[label]
        image = apply_mask(image,mask,color)
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        binary_1,contours_1,hierarchy_1 = cv2.findContours(padded_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        image = cv2.drawContours(image,contours_1,-1,color,1)
        if label == 'butterfly':
            image = cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            score = scores[i] if scores is not None else None
            caption = '{}:{:.2f}'.format(label,score) if score else label
            image = cv2.putText(image,caption,(x1,y1-7),cv2.FONT_HERSHEY_COMPLEX,0.4,color,1)
        else:
            continue
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            #p = Polygon(verts, facecolor="none", edgecolor=color)
            cv2.putText(image,"Area:{}".format(math.ceil(area(verts))),
                        (x1,y1-9),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)
    return image'''


if __name__ == '__main__':

    '''import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import sys
    import random
    import math
    import numpy as np
    import skimage.io
    import matplotlib
    import matplotlib.pyplot as plt

    import coco
    import utils
    import model as modellib
    import visualize
    import matplotlib.image as mpimg
    from moviepy.editor import VideoFileClip
    import tensorflow as tf

    tf.logging.set_verbosity(tf.logging.ERROR)

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    # Download COCO trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)'''

    import os
    import sys
    import random
    import math
    import re
    import time
    import glob
    import numpy as np
    #np.set_printoptions(threshold=np.inf)
    import cv2
    import skimage.io
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import matplotlib.image as mpimg
    import time

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    tf.reset_default_graph()
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model_unet as modellib
    from mrcnn import visualize
    from mrcnn.model_unet import log
    import coco
    from moviepy.editor import VideoFileClip

    # Directory to save logs and trained model
    MODEL_DIR = "./logs"

    # Local path to trained weights file
    COCO_MODEL_PATH = "./mask_rcnn_coco_smoother.h5"
    # Download COCO trained weights from Releases if needed

    class InferenceConfig(Config):

        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # Number of classes (including background)
        NUM_CLASSES = 1 + 6  # background + 3 shapes

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 768  # 832 #512
        IMAGE_MAX_DIM = 768  # 832 #512

        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 100

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 68

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 2

    config = InferenceConfig()
    #config.display()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights('./logs/mask_rcnn_last_resnet101_unet_graduate.h5', by_name=True)

    '''class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']'''

    class_names = ['BG','leaf','root','lyellow','lgreen','rwhite','rgreen']
    #class_names = ['BG', 'leaf', 'root']
    #class_names = ['BG', 'butterfly','eyespot']
    colors = random_colors1(len(class_names))
    class_dict = {name: color for name, color in zip(class_names, colors)}

    '''capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,1000)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,1000)

    while(True):
        ret, frame = capture.read()
        if ret is True:
            results = model.detect([frame], verbose=0)
            r = results[0]
            frame = display_instance(frame, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
            frame = np.array(frame)
            #frame = frame.astype(np.uint8)
            cv2.imshow('frame_window',frame)
            #cv2.waitKey(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()'''


    '''def process_image_video(img):
        results = model.detect([img], verbose=0)
        r = results[0]
        img = display_instance(img, r['rois'], r['masks'], r['class_ids'],
                                  class_names, r['scores'])
        img = np.array(img)
        return img

    white_output = 'project_video.mp4'
    clip1 = VideoFileClip("p1.mp4")
    white_clip = clip1.fl_image(process_image_video)  # NOTE: this function expects color images!!
    white_clip.write_videofile(white_output, audio=False)'''

    for i,img in enumerate(glob.glob('./flower_graduate/21.png')):
        image = skimage.io.imread(img)
        image = np.array(image)
        #M = np.float32([[1, 0, 30], [0, 1, 0]])
        #dst = cv2.warpAffine(image, M, (768,768))
        #print(image)
        results = model.detect([image], verbose=1)
        r = results[0]
        print(img)
        img = display_instance(image, r['rois'], r['masks'], r['class_ids'],
                             class_names,r['scores'])
        #print(np.reshape(r['masks'], (-1, r['masks'].shape[-1])).astype(np.float32).sum())
        plt.imshow(img)
        plt.show()
        '''cv2.imshow('window',img)
        cv2.waitKey(1)
        time.sleep(0.0002)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()'''
        #img = np.array(img)
        #print(img)













