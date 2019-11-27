import numpy as np
import cv2
import colorsys
from skimage.measure import find_contours
from skimage import morphology
from scipy import ndimage
#from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import math

def random_colors(N):
    #np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    random.shuffle(colors)
    return colors

def random_colors1(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    #np.random.seed(1)
    brightness = 1 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
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

def display_instance0(image,boxes,masks,ids,names,scores):

    n_instance = boxes.shape[0]
    colors = random_colors1(n_instance)
    if not n_instance:
        print("no instance to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    #colors = random_colors(n_instance)
    height,width = image.shape[:2]
    eyespot_num = 0
    for i in range(n_instance):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1,x1,y2,x2 = boxes[i]
        mask = masks[:,:,i]
        label = names[ids[i]]
        color = class_dict[label]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        binary_1,contours_1,hierarchy_1 = cv2.findContours(padded_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        score = scores[i] if scores is not None else None
        caption = '{}:{:.2f}'.format(label, score) if score else label
        if label == 'butterfly':
            butterfly_upperright = (x2, y1)
            butterfly_upperleft = (x1, y1)
            butterfly_lowerright = (x2, y2)
            butterfly_lowerleft = (x1, y2)
        else:
            continue
        return [butterfly_lowerright,butterfly_lowerleft,butterfly_upperleft,butterfly_upperright]

def display_instance1(image,raw_img,boxes,masks,ids,names,scores):

    butterfly_lr = display_instance0(image, boxes,masks,ids,names,scores)[0]
    butterfly_ll = display_instance0(image, boxes,masks,ids,names,scores)[1]
    butterfly_ul = display_instance0(image, boxes, masks, ids, names, scores)[2]
    butterfly_ur = display_instance0(image, boxes, masks, ids, names, scores)[3]
    #print(butterfly_lr[0])
    n_instance = boxes.shape[0]
    colors = random_colors1(n_instance)
    if not n_instance:
        print("no instance to display")
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
    #colors = random_colors(n_instance)
    height,width = image.shape[:2]
    eyespot_num = 0
    eyespot_fowardwing = 0
    eyespot_backwardwing = 0
    total_eyespot_area = 0
    avg_eyespot_area = 0
    single_avg_gray_pixel = 0
    diagonal_length = 0
    perimeter_length = 0
    butterfly_box_area = 0
    avg_butterfly_gray_pixel = 0
    sum_single_gray_pixel = 0
    len_sum_single_gray_pixel = 0
    sum_butterfly_gray_pixel = 0
    len_sum_butterfly_gray_pixel = 0
    for i in range(n_instance):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        y1,x1,y2,x2 = boxes[i]
        mask = masks[:,:,i]
        label = names[ids[i]]
        color = class_dict[label]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        binary_1,contours_1,hierarchy_1 = cv2.findContours(padded_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        score = scores[i] if scores is not None else None
        caption = '{}:{:.2f}'.format(label, score) if score else label
        if label == 'eyespot':
            #print(list(zip(np.where(padded_mask == [1])[0], np.where(padded_mask == [1])[1])))
            #single_gray_total_gray_pixel = []
            img_gray = cv2.cvtColor(raw_img,cv2.COLOR_RGB2GRAY)
            '''for indices in list(zip(np.where(padded_mask == [1])[0], np.where(padded_mask == [1])[1])):
                single_gray_total_gray_pixel.append(img_gray[indices])'''
            single_gray_total_gray_pixel = [img_gray[indices] for indices in list(zip(np.where(padded_mask == [1])[0], np.where(padded_mask == [1])[1]))]
            single_avg_gray_pixel += np.mean(single_gray_total_gray_pixel)
            sum_single_gray_pixel += np.sum(single_gray_total_gray_pixel)
            len_sum_single_gray_pixel += len(single_gray_total_gray_pixel)
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            image = cv2.putText(image,caption,(x1,y1-7),cv2.FONT_HERSHEY_COMPLEX,0.4,color,2)
            image = cv2.drawContours(image, contours_1, -1, color, 1)
            if y2 <= ((butterfly_lr[1]+butterfly_ur[1])/2):
                eyespot_fowardwing += 1
            elif y1+20 >= ((butterfly_lr[1]+butterfly_ur[1])/2):
                eyespot_backwardwing += 1
            #eyespot_area = 0
            '''for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                #p = Polygon(verts, facecolor="none", edgecolor=color)
                eyespot_area += math.ceil(area(verts))'''
            eyespot_area = np.sum([math.ceil(area(np.fliplr(verts) - 1)) for verts in contours])
            #cv2.putText(image,"Area:{}".format(eyespot_area),(x1,y1-17),cv2.FONT_HERSHEY_SIMPLEX,0.4,color,2)
            eyespot_num += 1
            total_eyespot_area += eyespot_area
        elif label == 'butterfly':
            #butterfly_gray_pixel = []
            image = apply_mask(image, mask, color)
            image = cv2.rectangle(image,(x1,y1),(x2,y2),color,2)
            image = cv2.putText(image, caption, (x1, y1 - 7), cv2.FONT_HERSHEY_COMPLEX, 0.4, color, 2)
            image = cv2.drawContours(image, contours_1, -1, color, 1)
            img_gray = cv2.cvtColor(raw_img, cv2.COLOR_RGB2GRAY)
            '''for indices in list(zip(np.where(padded_mask == [1])[0], np.where(padded_mask == [1])[1])):
                butterfly_gray_pixel.append(img_gray[indices])'''
            butterfly_gray_pixel = [img_gray[indices] for indices in list(zip(np.where(padded_mask == [1])[0], np.where(padded_mask == [1])[1]))]
            sum_butterfly_gray_pixel += np.sum(butterfly_gray_pixel)
            len_sum_butterfly_gray_pixel += len(butterfly_gray_pixel)
            middle_point = ((x1+x2)/2,(y1+y2)/2)
            diagonal_length = np.sqrt(np.square(x2-middle_point[0])+np.square(y1-middle_point[1]))
            perimeter_length = ((x2-x1)+(y2-y1))*2
            butterfly_box_area = (x2-x1)*(y2-y1)
            #image = cv2.line(image, (int(middle_point[0]),int(middle_point[1])), (x2,y1), (255,0,0), 3)
    if eyespot_num == 0:
        total_avg_gray_pixel = 0
        eyespot_fowardwing = 0
        eyespot_backwardwing = 0
        avg_eyespot_area = 0
    else:
        total_avg_gray_pixel = single_avg_gray_pixel/eyespot_num
        avg_eyespot_area = total_eyespot_area/eyespot_num
    print(total_avg_gray_pixel)
    grayer_array = [[255,255,255],[round(total_avg_gray_pixel) for i in range(3)],[0,0,0]]
    #plt.imshow(grayer_array, cmap='gray')
    #plt.show()
    image = cv2.putText(image, 'eyespot number is:{}'.format(eyespot_num), (0,25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,0,0), 2)
    image = cv2.putText(image, 'forward wing eyespot number is:{}'.format(eyespot_fowardwing), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255, 0, 0), 2)
    image = cv2.putText(image, 'backward wing eyespot number is:{}'.format(eyespot_backwardwing), (0, 75),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'average eyespot area is:{}'.format(avg_eyespot_area), (0, 100),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'average eyespot blackening pixel  is:{}'.format(total_avg_gray_pixel), (0, 125),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'diagonal length  is:{}'.format(diagonal_length), (0, 150),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'perimeter length  is:{}'.format(perimeter_length), (0, 175), cv2.FONT_HERSHEY_COMPLEX,0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'butterfly box area  is:{}'.format(butterfly_box_area), (0, 200), cv2.FONT_HERSHEY_COMPLEX,0.8, (255, 0, 0), 2)
    image = cv2.putText(image, 'butterfly blackening pixel  is:{}'.format((sum_butterfly_gray_pixel+sum_single_gray_pixel)/(len_sum_single_gray_pixel+len_sum_butterfly_gray_pixel)), (0, 225),cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    return image


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
    import mrcnn.model_resnet50_unet as modellib
    from mrcnn import visualize
    from mrcnn.model_resnet50_unet import log
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
        NUM_CLASSES = 1 + 2  # background + 3 shapes

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
    model.load_weights('./logs/mask_rcnn_last_resnet50_unet.h5', by_name=True)

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

    #class_names = ['BG','leaf','root','lyellow','lgreen','rwhite','rgreen']
    #class_names = ['BG', 'leaf', 'root']
    class_names = ['BG', 'butterfly','eyespot']
    colors = random_colors(len(class_names))
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

    for i,img in enumerate(glob.glob('./pic/1938.png')):
        image = skimage.io.imread(img)
        image = np.array(image)
        raw_img = skimage.io.imread(img)
        raw_img = np.array(raw_img)
        #image[np.where((image == [0,0,255]).all(axis=2))] = [255,0,0]
        #print(image)
        results = model.detect([image], verbose=1)
        r = results[0]
        print(img)
        img = display_instance1(image,raw_img, r['rois'], r['masks'], r['class_ids'],
                             class_names,r['scores'])
        #print(np.reshape(r['masks'], (-1, r['masks'].shape[-1])).astype(np.float32).sum())
        #print(img)
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













