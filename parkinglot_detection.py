# -- coding: utf-8 --
import os
import cv2
import sys
import time
import argparse
import multiprocessing
import math
#
import collections
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from moviepy.editor import ImageSequenceClip
from IPython.display import HTML


CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'parkinglot_output_inference_graph.pb'
#PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_CKPT = os.path.join('/home/cyj/Project/data/Parkinglot20180201_Output', MODEL_NAME, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(CWD_PATH, 'object_detection', 'data', 'parkinglot_label_map.pbtxt')
VIDEO_PATH='/home/cyj/Project/data/20180402/480p/VID_20180402_160919'
PATH_TO_OUTPUT_INFO = VIDEO_PATH+'.txt'
white_output = VIDEO_PATH+'_out.mp4'
video_input=VIDEO_PATH+'.mp4'
NUM_CLASSES = 2

if os.path.isfile(PATH_TO_OUTPUT_INFO):
    os.remove(PATH_TO_OUTPUT_INFO)
if os.path.isfile(white_output):
    os.remove(white_output)


# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# print box informations
def output_detection_information(image_np,
                                 boxes,
                                 classes,
                                 scores,
                                 catagory_index,
                                 instance_masks=None,
                                 keypoints=None,
                                 use_normalized_coordinates=False,
                                 max_boxes_to_draw=20,
                                 min_score_thresh=.65,
                                 agnostic_mode=False,
                                 line_thickness=8):
    class_names=[]
    scoreis=[]
    ymins=[]
    xmins=[]
    ymaxs=[]
    xmaxs=[]
    class_name=" "
    scorei=0
    ymin=0
    xmin=0
    ymax=0
    xmax=0
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(class_name,int(100*scores[i]))
                    #
                    scorei=scores[i]
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                class_names.append(class_name)
                scoreis.append(scorei)
                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)
    if(len(class_names)==0):
        class_names.append(" ")
        scoreis.append(0)
        ymins.append(0)
        xmins.append(0)
        ymaxs.append(0)
        xmaxs.append(0)
    #print("class_name:",class_name)
    #print("score:",scorei)
    #print("-------------")
    #print(display_str)
    #print(ymin,xmin,ymax,xmax)
    #return display_str,ymin,xmin,ymax,xmax
    return class_names, scoreis, ymins, xmins, ymaxs, xmaxs


def detect_objects(image_np, sess, detection_graph):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Write detection information to file
    classname, score, ymin, xmin, ymax, xmax = output_detection_information(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)
    # print(display_str,ymin,xmin,ymax,xmax)

    fp = open(PATH_TO_OUTPUT_INFO, "a")
    ob_num=len(classname)
    nu=0
    while nu<ob_num:
        fp.write(classname[nu])
        fp.write(' ')
        fp.write(str(score[nu]))
        fp.write(' ')
        fp.write(str(ymin[nu]))
        fp.write(' ')
        fp.write(str(xmin[nu]))
        fp.write(' ')
        fp.write(str(ymax[nu]))
        fp.write(' ')
        fp.write(str(xmax[nu]))
        fp.write('\n')
        nu=nu+1
    fp.close()
    return image_np

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Load a frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_process = detect_objects(image, sess, detection_graph)
            return image_process



#clip = VideoFileClip(video_input).subclip(0,5).set_fps(10)
clip = VideoFileClip(video_input).set_fps(10)
white_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!s
#%time white_clip.write_videofile(white_output, audio=False)
white_clip.write_videofile(white_output, audio=False)

