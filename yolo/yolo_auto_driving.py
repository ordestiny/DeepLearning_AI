import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model
from keras.utils.vis_utils import plot_model,model_to_dot
from yolo_utils import scale_boxes,read_classes,read_anchors,preprocess_image,generate_colors,draw_boxes
from yad2k.models.keras_yolo import yolo_boxes_to_corners,yolo_head
from IPython.display import SVG

#  阈值过滤，滤掉一部分box
def yolo_filter_boxes(box_confidence, boxes,box_class_probs, threshold = .6):
    box_scores = np.multiply(box_confidence,box_class_probs)
    box_classes = K.argmax(box_scores)
    box_class_scores = K.max(box_scores,axis=-1)
    filtering_mask= box_class_scores >= threshold
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes  = tf.boolean_mask(boxes,filtering_mask)
    classes  = tf.boolean_mask(box_classes,filtering_mask)
    return scores,boxes,classes

# 交并比: 交集面积大小 / 并集面积大小
def iou(box1, box2):
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi2 - xi1) * (yi2 - yi1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area
    return iou

# 非最大值抑制，再过滤一部分box，只保留一个最优的box
def yolo_non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)
    #  gather 在给定的2D张量中检索给定下标的向量
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)
    return scores, boxes, classes

# 计算
def yolo_eval(yolo_outputs,image_shape = (720.,1280.),max_boxes = 10,score_threshold = .6,iou_threshold = .5):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
    boxes  = yolo_boxes_to_corners(box_xy,box_wh)
    scores,boxes,classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs,score_threshold)
    boxes = scale_boxes(boxes,image_shape)
    scores,boxes,classes = yolo_non_max_suppression(scores,boxes,classes,max_boxes,iou_threshold)
    return scores,boxes,classes

def predict(sess,image_file):
    out_file = os.path.join("out",image_file)
    image,image_data = preprocess_image("images/" + image_file, model_image_size=(608,608))
    out_scores,out_boxes,out_classes = sess.run([scores,boxes,classes],feed_dict={yolo_model.input:image_data,K.learning_phase():0})
    color = generate_colors(class_names)
    draw_boxes(image,out_scores,out_boxes,out_classes,class_names,color)
    image.save(out_file,quality = 90)
    output_image = scipy.misc.imread(out_file)
    imshow(output_image)
    return out_scores,out_boxes,out_classes

if __name__=="__main__":
    print("main")
    sess = K.get_session()
    class_names = read_classes("model_data/coco_classes.txt")
    anchors = read_anchors("model_data/yolo_anchors.txt")
    image_shape = (720., 1280.)

    yolo_model = load_model("model_data/yolo.h5")
    yolo_model.summary()
    # plot_model(yolo_model,to_file="images/model.png")
    # SVG(model_to_dot(yolo_model).create(prog="dot",format="svg"))
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    scores, boxes, classes = yolo_eval(yolo_outputs, image_shape,max_boxes = 10,score_threshold=.5, iou_threshold=.4)
    out_scores,out_boxes,out_classes = predict(sess,"car.jpg")
