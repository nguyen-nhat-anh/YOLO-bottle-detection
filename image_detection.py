import argparse
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from yolo_utils import read_classes, read_anchors, generate_colors, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval
from retrain_yolo import create_model
import cv2

def preprocess_image(image, model_image_size):
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data



argparser = argparse.ArgumentParser(description="Bottle detection for images using YOLOv2")
argparser.add_argument(
    '-i',
    '--input',
    help="image file name (.jpg)")

INPUT_PATH = 'images'
OUTPUT_PATH = 'out'

CLASS_LIST_PATH = 'model_data/bottle_classes.txt'
YOLO_ANCHORS_PATH = 'model_data/yolo_anchors.txt'
PRETRAINED_WEIGHTS_PATH = 'model_data/yolo_bottle_weights.h5'

def _main(args):
    def predict(sess, image):
        # Preprocess your image
        image, image_data = preprocess_image(image, model_image_size = (416, 416))

        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict = {yolo_model.input:image_data, K.learning_phase():0})

        # Print predictions info
        print('Found {} boxes'.format(len(out_boxes)))
        # Generate colors for drawing bounding boxes.
        colors = generate_colors(class_names)
        # Draw bounding boxes on the image file
        out_image = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    
        return out_image, out_scores, out_boxes, out_classes


    image_path = os.path.join(INPUT_PATH, args.input)
    input = cv2.imread(image_path)
    if (input is None):
        print("Error opening image file")
        return
    height = float(input.shape[0])
    width = float(input.shape[1])

    image_shape = (height, width)    

    sess = K.get_session()

    class_names = read_classes(CLASS_LIST_PATH)
    anchors = read_anchors(YOLO_ANCHORS_PATH)
    yolo_model, model = create_model(anchors, class_names)
    yolo_model.load_weights(PRETRAINED_WEIGHTS_PATH)
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)

    process_img = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    process_img = Image.fromarray(process_img)
    process_img, out_scores, out_boxes, out_classes = predict(sess, process_img)
    output = cv2.cvtColor(process_img, cv2.COLOR_BGR2RGB)

    cv2.imshow('prediction', output)
    cv2.waitKey(0)
    
    output_path = os.path.join(OUTPUT_PATH, args.input)
    cv2.imwrite(output_path, output)
    print("Saved: ", output_path)


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)