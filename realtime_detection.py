import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from keras import backend as K
from yolo_utils import read_classes, read_anchors, generate_colors, draw_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_eval
from retrain_yolo import create_model
import cv2

CLASS_LIST_PATH = 'model_data/bottle_classes.txt'
YOLO_ANCHORS_PATH = 'model_data/yolo_anchors.txt'
PRETRAINED_WEIGHTS_PATH = 'model_data/yolo_bottle_weights3.h5'


def preprocess_image(image, model_image_size):
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


argparser = argparse.ArgumentParser(description="Realtime bottle detection using YOLOv2")
argparser.add_argument(
    'input',
    type=int,
    default=0,
    nargs='?',
    help="camera index"
)
argparser.add_argument(
    '--score_threshold',
    type=float,
    default=0.15,
    help="box confidence threshold"
)
argparser.add_argument(
    '--iou_threshold',
    type=float,
    default=0.3,
    help="non-max suppression overlap threshold"
)
argparser.add_argument(
    '--weights',
    default=PRETRAINED_WEIGHTS_PATH,
    help="model weights path"
)


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


    cap = cv2.VideoCapture(args.input)
    if (cap.isOpened() == False):
        print("Error opening video capture device.")
        return
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    image_shape = (height, width)

    sess = K.get_session()

    class_names = read_classes(CLASS_LIST_PATH)
    anchors = read_anchors(YOLO_ANCHORS_PATH)
    yolo_model, model = create_model(anchors, class_names)
    yolo_model.load_weights(args.weights)
    yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
    boxes, scores, classes = yolo_eval(yolo_outputs, image_shape, score_threshold=args.score_threshold, iou_threshold=args.iou_threshold)

    while True:
        ret, original_frame = cap.read()

        cv2.imshow('original', original_frame)

        process_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        process_frame = Image.fromarray(process_frame)
        process_frame, out_scores, out_boxes, out_classes = predict(sess, process_frame)
        process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)

        cv2.imshow('output', process_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)