# Imports
# From gesture recognize.py
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
# From project 4
import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver

from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# global variables
bg = None


# Parse Arguments to be able to skip training process

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's recognize hand gestures!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--weights', #Names "load-checkpoint" in project 4
        default=None,
        help='''Path to model weights file (should end with the
        extension .h5). Evaluates camera images based on these weights''')
    parser.add_argument(
        '--data',
        default='data'+os.sep,
        help='Location where the dataset is stored.')
    parser.add_argument(
        '--confusion',
        action='store_true',
        help='''Log a confusion matrix at the end of each
        epoch (viewable in Tensorboard). This is turned off
        by default as it takes a little bit of time to complete.''')

    return parser.parse_args()

# Train functions from project 4
def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, hp.max_num_weights)
    ]

    # Include confusion logger in callbacks if flag set
    if ARGS.confusion:
        callback_list.append(ConfusionMatrixLogger(logs_path, datasets))

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=hp.batch_size,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )
#--------------------------------------------------
# To find the running average over the background
#--------------------------------------------------
def run_avg(image, accumWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented, model):
    count = -1
    # Not sure how to return classified label
    

    # print(thresholded, thresholded.shape)
    current_frame = np.array(thresholded)
    # current_frame = np.reshape(current_frame, (215,240,,3))
    #print(current_frame)
    dataset = []
    dataset.append(current_frame)
    dataset = tf.cast(np.array(dataset), tf.float32)    
    result = model.predict( 
    x=dataset, verbose=10)
    count = result[0] 
    print(count) 
    max_prediction = np.argmax(np.array(count))
    return max_prediction

# Test function modified from gesture recognition & project 4
def test(model):
    """ Testing routine. """
    # Enter a repl to obtain input images
    # Run model on test set
    # initialize accumulated weight
    accumWeight = 0.5

    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False

    # keep looping, until interrupted
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()

        # resize the frame
        frame = imutils.resize(frame, width=700)

        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)

        # clone the frame
        clone = frame.copy()

        # get the height and width of the frame
        (height, width) = frame.shape[:2]

        # get the ROI
        roi = frame[top:bottom, right:left]
        roi = cv2.resize(roi, (224,224))
        # roi = np.resize(roi, (224,224,3))
        # roi = imutils.resize(roi, (128,128))

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        ############
        # run_avg(gray, accumWeight)
        # hand = segment(gray)
        # (thresholded, segmented) = hand
        # cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
        
        # count the number of fingers
        fingers = count(roi, None, model)
        cv2.putText(clone, str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        # show the thresholded image
        # cv2.imshow("Thesholded", thresholded)
        ###########
        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (0,128,0), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
    camera.release()

# Main function from project 4
def main():
    """ Main function. """

    # This is for training
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    # If paths provided by program arguments are accurate, then this will
    # ensure they are used. If not, these directories/files will be
    # set relative to the directory of run.py
    # Not quite sure what this is...?????
    if os.path.exists(ARGS.data):
        ARGS.data = os.path.abspath(ARGS.data)

    # Run script from location of run.py
    os.chdir(sys.path[0])

    datasets = Datasets(ARGS.data)
    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
        "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + \
        os.sep + timestamp + os.sep

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"])

    if ARGS.weights is None:
        # We will train model to obtain weights if we don't have weights
        # Print summary of model
        model.summary()
        # Make checkpoint directory if needed
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        train(model, datasets, checkpoint_path, logs_path, init_epoch)
        evaluation = model.evaluate( x=datasets.test_data, verbose=1, batch_size = hp.batch_size)
        print(evaluation)
    else:
        model.load_weights(ARGS.weights, by_name = False)
        test(model)
        

# Make arguments global
ARGS = parse_args()

main()

cv2.destroyAllWindows()