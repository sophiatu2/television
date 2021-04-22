# Imports
import cv2
import imutils
import numpy as np
import pygame
import os
import sys
import argparse
import re
import tensorflow as tf
import hyperparameters as hp
import random

from datetime import datetime
from sklearn.metrics import pairwise
from scipy import stats
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import \
        ImageLabelingLogger, ConfusionMatrixLogger, CustomModelSaver
from skimage.io import imread
from lime import lime_image
from skimage.segmentation import mark_boundaries
from matplotlib import pyplot as plt

# set up the environment 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# global variables
bg = None

'''Parse Arguments that are given to the command line after running <python main.py>'''
def parse_args():
    # Add arguements to the command line
    parser = argparse.ArgumentParser(
        description="Let's recognize hand gestures!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--weights', 
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


def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    '''Trains the data from the files in the television folder
    Inputs:
        model: model for the CNN 
        datasets: the data used to train/test the CNN on
        checkpoint_path: path to the checkpoint
        logs_path: path to the logs
        init_epoch: the epochs that the program is running on
    Outputs:
        Trained weights that can be used in testing'''

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
        batch_size=None,
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )

def run_avg(image, accumWeight):
    '''Finds the running average over the background
    Inputs:
        image: the input images used
        accumWeight: the weight placed on that imaged
    Outputs:
        accumulates the weights of image'''
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, accumWeight)

def segment(image, threshold=25):
    '''Segments the region of hand in the image
    Inputs:
        image: the input images used
        threshold: the threshold of which we use to segment the image
    Outputs:
        segments the image by giving a black background'''

    global bg

    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_TOZERO)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

def count(thresholded, segmented, model):
    '''Counts the number of fingers in the segmented hand region
    Inputs:
        threshold: the threshold of which we use to segment the image
        segmented: the segemented image with a black background 
        model: the CNN model we used to get the weights
    Outputs:
        get a predicted weight of which finger is displayed'''

    count = -1
    # set the current frame
    current_frame = np.array(thresholded)

    # set the database
    dataset = []
    dataset.append(current_frame)
    dataset = tf.cast(np.array(dataset), tf.float32) 

    # predict the dataset   
    result = model.predict( 
    x=dataset, verbose=10)
    count = result[0] 

    # get the prediction 
    max_prediction = np.argmax(np.array(count))
    return max_prediction, count

def test(model, music):
    '''Test function modified that runs the camera capturing
    Inputs:
        model: the CNN model we used to get the weights
        music: file name of the music that is being played
    Outputs:
        uses a camera to get live feed connection'''

    # set the accumilative weight
    accumWeight = 0.5
    
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # initialize num of frames
    num_frames = 0

    # calibration indicator
    calibrated = False
    
    # variables used in the while loop 
    average_numb = 0
    temp_average = []
    recount_frame = 0
    command = "Command: "
    songs = ["friends", "hallelujah", "flamingo", "twistAndShout", "world", "dance",]

    # loads the music and pauses it
    loadMusic(music)
    pygame.mixer.music.pause()

    # keep running until terminated
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

        # increase the brightness
        brightness = 127
        contrast = 127
        img = np.int16(roi)
        img = img * (contrast/127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)
        roi = img

        # convert the roi to grayscale and blur it
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # calibrate the live feed
        if num_frames < 30:
            run_avg(gray, accumWeight)
            if num_frames == 1:
                cv2.putText(clone, "Calibrating", (350, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (227,132,36), 2)
                print("calibration of background image")
            elif num_frames == 29:
                cv2.putText(clone, "successfull", (350, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (227,132,36), 2)
                print("calibration successfull")
        else:
            # segment the hand region
            hand = segment(gray)

            # check whether hand region is segmented
            if hand is not None:
                # if yes, unpack the thresholded image and
                # segmented region
                (thresholded, segmented) = hand

                # draw the segmented region and display the frame
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                
                gray = thresholded
                # clone = gray.copy()

                # modify the input image
                gray = np.stack([gray, gray, gray], axis=-1)
                gray = cv2.resize(gray, (224,224))

                # get the predicted number of fingers and the arrary of predictions
                (fingers, check_hand) = count(gray, None, model)
                
                # checks if the hand is in the screen
                proceed = True
                for i in check_hand:
                    if (i != 1 and i != 0):
                        proceed = False
                        break
                
                # if the hand is in the screen, then proceed
                if proceed:
                    if recount_frame < 3:
                        temp_average.append(fingers)
                        recount_frame += 1
                    else:
                        # commands for the live fee
                        average_numb = stats.mode(temp_average)[0][0]
                        print("Command " + str(average_numb))
                        if average_numb == 0:
                            pygame.mixer.music.pause()
                            command = "Pausing: "
                        if average_numb == 1:
                            music = random.choice(songs) + ".mp3"
                            loadMusic(music)
                            command = "Random: "
                        elif average_numb == 3:
                            loadMusic(music)
                            command = "Restart: "
                        elif average_numb == 5:
                            pygame.mixer.music.unpause()
                            command = "Unpause: "
                        temp_average = []
                        recount_frame = 0
                    
                    # write text on the popup
                    cv2.putText(clone, "Fingers: " + str(fingers), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (186, 227, 36), 2)
                    cv2.putText(clone, command + str(average_numb), (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (227,195,36), 2)
                
                cv2.putText(clone, "Place hand in box", (350, 275), cv2.FONT_HERSHEY_SIMPLEX, 1, (227,132,36), 2)

                # show the thresholded image
                cv2.imshow("Thesholded", thresholded)

        # draw the segmented hand
        cv2.rectangle(clone, (left, top), (right, bottom), (227,71,36), 2)

        # increment the number of frames
        num_frames += 1

        # display the frame with segmented hand
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) 

        # if the user pressed "c", then change the song
        if keypress == ord("c"):
            if (input("Would you like to change songs? (y/n)") == "y"):
                music = input("Please input an audio file from the list below: \n- friends\n- hallelujah\n- flamingo\n- twistAndShout\n- world\n- dance\n") + ".mp3"
                loadMusic(music)
                pygame.mixer.music.pause()
        
        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break


def loadMusic(music):
    ''' loads the music if the file name exists within the directory 
        inputs: 
            music: the file name of the music
        output: 
            loads and starts playing the music
    '''

    try:
        pygame.mixer.init()
        pygame.mixer.music.load(music)
        print("Playing:", music)
        pygame.mixer.music.play()
    except:
        # if the file does not exist, ask for one that does
        print("The audio file you requested is invalid: " + music)
        music = input("Please input an audio file from the list below: \n- friends\n- hallelujah\n- flamingo\n- twistAndShout\n- world\n- dance\n") + ".mp3"
        loadMusic(music)


def main():
    """ Main function. """

    # ask for the music file path
    music = input("Please input an audio file from the list below: \n- friends\n- hallelujah\n- flamingo\n- twistAndShout\n- world\n- dance\n")
    music += ".mp3"

    # set up training
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

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
        model.summary()
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        train(model, datasets, checkpoint_path, logs_path, init_epoch)
        evaluation = model.evaluate( x=datasets.test_data, verbose=1)
        print(evaluation)
    else:
        model.load_weights(ARGS.weights, by_name = False)
        test(model, music)

# Make arguments global
if __name__ == '__main__':
    cv2.__version__
    ARGS = parse_args()
    main()
    camera.release()
    cv2.destroyAllWindows()