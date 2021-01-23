#!/usr/bin/env python3
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to run generic MobileNet based classification model."""

import argparse
import re
import paho.mqtt.client as mqtt
from picamera import Color
from picamera import PiCamera
import time
from aiy.vision import inference
from aiy.vision.models import utils
import time
import pygame
import os


#Sets the IP used to subscribe to ThermalPI using MQTT.
#Sets the channel/path to be listened to which is temp.txt
#Sets the global variable temp_msg_payload to string "0".
MQTT_SERVER = "192.168.0.31"
MQTT_PATH = "temp.txt"
temp_msg_payload = "0"

#Was Pre-written for the AIY kit.
#Used to read the label file that sets the classes that the script can classify.
def read_labels(label_path):
    with open(label_path) as label_file:
        return [label.strip() for label in label_file.readlines()]

#Handles logic using the result and temp_msg values.
#result is a list of the classifications that the camera sees and a percentage of certainty
#example would be mug (1.00) = mug detected 100%.
#temp_msg is returned from MQTT as a string. for example "22" this allowed for easy type conversion.
def get_message(result,temp_msg):
    matching_mug = [s for s in result if "mug" in s] #Searches the result list for "mug" which could be of any percentage certainty.
    in_temp_msg = int(temp_msg) #Converts the string temperature value into an Int.

    #Conditions to decide which audio message to play.
    #Whether the mug is detected or not there is responses for high temperatures.
    #Any temperature detected above 30 degrees alerts the user to a hot object.
    #The if statement determines if there is No mug, 100% a mug or a possibly detected mug.
    #Filename contains the name of the audio clip to be played. that is set depending on which condition is met.
    #Then filename is passed to audio_readout() which handles playing the voice response.
    if matching_mug == [] and in_temp_msg <30:
        Filename = "NoMug_NoHeat.mp3"
    elif matching_mug == [] and in_temp_msg >30:
        Filename = "NoMug_YesHeat.mp3"
    elif matching_mug[0] == 'mug (1.00)' and in_temp_msg <30:
        Filename = "YesMug_NoHeat.mp3"
    elif matching_mug[0] == 'mug (1.00)' and in_temp_msg >30:
        Filename = "YesMug_YesHeat.mp3"
    elif "mug" in matching_mug[0] and in_temp_msg >30:
        Filename = "Possibly_YesMug_YesHeat.mp3"
    elif "mug" in matching_mug[0] and in_temp_msg <30:
        Filename = "Possibly_YesMug_NoHeat.mp3"
    audio_readout(Filename)

#Takes Filename input and appends it the path where the audio files are kept.
#Uses the pygame libary to play the audio.
def audio_readout(Filename):
    Filepath = "/home/pi/Desktop/Hons_ProjectFolder/HonsVoiceFiles/"    
    pygame.mixer.init()
    pygame.mixer.music.load(Filepath + Filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

'''MQTT setup code uses to establish a connection and recieve messages '''
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_PATH)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global temp_msg_payload # sets global variable temp_msg_payload
    split_temp = msg.payload.decode("utf-8").strip("'").split('.') #Parses temperature data recieved form MQTT message.
    temp_msg_payload = split_temp[0]
    return msg.payload


client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_SERVER, 1883, 60)
''' End of MQTT setup code'''


#Called from within a loop in main to use MQTT to check for any new temperature messages.
def get_tempature():
    client.loop()
    return ""

#Was Pre-written for the AIY kit.
#The purpose is to handle the data returned by the tensorflow neural network.
def process(result, labels, tensor_name, threshold, top_k):
    """Processes inference result and returns labels sorted by confidence."""
    # MobileNet based classification model returns one result vector.
    assert len(result.tensors) == 1
    tensor = result.tensors[tensor_name]
    probs, shape = tensor.data, tensor.shape
    assert shape.depth == len(labels)
    pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pairs = pairs[0:top_k]
    return ['%s (%.2f)' % (labels[index], prob) for index, prob in pairs]


#Was Pre-written for the AIY kit. Then adpated to suit the projects needs.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
        help='Path to converted model file that can run on VisionKit.')
    parser.add_argument('--label_path', required=True,
        help='Path to label file that corresponds to the model.')
    parser.add_argument('--input_height', type=int, required=True, help='Input height.')
    parser.add_argument('--input_width', type=int, required=True, help='Input width.')
    parser.add_argument('--input_layer', required=True, help='Name of input layer.')
    parser.add_argument('--output_layer', required=True, help='Name of output layer.')
    parser.add_argument('--num_frames', type=int, default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    parser.add_argument('--input_mean', type=float, default=128.0, help='Input mean.')
    parser.add_argument('--input_std', type=float, default=128.0, help='Input std.')
    parser.add_argument('--input_depth', type=int, default=3, help='Input depth.')
    parser.add_argument('--threshold', type=float, default=0.1,
        help='Threshold for classification score (from output tensor).')
    parser.add_argument('--top_k', type=int, default=3, help='Keep at most top_k labels.')
    parser.add_argument('--preview', action='store_true', default=False,
        help='Enables camera preview in addition to printing result to terminal.')
    parser.add_argument('--show_fps', action='store_true', default=False,
        help='Shows end to end FPS.')
    args = parser.parse_args()

    model = inference.ModelDescriptor(
        name='mobilenet_based_classifier',
        input_shape=(1, args.input_height, args.input_width, args.input_depth),
        input_normalizer=(args.input_mean, args.input_std),
        compute_graph=utils.load_compute_graph(args.model_path))
    labels = read_labels(args.label_path)

    with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
        if args.preview:
            camera.start_preview()

        with inference.CameraInference(model) as camera_inference:
            for result in camera_inference.run(args.num_frames):
                processed_result = process(result, labels, args.output_layer,
                                           args.threshold, args.top_k)
                get_tempature() #Causes MQTT to loop to recieve temperature data.
                try:
                    get_message(processed_result,temp_msg_payload) #Is called to handle the voice response logic.
                except:
                    print("Index error occured, handled gracefully") #Handles an error that is sometimes causes by a list index error.
                if args.show_fps:
                    message += '\nWith %.1f FPS.' % camera_inference.rate
                

                if args.preview:
                    camera.annotate_foreground = Color('black')
                    camera.annotate_background = Color('white')
                    # PiCamera text annotation only supports ascii.
                    camera.annotate_text = '\n %s' % message.encode(
                        'ascii', 'backslashreplace').decode('ascii')

        if args.preview:
            camera.stop_preview()


if __name__ == '__main__':
    main()
