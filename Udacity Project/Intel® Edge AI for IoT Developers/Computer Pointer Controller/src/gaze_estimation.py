import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import time
import sys
import math

class GazeEstimation:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions

        try:
            self.network = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.network.inputs))
        self.left_shape=self.network.inputs['left_eye_image'].shape
        self.right_shape=self.network.inputs['right_eye_image'].shape
        self.output_name=next(iter(self.network.outputs))

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        if self.extensions and 'CPU' in self.device:
            self.core.add_extension(extension_path=self.extensions, deivce_name=self.device)

        self.exec_network = self.core.load_network(self.network, self.device, num_requests=1)

    def check_model(self):
        '''
        Check for any unsupported layers, and let the user
        know if anything is missing. Exit the program, if so.
        '''
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)

        unsupported_layers = [l for l in exec_network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def predict(self, left_image, right_image, pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict = self.preprocess_input(left_image, right_image, pose_angles)
        self.exec_network.start_async(request_id=0, inputs=input_dict)
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            output = self.preprocess_output(left_image, right_image, outputs)
            return output

    def preprocess_input(self, left_image, right_image, pose_angles):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        left_image = cv2.resize(left_image, (self.left_shape[2], self.left_shape[3]))
        left_image = left_image.transpose((2,0,1))
        left_image = left_image.reshape(1, *self.left_shape)

        right_image = cv2.resize(right_image, (self.right_shape[2], self.right_shape[3]))
        right_image = right_image.transpose((2,0,1))
        right_image = right_image.reshape(1, *self.right_shape)

        input_dict = {'left_eye_image': left_image, 'right_eye_image': right_image, 'head_pose_angles': pose_angles}
        return input_dict

    def preprocess_output(self, left_image, right_image, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        gaze_coords = []
        gaze_vector = outputs[0]
        roll_vector = gaze_vector[2]
        # gaze_vector = gaze_vector / np.linalg.norm(gaze_vector)
        gaze_coords_x = gaze_vector[0] * left_image.shape[1]
        gaze_coords_y = gaze_vector[1] * left_image.shape[0]
        cos = math.cos(roll_vector * math.pi / 180.0)
        sin = math.sin(roll_vector * math.pi / 180.0)

        tarX = gaze_vector[0] * cos + gaze_vector[1] * sin
        tarY = -gaze_vector[0] * sin + gaze_vector[1] * cos

        gaze_coords = [gaze_coords_x, gaze_coords_y]
        return (tarX,tarY), (gaze_vector), gaze_coords

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.network
        del self.exec_network
