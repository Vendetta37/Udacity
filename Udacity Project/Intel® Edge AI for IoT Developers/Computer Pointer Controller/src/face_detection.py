import numpy as np
from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import argparse
import time
import sys

class FaceDetection:
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
        self.input_shape=self.network.inputs[self.input_name].shape
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

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        input_dict = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0, inputs=input_dict)
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            frame, coords = self.preprocess_output(image, outputs)
            return coords, frame

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

    def preprocess_input(self, image):
        '''
        TODO: You will need to complete this method.
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_image = cv2.resize(image,  (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)
        input_dict = {self.input_name:p_image}
        return input_dict

    def preprocess_output(self, image, outputs):
        '''
        TODO: You will need to complete this method.
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        coords = []
        for box in outputs[0][0]:
            if box[2] > 0.5:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                coords.append([xmin,ymin,xmax,ymax])
                break

        return image, coords

    def clean(self):
        """
        Deletes all the instances
        :return: None
        """
        del self.network
        del self.exec_network
