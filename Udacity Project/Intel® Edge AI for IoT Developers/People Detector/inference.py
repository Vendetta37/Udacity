import os
import sys
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and configure inference plugins for the specified target devices
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self, model_name, device, extensions, threshold):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_structure = model_name + '.xml'
        self.model_weights = model_name + '.bin'
        self.device = device
        self.extensions = extensions
        self.threshold = threshold

        try:
            self.network = IENetwork(self.model_structure, self.model_weights)
        except Exception as e:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

        self.input_name=next(iter(self.network.inputs))
        self.input_shape=self.network.inputs[self.input_name].shape
        self.output_name=next(iter(self.network.outputs))

    def load_model(self):
        self.core = IECore()
        self.check_model()
        self.exec_network = self.core.load_network(self.network, self.device, num_requests=1)

    def predict(self, image):
        input_dict = self.preprocess_input(image)
        self.exec_network.start_async(request_id=0, inputs=input_dict)
        if self.exec_network.requests[0].wait(-1) == 0:
            outputs = self.exec_network.requests[0].outputs[self.output_name]
            coords, image = self.preprocess_output(image, outputs)
            return coords, image

    def check_model(self):
        supported_layers = self.core.query_network(network=self.network, device_name=self.device)

        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

    def preprocess_input(self, image):
        p_image = cv2.resize(image,  (self.input_shape[3], self.input_shape[2]))
        p_image = p_image.transpose((2,0,1))
        p_image = p_image.reshape(1, *p_image.shape)
        input_dict = {self.input_name:p_image}
        return input_dict

    def preprocess_output(self, image, outputs):
        coords = []
        for box in outputs[0][0]:
            if box[2] > self.threshold:
                xmin = int(box[3] * image.shape[1])
                ymin = int(box[4] * image.shape[0])
                xmax = int(box[5] * image.shape[1])
                ymax = int(box[6] * image.shape[0])

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 3)
                coords.append([xmin,ymin,xmax,ymax])

        return coords, image
