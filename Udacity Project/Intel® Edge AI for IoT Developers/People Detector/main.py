import os
import sys
import time
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3002
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", default="models/frozen_inference_graph",
                        type=str, help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", default='resources/Pedestrian_Detect_2_1_1.mp4',
                        type=str, help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.001,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    input_path = args.input
    infer_network = Network(args.model, args.device, args.cpu_extension, args.prob_threshold)

    infer_network.load_model()

    try:
        if input_path == 'cam':
            cap = cv2.VideoCapture(0)
            print("Cam successfully loaded")
        elif input_path == 'image':
            cap = cv2.imread(input_path)
            print("Image successfully loaded")
        else:
            cap = cv2.VideoCapture(input_path)
            print("Video successfully loaded")
    except FileNotFoundError:
        print("Cannot locate video file: " + input_path)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = 0x7634706d

    out_video = cv2.VideoWriter('output.mp4', fourcc, fps, (initial_w, initial_h), True)

    counter = 0
    total_count = 0
    enter_time = 0
    leave_time = 0
    real_enter_time = 0
    real_leave_time = 0
    duration = 0
    incident_flag = False

    try:
        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                break
            counter+=1

            coords, frame = infer_network.predict(frame)
            current_count = len(coords)
            if current_count != 0 and not incident_flag:
                enter_time = counter / 10
                incident_flag = True
            elif current_count == 0 and incident_flag:
                incident_stop = counter / 10
                incident_flag = False
                if (incident_stop - enter_time) > 5:
                    leave_time = counter / 10
                    if (leave_time - enter_time) > 2:
                        real_enter_time = enter_time
                        real_leave_time = leave_time
                        total_count += 1
                        duration = int(real_leave_time - real_enter_time)
                        client.publish("person", json.dumps({"current_count": current_count,
                                                             "total_count": total_count}))
                        client.publish("person/duration", json.dumps({"duration": duration}))
                        sys.stdout.buffer.write(frame)
                        sys.stdout.flush()

            cv2.putText(frame, f"Current Count: {current_count}", (10, 30),  cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Total Count: {total_count}", (10, 70),  cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Duration: {duration}", (10, 110),  cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
            out_video.write(frame)

        print("Inference completed")
        cap.release()
        cv2.destroyAllWindows()
        client.disconnect()
    except Exception as e:
        print("Could not run Inference: ", e)

def main():
    args = build_argparser().parse_args()
    client = connect_mqtt()
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
