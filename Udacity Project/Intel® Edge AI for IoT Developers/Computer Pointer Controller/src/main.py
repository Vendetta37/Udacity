import os
import sys
import json
import time
import cv2

import argparse
import logging
from input_feeder import InputFeeder
from face_detection import FaceDetection
from head_pose_estimation import HeadPoseEstimation
from facial_landmark_detection import FaceLandmarksDetection
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController

def main(args):
    """
    Load the network and parse the output.
    :return: None
    """
    logging.basicConfig(filename="test.log", filemode="w",
            format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
            datefmt="%d-%m-%Y %H:%M:%S", level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    face_model = FaceDetection(args.facemodel, args.device, args.extension)
    pose_model = HeadPoseEstimation(args.posemodel, args.device, args.extension)
    land_model = FaceLandmarksDetection(args.landmarksmodel, args.device, args.extension)
    gaze_model = GazeEstimation(args.gazemodel, args.device, args.extension)
    video_path = args.input
    output_path = args.output
    input_type = args.type
    controller = MouseController("medium", "fast")

    start_model_load_time = time.time()
    face_model.load_model()
    pose_model.load_model()
    land_model.load_model()
    gaze_model.load_model()
    total_model_load_time = time.time() - start_model_load_time
    logger.info(f"Time taken to load model = {total_model_load_time} seconds")
    print(f"Time taken to load model = {total_model_load_time} seconds")
    print("All models are loaded successfully")

    feed = InputFeeder(input_type, video_path)
    try:
        cap = feed.load_data()
        logger.info("Video succesfully loaded")
        print("Video succesfully loaded")
    except FileNotFoundError:
        logger.error(f"Cannot locate video file: {video_path}")
        print("Cannot locate video file: "+ video_path)
    except Exception as e:
        logger.error(f"Something else went wrong with the video file: {e}")
        print("Something else went wrong with the video file: ", e)

    initial_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = 0x7634706d

    out_video = cv2.VideoWriter(os.path.join(output_path, "output.mp4"),
                          fourcc, fps, (initial_w, initial_h), True)

    if args.visualization:
        out_fm = cv2.VideoWriter(os.path.join(output_path, "output_fm.mp4"),
                                 fourcc, fps, (initial_w, initial_h), True)
        out_lm = cv2.VideoWriter(os.path.join(output_path, "output_lm.mp4"),
                                 fourcc, fps, (initial_w, initial_h), True)
        out_pm = cv2.VideoWriter(os.path.join(output_path, "output_pm.mp4"),
                                 fourcc, fps, (initial_w, initial_h), True)
        out_gm = cv2.VideoWriter(os.path.join(output_path, "output_gm.mp4"),
                                 fourcc, fps, (initial_w, initial_h), True)

    counter = 0
    start_inference_time = time.time()

    try:
        for frame in feed.next_batch():
            counter += 1

            coords, frame = face_model.predict(frame)

            if args.visualization:
                out_fm.write(frame)

            if len(coords) != 0:
                [xmin,ymin,xmax,ymax] = coords[0]
                head_pose = frame[ymin:ymax, xmin:xmax]
                is_looking, pose_angles = pose_model.predict(head_pose)

                if args.visualization:
                    out_pm.write(frame)

                if is_looking:
                    land_frame, eye_box, eye_coords = land_model.predict(head_pose)
                    frame[ymin:ymax, xmin:xmax] = land_frame

                    if args.visualization:
                        out_lm.write(frame)

                    [[xlmin,ylmin,xlmax,ylmax], [xrmin,yrmin,xrmax,yrmax]] = eye_box
                    left_image = land_frame[ylmin:ylmax, xlmin:xlmax]
                    right_image = land_frame[yrmin:yrmax, xrmin:xrmax]

                    tar_xy, gaze_vector, gaze_coords = gaze_model.predict(left_image, right_image, pose_angles)

                    gaze_x = int(gaze_coords[0]*land_frame.shape[1])
                    gaze_y = int(gaze_coords[1]*land_frame.shape[0])
                    cv2.line(land_frame, tuple(eye_coords[:2]), (gaze_x, -gaze_y), (255,255,0), 2)
                    cv2.line(land_frame, tuple(eye_coords[2:4]), (gaze_x, -gaze_y), (255,255,0), 2)
                    if args.visualization:
                        out_gm.write(frame)

                    if counter%10 == 0:
                        controller.move(tar_xy[0], tar_xy[1])

            out_video.write(frame)

        total_time = time.time() - start_inference_time
        total_inference_time = round(total_time, 1)
        fps = counter / total_inference_time
        logger.info(f"Total time taken to run inference is = {total_inference_time} seconds FPS = {fps}")
        print(f"Total time taken to run inference is = {total_inference_time} seconds FPS = {fps}")

        with open(os.path.join(output_path, 'stats.txt'), 'w') as f:
            f.write(f'total_model_load_time: {total_model_load_time}'+'\n')
            f.write(f'total_inference_time: {total_inference_time}'+'\n')
            f.write(f'fps: {fps}')

        face_model.clean()
        pose_model.clean()
        land_model.clean()
        gaze_model.clean()
        cap.release()
        out_video.release()
        cv2.destroyAllWindows()

        if args.visualization:
            out_fm.release()
            out_pm.release()
            out_lm.release()
            out_gm.release()

    except Exception as e:
        logger.error(f"Could not run inference: {e}")
        print("Could not run inference: " , e)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fm', '--facemodel',
            default='./intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
            type=str, help='path of face detection model')
    parser.add_argument('-pm', '--posemodel',
            default='./intel/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001',
            type=str, help='path of pose detection model')
    parser.add_argument('-lm', '--landmarksmodel',
            default='./intel/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009',
            type=str, help='path of landmarks detection model')
    parser.add_argument('-gm', '--gazemodel',
            default='./intel/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002',
            type=str, help='path of gaze detection model')
    parser.add_argument('-d', '--device', default='CPU', type=str,
            help='type of inference device')
    parser.add_argument('-e', '--extension', default=None, type=str,
            help='path of CPU extension if applicable')
    parser.add_argument('-i', '--input', default='./bin/demo.mp4', type=str,
            help='path of input(video, cam or image)')
    parser.add_argument('-t', '--type', default='video', type=str,
            help='type of input (video, cam or image)')
    parser.add_argument('-o', '--output', default='./results', type=str,
            help='path of output directory')
    parser.add_argument('-vi', '--visualization', default=False, type=bool,
            help='Visualize every inference model')

    args=parser.parse_args()

    main(args)
    sys.exit()
