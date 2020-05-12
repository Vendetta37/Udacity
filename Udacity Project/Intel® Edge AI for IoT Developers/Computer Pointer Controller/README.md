# Computer Pointer Controller
In this project we use a Gaze Detection model to control the mouse pointer of your computer. To be more specific, we use four models, Face Detection, Head Pose Estimation, Facial Landmarks Detection and Gaze Estimation to create a pipeline to do all the inference works. The output of the model can be used to control the mouse.

## Project Set Up and Installation
1. Install `Conda` [Link](https://www.anaconda.com/products/individual)
2. Install `OpenVINO` [Link](https://docs.openvinotoolkit.org/2020.1/_docs_install_guides_installing_openvino_linux.html)
3. Create a virtual environment for the project `conda create -n {env_name} python=3.6`
4. Activate the environment `conda activate {env_name}`
5. Install required packages `pip install -r requirements.txt`
6. Install numpy `conda install numpy`
7. Install PyAutoGUI `conda install -c conda-forge pyautogui`
8. Download all the required models

```
python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name face-detection-adas-binary-0001

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name head-pose-estimation-adas-0001

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name landmarks-regression-retail-0009

python3 /opt/intel/openvino/deployment_tools/tools/model_downloader/downloader.py --name gaze-estimation-adas-0002
```

## Demo
1. Clone this repo
2. Run `python main.py` in homedirectory

## Command Line Options
1. `-fm` Path of face detection model
2. `-pm` Path of pose detection model
3. `-lm` Path of landmarks detection model
4. `-gm` Path of gaze detection model
5. `-vi` Visualize every inference model
6. `-d`  Type of inference device
7. `-e`  Path of CPU extension if applicable
8. `-i`  Path of input(video, cam or image)
9. `-t`  Type of input (video, cam or image)
10. `-o` Path of output directory

## Project Directory Structure

```
├── README.md
├── requirements.txt
└── src
    ├── bin
    │   └── demo.mp4
    ├── face_detection.py
    ├── facial_landmark_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── intel
    │   ├── face-detection-adas-binary-0001
    │   │   └── FP32-INT1
    │   │       ├── face-detection-adas-binary-0001.bin
    │   │       └── face-detection-adas-binary-0001.xml
    │   ├── gaze-estimation-adas-0002
    │   │   ├── FP16
    │   │   │   ├── gaze-estimation-adas-0002.bin
    │   │   │   └── gaze-estimation-adas-0002.xml
    │   │   ├── FP16-INT8
    │   │   │   ├── gaze-estimation-adas-0002.bin
    │   │   │   └── gaze-estimation-adas-0002.xml
    │   │   └── FP32
    │   │       ├── gaze-estimation-adas-0002.bin
    │   │       └── gaze-estimation-adas-0002.xml
    │   ├── head-pose-estimation-adas-0001
    │   │   ├── FP16
    │   │   │   ├── head-pose-estimation-adas-0001.bin
    │   │   │   └── head-pose-estimation-adas-0001.xml
    │   │   ├── FP16-INT8
    │   │   │   ├── head-pose-estimation-adas-0001.bin
    │   │   │   └── head-pose-estimation-adas-0001.xml
    │   │   └── FP32
    │   │       ├── head-pose-estimation-adas-0001.bin
    │   │       └── head-pose-estimation-adas-0001.xml
    │   └── landmarks-regression-retail-0009
    │       ├── FP16
    │       │   ├── landmarks-regression-retail-0009.bin
    │       │   └── landmarks-regression-retail-0009.xml
    │       ├── FP16-INT8
    │       │   ├── landmarks-regression-retail-0009.bin
    │       │   └── landmarks-regression-retail-0009.xml
    │       └── FP32
    │           ├── landmarks-regression-retail-0009.bin
    │           └── landmarks-regression-retail-0009.xml
    ├── main.py
    ├── mouse_controller.py
    ├── results
    │   ├── output_fm.mp4
    │   ├── output_gm.mp4
    │   ├── output_lm.mp4
    │   ├── output.mp4
    │   └── output_pm.mp4
    └── test.log
```

## Benchmarks
Benchmarking results for models of different precisions
### FP32
#### Inference Time
![fp32_inference_time](https://github.com/Vendetta37/Computer_Pointer_Controller/blob/master/src/benchmarks/fp32_inference_time.png)

#### FPS
![fp32_frames_ps](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/fp32_frames_ps.png)

#### Model Loading Time
![fp32_model_load__time](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/fp32_model_load__time.png)

### FP16
#### Inference Time
![fp16_inference_time](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/fp16_inference_time.png)

#### FPS
![fp16_frames_ps](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/fp16_frames_ps.png)

#### Model Loading Time
![fp16_model_load__time](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/fp16_model_load__time.png)

### INT8
#### Inference Time
![int8_inference_time](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/int8_inference_time.png)

#### FPS
![int8_frames_ps](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/int8_frames_ps.png)

#### Model Loading Time
![int8_model_load__time](https://github.com/Vendetta37/Udacity/blob/master/Udacity%20Project/Intel%C2%AE%20Edge%20AI%20for%20IoT%20Developers/Computer%20Pointer%20Controller/src/benchmarks/int8_model_load__time.png)

## Results
As we can see, INT8 gives us the worst result in all aspects. FP16 and FP32 give very similar results. As FP32 has higher accuracy than FP16, it is highly recommended.
