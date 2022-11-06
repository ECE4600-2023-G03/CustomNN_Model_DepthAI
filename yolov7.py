#!/usr/bin/env python3



from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json

# Get argument first
nnPath = str((Path(__file__).parent / Path('./result/yolov7tiny_openvino_2021.4_6shave.blob')).resolve().absolute())

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')


file = open('result/yolov7tiny.json', 'r');

model_params_text = file.read();

model_dict = json.loads(model_params_text);


syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutVideo = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutVideo.setStreamName("video")
xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")


# Properties

camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
camRgb.setInterleaved(False)
camRgb.setVideoSize(640, 640)
camRgb.setPreviewSize(1920, 1080)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

xoutVideo.input.setBlocking(False)
xoutVideo.input.setQueueSize(1)

# Network specific settings
detectionNetwork.setConfidenceThreshold(model_dict['nn_config']['NN_specific_metadata']['confidence_threshold'])
detectionNetwork.setNumClasses(model_dict['nn_config']['NN_specific_metadata']['classes'])
detectionNetwork.setCoordinateSize(model_dict['nn_config']['NN_specific_metadata']['coordinates'])
detectionNetwork.setAnchors(model_dict['nn_config']['NN_specific_metadata']['anchors'])
detectionNetwork.setAnchorMasks(model_dict['nn_config']['NN_specific_metadata']['anchor_masks'])
detectionNetwork.setIouThreshold(model_dict['nn_config']['NN_specific_metadata']['iou_threshold'])
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

labelMap = model_dict['mappings']['labels']

# Linking
camRgb.video.link(xoutVideo.input)
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
    video = device.getOutputQueue(name="video", maxSize=1, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
            videoIn = video.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frameRGB = inRgb.getCVFrame()

        if videoIn is not None:
            frame = videoIn.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("video", frame)

        if frameRGB is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
