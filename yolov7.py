#!/usr/bin/env python3



from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json

# Get argument first
nnPath = str((Path(__file__).parent / Path('./DepthAI_Conversion/yolov7tiny_openvino_2021.4_6shave.blob')).resolve().absolute())

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')



file = open('DepthAI_Conversion/yolov7tiny.json', 'r');

model_params_text = file.read();

model_dict = json.loads(model_params_text);


syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
#stereo = pipeline.create(dai.node.StereoDepth)
manip = pipeline.create(dai.node.ImageManip)
spatialDetectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)

nnOut = pipeline.create(dai.node.XLinkOut)
#disparityOut = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)

#disparityOut.setStreamName("disparity")
xoutRight.setStreamName("rectifiedRight")
nnOut.setStreamName("nn")



monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

#stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
#stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout from rectification (black stripe on the edges)
# Convert the grayscale frame into the nn-acceptable form
manip.initialConfig.setResize(640, 640)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGRF16F16F16p)         # might not be right

#stereo.setOutputSize(640, 640)


# Network specific settings
spatialDetectionNetwork.setConfidenceThreshold(model_dict['nn_config']['NN_specific_metadata']['confidence_threshold'])
spatialDetectionNetwork.setNumClasses(model_dict['nn_config']['NN_specific_metadata']['classes'])
spatialDetectionNetwork.setCoordinateSize(model_dict['nn_config']['NN_specific_metadata']['coordinates'])
spatialDetectionNetwork.setAnchors(model_dict['nn_config']['NN_specific_metadata']['anchors'])
spatialDetectionNetwork.setAnchorMasks(model_dict['nn_config']['NN_specific_metadata']['anchor_masks'])
spatialDetectionNetwork.setIouThreshold(model_dict['nn_config']['NN_specific_metadata']['iou_threshold'])
spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setNumInferenceThreads(2)
spatialDetectionNetwork.input.setBlocking(False)



labelMap = model_dict['mappings']['labels']

# Linking
#stereo.rectifiedRight.link(manip.inputImage)
#stereo.disparity.link(disparityOut.input)
manip.out.link(xoutRight.input)
manip.out.link(spatialDetectionNetwork.input)
spatialDetectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRight = device.getOutputQueue("rectifiedRight", maxSize=4, blocking=False)
    #qDisparity = device.getOutputQueue("disparity", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("nn", maxSize=4, blocking=False)


    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)
    printOutputLayersOnce = True

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    #disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

    rightFrame = None
    while True:
        # Instead of get (blocking), we use tryGet (non-blocking) which will return the available data or None otherwise
        if qDet.has():
            detections = qDet.get().detections

        if qRight.has():
            rightFrame = qRight.get().getCvFrame()

        # if qDisparity.has():
        #     # Frame is transformed, normalized, and color map will be applied to highlight the depth info
        #     disparityFrame = qDisparity.get().getFrame()
        #     disparityFrame = (disparityFrame*disparityMultiplier).astype(np.uint8)
        #     # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        #     disparityFrame = cv2.applyColorMap(disparityFrame, cv2.COLORMAP_JET)
        #     displayFrame("disparity", disparityFrame)

        if rightFrame is not None:
            displayFrame("rectified right", rightFrame)

        if cv2.waitKey(1) == ord('q'):
            break
