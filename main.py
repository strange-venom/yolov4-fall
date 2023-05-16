import numpy as np
import cv2
import os
from twilio.rest import Client

MODEL_PATH = "yolov4-coco"
display = 1
MIN_CONF = 0.3
NMS_THRESH = 0.3
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = os.path.sep.join([MODEL_PATH, "yolov4.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov4.cfg"])
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
src = cv2.cuda_GpuMat()
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
print("Accessing video stream...")

vs = cv2.VideoCapture(0)
writer = None


def detect_people(frame, net, ln, personIdx=0):
    (H, W) = frame.shape[:2]
    results = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    boxes = []
    centroids = []
    confidences = []
    BoxCoordinates = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if classID == personIdx and confidence > MIN_CONF:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
    return results


j = 0
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    Fwidth = 700
    Fheight = frame.shape[0]
    dim = (Fwidth, Fheight)
    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"))
    for (i, (prob, bbox, centroid)) in enumerate(results):
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)
        x, y = startX, startY
        w = endX - startX
        h = endY - startY
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)
        print(h, " ", w)
        if h < (0.75 * w):
            j += 1
        # Conditions for fall detection contour Area
        if j > 2:
            cv2.putText(frame, 'FALLEN', (int(x + 0.5 * w), y - 5), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            account_sid = 'ACd664760de7d34adae3ed44ab92c7a7eb'
            # Twilio Account Auth Token
            auth_token = 'fe3250c4fdfd44385de5e55c5bc5fd6b'
            # Initialise the client
            client = Client(account_sid, auth_token)
            # Creation of Message API
            message = client.messages.create(to="+919103555145", from_="+16506996568",
                                             body="Your Kid or elder has fallen Down")
        if h >= (0.75 * w):
            j = 0
        print(j)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
vs.release()
cv2.destroyAllWindows()