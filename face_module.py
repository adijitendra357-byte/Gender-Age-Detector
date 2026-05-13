import cv2

def detect_faces(frame, faceNet):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > 0.7:
            x1 = int(detections[0,0,i,3]*w)
            y1 = int(detections[0,0,i,4]*h)
            x2 = int(detections[0,0,i,5]*w)
            y2 = int(detections[0,0,i,6]*h)
            boxes.append([x1,y1,x2,y2])

    return boxes