def detect_vehicles(frame, yolo):
    results = yolo(frame)
    detections = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            name = yolo.names[cls]

            if name in ["car","motorbike","bus","truck"]:
                x1,y1,x2,y2 = map(int, b.xyxy[0])
                detections.append((x1,y1,x2,y2,name))

    return detections