import numpy as np

def update_tracks(detections, tracks, track_id, line_y, vehicle_count):
    new_tracks = {}

    for x1,y1,x2,y2,name in detections:
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        matched = None

        for tid,(px,py,done) in tracks.items():
            dist = np.sqrt((cx-px)**2 + (cy-py)**2)
            if dist < 60:
                matched = tid
                break

        if matched is None:
            matched = f"id_{track_id}"
            track_id += 1

        prev_done = tracks.get(matched,(0,0,False))[2]

        if cy > line_y and not prev_done:
            vehicle_count += 1
            prev_done = True

        new_tracks[matched] = (cx,cy,prev_done)

    return new_tracks, track_id, vehicle_count