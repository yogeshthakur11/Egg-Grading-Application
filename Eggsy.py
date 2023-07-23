
import os
os.chdir(r'C:\Users\Yogesh Thakur\obj_detection')
# Importing necessary libraries for object detection
import cv2
import numpy as np
import math

# Storing the model weights and classes in variables
modelWeights = r"C:\Users\Yogesh Thakur\obj_detection\vib.onnx"

# Defining a function to draw, detect, and count eggs
def detect_objects(filename, stframe):
    classes = ['Brown Egg', 'White Egg']
    cap = cv2.VideoCapture(filename)
    
    net = cv2.dnn.readNet(modelWeights)
    white_counter = 0
    brown_counter = 0
    center_pt_prev_frame = []
    tracking_object = {}
    track_id = 1
    obj_id1 = 0
    offset = 30
    counter = 0
    j = 0
    k = []
    
    
    while True:
        _, frame = cap.read()
        if frame is None:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = cv2.dnn.blobFromImage(frame, 1/255,  (416, 416), [0, 0, 0], 1, crop=False)
        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        
        rows = outputs[0].shape[1]
        image_height, image_width = frame.shape[:2]
        
        # Resizing factor.
        x_factor = image_width / 416
        y_factor = image_height / 416
        
        # Iterate through detections.
        class_ids = []
        confidences = []
        boxes = []
        center = []
        detect = []
        center_pt_cur_frame = []
        cv2.line(frame, (0, image_height // 2), (image_width, image_height // 2), (0, 255, 0), 3)
        
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            
            # Discard bad detections and continue.
            if confidence >= 0.20:
                classes_scores = row[5:]
                
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                
                # Continue if the class score is above threshold.
                if classes_scores[class_id] > 0.5:
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
    
                    left = int((cx - w/2) * x_factor)
                    top = int((cy - h/2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
    
                    box = np.array([left, top, width, height])
                    boxes.append(box)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.50, 0.45)
        
        for i in indices:
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            cx = int((left + left + width) / 2)
            cy = int((top + top + height) / 2)
            center_pt_cur_frame.append((cx, cy))
            
            class_id = class_ids[i]  # Get the class ID
            label = "{}:{:.2f}".format(classes[class_id], confidences[i])
            
            if classes[class_id] == 'White Egg':
                color = (255, 140, 0)  # Blue color for white eggs
            elif classes[class_id] == 'Brown Egg':
                color = (139, 0, 139)  # Red color for brown eggs
            
            cv2.rectangle(frame, (left, top), (left + width, top + height), color, 4)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(frame, (left, top), (left + dim[0], top + dim[1] + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, label, (left, top + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
            center = cx, cy
            detect.append(center)
        
        if len(center_pt_prev_frame) == 0:
            for pt in center_pt_cur_frame:
                for pt2 in center_pt_prev_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    
                    if distance < 50:
                        tracking_object[track_id] = pt
                        track_id += 1
        else:
            tracking_object_copy = tracking_object.copy()
            center_pt_cur_frame_copy = center_pt_cur_frame.copy()
            
            for obj_id, pt2 in tracking_object_copy.items():
                object_exists = False
                
                for pt in center_pt_cur_frame:
                    distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                    
                    if distance < 50:
                        tracking_object[obj_id] = pt
                        
                        for i in tracking_object.keys():
                            if tracking_object[i][1] < (image_height // 2 + offset) and tracking_object[i][1] > (image_height // 2 - offset):
                                if i not in k:
                                    if classes[class_ids[i]] == 'White Egg':
                                        white_counter += 1
                                    elif classes[class_ids[i]] == 'Brown Egg':
                                        brown_counter += 1
                                    k.append(i)
                        
                        object_exists = True
                        if pt in center_pt_cur_frame:
                            center_pt_cur_frame.remove(pt)
                            continue
            
            for pt in center_pt_cur_frame:
                tracking_object[track_id] = pt
                track_id += 1
        
        for i in list(tracking_object.keys()):
            if tracking_object[i][1] > 7 * image_height // 8:
                for obj_id, pt in tracking_object.items():
                    cv2.circle(frame, pt, 1, (0, 0, 255), -1)
                    obj_id1 = obj_id
                    tracking_object[i] = (0, 0)
        
        cv2.putText(frame, "White Eggs: " + str(white_counter), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Brown Eggs: " + str(brown_counter), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Total Eggs: " + str(white_counter + brown_counter), (20, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        vid = frame
        scale_percent = 50
        width = int(vid.shape[1] * scale_percent / 100)
        height = int(vid.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(vid, dim, interpolation=cv2.INTER_AREA)
        stframe.image(resized)
        
        center_pt_prev_frame = center_pt_cur_frame.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()    
    cv2.destroyAllWindows()


# Usage example
#filename = 'path/to/video/file.mp4'
#detect_objects(filename, stframe)  # Replace stframe with the appropriate frame object
