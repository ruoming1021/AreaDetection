import cv2
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
from shapely.geometry import Polygon, box
import threading
import numpy as np
import http.client
import base64
import time 
import pygame
pygame.init()
pygame.mixer.init()
drawing = False
show_item = False
sound_played_in_current_frame = False
show = True
get_time = ""
points = []
polygons = [] # save finish pologons
bbox_timestamps = {}  
model = YOLO("/home/vip/Documents/PyQt_Demo/weights/yolov8m/yolov8m.engine") # for RGB
# model = YOLO("/home/vip/Documents/PyQt_Demo/Thermal_dataset/weights/best.engine") # for Thermal
current_frame1 = None
current_frame2 = None
continuous_frames = dict()

# Linin setting  
s_username = 'admin'
s_password = 'Pass1234'
s_count = s_username + ':' + s_password
encoded_u = base64.b64encode(s_count.encode()).decode()
headers = {"Authorization" : "Basic %s" % encoded_u}
connection = http.client.HTTPConnection('192.168.50.210', 80, timeout=10)
connection.request('GET', '/system?get=server', headers=headers)

s_URL = '192.168.50.210'
i_http_port = 80
i_rtsp_port = 554
i_rtsp_stream = 'stream1'


def find_homography():
    # Origin
    # RGB_points = np.array([[287,192],[341,216],[322,281],[256,251]])
    # Thermal_points = np.array([[153,9],[378,63],[327,253],[47,188]])

    # Zoom_in_max
    RGB_points = np.array([[216,141],[369,264],[283,452],[94,296]])
    Thermal_points = np.array([[178,90],[385,187],[288,365],[34,244]])

    thermal_check, s2 = cv2.findHomography(Thermal_points,RGB_points) 

    return thermal_check

def read_camera1():
    global current_frame1
    while True:
        ret, frame = cap.read()
        if ret:
            current_frame1 = frame

def read_camera2():
    global current_frame2
    while True:
        ret, frame = cap2.read()
        if ret:
            current_frame2 = frame
def on_left_click(event):
    global drawing, points
    if drawing:
        x, y = event.x, event.y
        points.append((x, y))

def on_double_click(event):
    global drawing, points, polygons
    if drawing and len(points) > 2:
        polygons.append(points)
        bbox_coords = get_polygon_bbox(points)
        points = []

def get_polygon_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    x1, y1, x2, y2 = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
    return (x1, y1, x2, y2)

def get_overlapping_bbox_label(box_coords, polygons):
    bbox = box(*box_coords)
    for poly_coords in polygons:
        poly = Polygon(poly_coords)
        if bbox.intersects(poly):
            label = "Overlapping with polygon"
            return label
    x1, y1, x2, y2 = box_coords
    label = f"Box at ({x1}, {y1}) - ({x2}, {y2})"
    return label

def draw_polygons(image, points_list):
    for points in points_list:
        if len(points) > 1:
            for i in range(len(points) - 1):
                cv2.line(image, points[i], points[i + 1], (255, 0, 0), 2)
            cv2.line(image, points[-1], points[0], (255, 0, 0), 2)

def is_overlapping(box_coords, polygons):
    bbox = box(*box_coords)
    for poly_coords in polygons:
        poly = Polygon(poly_coords)
        if poly.intersects(bbox):
            return True
    return False

def toggle_drawing():
    global drawing
    drawing = not drawing

def clear_polygons():
    global polygons
    polygons = []
    polygon_coords_text.delete(1.0, tk.END)

def show_target():
    global show
    if show == True:
        button3.config(text='All show')
        show = False
    else:
        button3.config(text='只顯示違規物件')
        show = True
    
def distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
def draw_results(frame, results):
    first_result = results[0]    
    current_time = time.time()
    global get_time
    global show_item
    global sound_played_in_current_frame
    deteced_object = set()
    num_person = 0
    num_car = 0
    num_bus = 0
    if first_result.boxes is not None:
        boxes = first_result.boxes.xyxy.cpu().numpy()
        confs = first_result.boxes.conf.cpu().numpy()
        class_id = first_result.boxes.cls.cpu().numpy()
        bbox_label = []
        for i, box in enumerate(boxes):
            if box.ndim == 1 and confs.ndim == 1 and class_id.ndim == 1:
                x1, y1, x2, y2, score, cls_id = box[0], box[1], box[2], box[3], confs[i], class_id[i]
                label = first_result.names[int(cls_id)]
                if label == "person" or label == "car" or label == "motorcycle":
                    color = (0, 255, 0) if not is_overlapping((x1, y1, x2, y2), polygons) else (0, 0, 255)
                    if is_overlapping((x1, y1, x2, y2), polygons):
                        connection = http.client.HTTPConnection('192.168.50.210', 80, timeout=10)
                        connection.request('GET', '/system?get=server', headers=headers)
                        response = connection.getresponse()
                        data = response.read()
                        data_decoded = data.decode("utf-8")
                        get_time = data_decoded.split("\n")[2]
                        # Update the timestamp of the bounding box
                        center = ((x1 + x2) //2, (y1 + y2) // 2)
                        key = (label, center)
                        # deteced_object.add(key)
                        replaced = False
                        for prev_key in bbox_timestamps.keys():
                            # print("prev_key",prev_key)
                            # print("key",key)
                            # print("distance(prev_key[1],key[1])",distance(prev_key[1],key[1]))
                            # check object is same or different, threshold set 3
                            if distance(prev_key[1],key[1]) <= 5 and prev_key[0] == key[0]:
                                count = bbox_timestamps[prev_key][1] + 1
                                del bbox_timestamps[prev_key]
                                bbox_timestamps[key] = [current_time, count]
                                replaced = True
                                break
                        # if current_time - bbox_timestamps[key][0] >= 1:
                        #         bbox_timestamps[key][0] = current_time
                        if not replaced:
                            bbox_timestamps[key] = [current_time, 1]
                        # print("-*----",bbox_timestamps[key])
                        # print("####################",bbox_timestamps[key][1])
                        # If the count of detections within 3 seconds is 3 or more, play the sound
                        # check object detect frequency, threshold set 40 as 1 second
                        if bbox_timestamps[key][1] >= 20 and not sound_played_in_current_frame:
                            pygame.mixer.music.load("/home/vip/Documents/PyQt_Demo/Sound_2.mp3")
                            pygame.mixer.music.play()
                            # pygame.time.delay(3000)
                            show_item = True
                            sound_played_in_current_frame = True
                            bbox_timestamps.clear()
                            # polygon_coords_text.insert(tk.END, f"{tmp}\n") 
                        bbox_label.append(label)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"{label}: {score:.2f}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:
                        if show == True:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            cv2.putText(frame, f"{label}: {score:.2f}", (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # for key in list(bbox_timestamps.keys()):
        #     if key not in deteced_object:
        #         del bbox_timestamps[key]
        
        # text = polygon_coords_text.get("1.0", tk.END)
        # print("####################",bbox_label)
        if len(bbox_label) > 0 and show_item == True:
            # polygon_coords_text.delete(1.0, tk.END)
            for l in bbox_label:
                if l == "person":
                    num_person = num_person + 1
                elif l == "car":
                    num_car = num_car + 1
                elif  l == "bus":
                   num_bus = num_bus + 1 
            messege = get_time[5:29] + "Person" + "*" + str(num_person) + ", " + "Car" + "*" + str(num_car) + ", " + "Bus" + "*" + str(num_bus) 
            polygon_coords_text.insert(tk.END, f"{messege}\n") 
            polygon_coords_text.see(tk.END)
            show_item = False
    else:
        print("No boxes found")
    return frame
def on_key_press(event):
    if event.keysym == "Escape":
        root.destroy()
def update_all_image_labels():
    global current_frame1, current_frame2, sound_played_in_current_frame
    while True:
        sound_played_in_current_frame = False
        frame1 = current_frame1
        frame2 = current_frame2
        if frame1 is not None and frame2 is not None:
            frame1 = cv2.resize(frame1, (640, 480))
            frame2 = cv2.resize(frame2, (640, 480))
            h_t = find_homography()

            # Update image_label1
            frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            pil_img1 = Image.fromarray(frame1_rgb)
            tk_img1 = ImageTk.PhotoImage(pil_img1)
            image_label1.config(image=tk_img1)
            image_label1.image = tk_img1
            
            # Update image_label2
            pil_img2 = Image.fromarray(frame2)
            tk_img2 = ImageTk.PhotoImage(pil_img2)
            image_label2.config(image=tk_img2)
            image_label2.image = tk_img2
            
            # Update image_label3
            process_thermal = cv2.warpPerspective(frame2, h_t, (frame2.shape[1], frame2.shape[0]))
            combined_frame = cv2.addWeighted(frame1_rgb, 0.5, process_thermal, 0.7, 0)

            results = model(frame1_rgb,device=0) # only infer rgb input
            # results = model(process_thermal,device=0) # only infer thermal input
            process_frame = draw_results(combined_frame,results)
            draw_polygons(process_frame, polygons)
            if drawing:
                draw_polygons(process_frame, [points])
            pil_img3 = Image.fromarray(process_frame)
            tk_img3 = ImageTk.PhotoImage(pil_img3)
            image_label3.config(image=tk_img3)
            image_label3.image = tk_img3
            
# Linin camera 
cap = cv2.VideoCapture('rtsp://' + str(s_count) + '@' + str(s_URL) + ':' + str(i_rtsp_port) + '/' + str(i_rtsp_stream))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Thermal camera 
cap2 = cv2.VideoCapture(0)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# cap.set(cv2.CAP_PROP_FPS,15)
# cap2.set(cv2.CAP_PROP_FPS,60)

root = tk.Tk()
root.title("智慧安防於禁區偵測之應用")
root.bind("<KeyPress>",on_key_press)
frame1 = tk.Frame(root)
frame1.grid(row=0, column=0)
frame2 = tk.Frame(root)
frame2.grid(row=0, column=1)
frame3 = tk.Frame(root)
frame3.grid(row=1, column=0)
frame4 = tk.Frame(root)
frame4.grid(row=1, column=1)

image_label1 = tk.Label(frame1)
image_label1.pack()
image_label2 = tk.Label(frame2)
image_label2.pack()
image_label3 = tk.Label(frame3)
image_label3.pack()
image_label3.bind("<Button-1>", on_left_click)
image_label3.bind("<Double-Button-1>", on_double_click)

polygon_label = tk.Label(frame4, text="違規物件 ： ")
polygon_label.pack()
polygon_coords_text = tk.Text(frame4, width=52, height=20)
polygon_coords_text.pack()

buttons_frame = tk.Frame(frame4)
buttons_frame.pack(pady=10)
button1 = tk.Button(buttons_frame, text="繪製禁區", command=toggle_drawing,width=8)
button1.grid(row=0, column=0, padx=5)

button2 = tk.Button(buttons_frame, text="清除禁區", command=clear_polygons,width=8)
button2.grid(row=0, column=1, padx=5)

button3 = tk.Button(buttons_frame, text="只顯示違規物件", command=show_target,width=8)
button3.grid(row=0, column=2, padx=5)

thread_camera1 = threading.Thread(target=read_camera1)
thread_camera2 = threading.Thread(target=read_camera2)
thread_update_labels = threading.Thread(target=update_all_image_labels)

thread_camera1.start()
thread_camera2.start()
thread_update_labels.start()

root.mainloop()
cap.release()
cap2.release()