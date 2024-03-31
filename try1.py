from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import atexit
from time import time
from datetime import datetime, timedelta
import pytz
from get_message import getConfig

app = Flask(__name__)
fps=getConfig("config.txt","parameter","fps")
conf=getConfig("config.txt","parameter","conf")
pipeline = None
record_video = False
video_writer = None
frames_processed = 0
current_time_str= datetime.utcnow()
recording_duration = int(getConfig("config.txt","parameter","recording_duration"))  # Video Recording Duration(Seconds)
gap_duration = int( getConfig("config.txt","parameter","gap_duration")) # Seconds
last_recording_time = time()
video_save_path=""
flag=False
vid_status=""

data = {
     "x":0,
     "y": 0,
                                        "z": 0,
                                        "Date": current_time_str,
                                        "Confience":"Null",
                                        "Flag":flag,
                                        "Video Status":vid_status,
                                        "video file Path" : video_save_path
}
pig_id=101

def start_recording(display_image):
    global record_video, video_writer, gap_duration, last_recording_time,vid_status
    record_video = True
    current_time_utc = datetime.utcnow()
    tz = pytz.timezone('Asia/Shanghai')  # Use the China Standard Time zone
    current_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(tz)
    video_save_path = f"/usr/src/ultralytics/tests/gitpl_main/{current_time.strftime('%Y%m%d')}_{pig_id}_{current_time.strftime('%I%M%S')}.mp4"  # Path for recorded Video
    video_writer = cv2.VideoWriter(video_save_path, cv2.VideoWriter_fourcc(*'avc1'), int(fps), (640, 480))
    print("*******************************************************************************Started", video_save_path)
    vid_status="Video Started"
   
    video_writer.write(display_image)
    last_recording_time = time()
    return video_save_path
    
def stop_recording():
    global record_video, video_writer, gap_duration, last_recording_time,vid_status,video_save_path
    record_video = False
    if video_writer is not None:
        print('Released')
        vid_status= "Video Stopped"
        video_writer.release()
        video_writer = None
        gap_duration = 60  # Adjust the gap duration as needed
        video_save_path=""
    print("Stopped *************************************************************************************************")
    return video_save_path   
    


def generate_frames():
    global pipeline, record_video, video_writer, start_time, last_recording_time, data,flag,vid_status,video_save_path,current_time_str,current_time_utc

    W, H = 640, 480
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, int(fps))
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, int(fps))  # Configure depth stream

    profile = pipeline.start(config)

    #model_directory_H = '/usr/src/ultralytics/tests/gitpl_main/best_50.pt'
    model_directory_H=getConfig("config.txt","path","model_directory_H")
    classification_model = YOLO(model_directory_H)
    #model_directory = '/usr/src/ultralytics/tests/gitpl_main/best.pt'  # Path for Model
    model_directory=getConfig("config.txt","path","model_directory")
    img_path = '/usr/src/ultralytics/tests/gitpl_main/frame_24.png'
   
    model = YOLO(model_directory)
    predict1 = model.predict(source=img_path, conf=0.5, boxes=False, device=0)
    seg1 = predict1[0].masks.xy[0]
    start_time = time()
    frames_processed = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not depth_frame:
                continue
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            color_image_resized = cv2.resize(color_image, (640, 360))

            predictions = classification_model.predict(source=color_image_resized, conf=0.5, boxes=False)
            class_labels = classification_model.names

            if predictions:
                for result in predictions:
                    class_name = result.names
                    confidence = result.probs.top1conf
                    top_label = class_labels[result.probs.top1]

                    if top_label == "pig_back" and confidence.item() >= 0.60:  # Confidence Settings
                        if confidence.item() > float(conf):
                            results = model(color_image)

                            for r in results:
                                prediction = r
                                if prediction.masks is not None and prediction.masks.xy is not None and len(
                                        prediction.masks.xy) > 0:
                                    seg = prediction.masks.xy[0]
                                    message = "Start Recording"
                                    print(f"Start Recording")
                                    print("Here is a valid pig image!")
                                    flag=True
                                    cv2.putText(color_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)
                                    cv2.polylines(color_image, [np.int32(prediction.masks.xy[0])],
                                                  isClosed=True,
                                                  color=(0, 0, 255), thickness=2)
                                    depth_colormap = np.zeros((H, W, 3), dtype=np.uint8)

                                    cv2.putText(color_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)

                                    cv2.polylines(color_image, np.int32([seg]), isClosed=True,
                                                  color=(0, 0, 255),
                                                  thickness=2)

                                    x, y, w, h = cv2.boundingRect(np.int32([seg]))

                                    centroid = (x + w // 2, y + h // 2)

                                    cv2.line(color_image, (0, centroid[1]),
                                             (color_image.shape[1], centroid[1]),
                                             color=(0, 255, 0), thickness=2)

                                    midpoint = (centroid[0], centroid[1])
                                    cv2.circle(color_image, midpoint, radius=5, color=(255, 0, 0),
                                               thickness=-1)

                                    left_point_x = centroid[0] - 70
                                    left_point_y = centroid[1]
                                    cv2.circle(color_image, (int(left_point_x), int(left_point_y)), radius=5,
                                               color=(0, 255, 255), thickness=-1)

                                    vertical_line_y = int(left_point_y + 65)

                                    cv2.line(color_image, (int(left_point_x), int(left_point_y)),
                                             (int(left_point_x), vertical_line_y), color=(255, 255, 0),
                                             thickness=2)
                                    cv2.circle(color_image, (int(left_point_x), int(left_point_y + 65)), radius=5,
                                               color=(0, 0, 0), thickness=-1)
                                    dis = depth_frame.get_distance(int(left_point_x), int(left_point_y + 65))
                                    depth_pixel = (int(left_point_x), int(left_point_y + 65))
                                    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)

                                    coordinates_text = f"({int(left_point_x)}, {int(left_point_y + 65)},{round(dis * 1000, 3)})"
                                    cv2.putText(color_image, coordinates_text,
                                                (int(left_point_x) + 10, int(left_point_y + 65)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                                    cv2.putText(color_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)
                                    xyz = f"x:{int(left_point_x)} Y:{int(left_point_y + 65)} Z:{round(dis * 1000, 3)} mm"
                                    print(xyz)
                                    cv2.putText(color_image, xyz, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)
                                    rect_x = int(left_point_x) - 30
                                    rect_y = vertical_line_y - 30
                                    rect_width = 100
                                    rect_height = 50
                                    cv2.rectangle(color_image, (rect_x, rect_y),
                                                  (rect_x + rect_width, rect_y + rect_height),
                                                  color=(0, 255, 255),
                                                  thickness=2)

                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    font_color = (255, 255, 255)
                                    h, w, _ = color_image.shape
                                    text_size = cv2.getTextSize(message, font, 1, 2)[0]
                                    text_x = (w - text_size[0]) // 2
                                    text_y = (h + text_size[1]) // 2
                                    cv2.putText(color_image, message, (text_x, text_y), font, 1, font_color,
                                                2)

                                    # Add current date and time with time zone information
                                    current_time_utc = datetime.utcnow()
                                    tz = pytz.timezone('Asia/Shanghai')  # Use the China Standard Time zone
                                    current_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(tz)
                                    current_time_str = current_time.strftime("%Y-%m-%d %I:%M:%S %p")
                                    print("Date & Time : ", current_time_str)
                                    cv2.putText(color_image, current_time_str, (20, 100),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                                (255, 255, 255), 2)

                                    #display_image = np.hstack((color_image, color_image))
                                    display_image= color_image
                                    _, jpeg = cv2.imencode('.jpg', display_image)
                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

                                    print(record_video, time() - start_time, recording_duration, gap_duration,
                                          time() - last_recording_time)
                                    current_time1 = current_time_utc.replace(tzinfo=pytz.utc).astimezone(tz)
                                    current_time_str1 = current_time1.strftime("%I:%M:%S %p")
                                    
                                    data = {
                                    
                                        "x": int(left_point_x),
                                        "y": int(left_point_y + 65),
                                        "z": round(dis * 1000, 3),
                                        "Date": current_time_str1,
                                        "Confience": confidence.item(),
                                        "Flag":flag,
                                        "Video Status":vid_status,
                                        "video file Path" : video_save_path
                                    }

                                    # Check if it's time to stop recording
                                    if record_video and time() - last_recording_time >= recording_duration:
                                        print("Stopping recording...")
                                        vid_status ="Stopping recording"
                                        video_save_path = stop_recording()

                                    # Start recording if the condition is met
                                    if not record_video and time() >= last_recording_time + recording_duration + gap_duration:
                                        vid_status ="Start recording"
                                        video_save_path= start_recording(color_image)
                                        start_time = time()

                                    # Write frames to the video file if recording is in progress
                                    if record_video:
                                        if video_writer is not None:
                                            video_writer.write(color_image)

                                        # Check if it's time to stop recording
                                        if time() - start_time >= recording_duration + gap_duration:
                                            print("Stopping recording...")
                                            vid_status ="Stopping recording"
                                            video_save_path=stop_recording()

                        else:
                            print("Not a valid pig image!")
                            flag=False
                            current_time_utc = datetime.utcnow()
                            tz = pytz.timezone('Asia/Shanghai') 
                            current_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(tz)
                            current_time_str = current_time.strftime("%I:%M:%S %p")
                            display_image = color_image
                            _, jpeg = cv2.imencode('.jpg', display_image)
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                            data = {
                                       "x":0,
                                       "y": 0,
                                        "z": 0,
                                        "Date": current_time_str,
                                        "Confience":"Null",
                                        "Flag":flag,
                                        "Video Status":"Video Stopped",
                                        "video file Path" :""
                                  }

                            
                    else:
                        print("Not a valid pig image!")
                        flag=False
                        current_time_utc = datetime.utcnow()
                        tz = pytz.timezone('Asia/Shanghai') 
                        current_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(tz)
                        current_time_str = current_time.strftime("%I:%M:%S %p")
                        display_image = color_image
                        _, jpeg = cv2.imencode('.jpg', display_image)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                        data = {
                                       "x":0,
                                       "y": 0,
                                        "z": 0,
                                        "Date": current_time_str,
                                        "Confience":"Null",
                                        "Flag":flag,
                                        "Video Status":"Video Stopped",
                                        "video file Path" : ""
                                  }
                              

    except Exception as e:
        print(e)


def cleanup():
    # Release camera resources if the pipeline is defined
    if pipeline:
        pipeline.stop()

    stop_recording()


@app.route('/')
def index():
    print('&&&&&&&&&&&&&&&&&&&&&&&&&', data)
    return jsonify(data)
    # return render_template('index2.html')


@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_recording', methods=['POST'])
def start_recording_route():
    start_recording(np.zeros((480, 640, 3), dtype=np.uint8))  # Pass a dummy image for start_recording
    return jsonify({"status": "Recording started"})


@app.route('/stop_recording', methods=['POST'])
def stop_recording_route():
    stop_recording()
    return jsonify({"status": "Recording stopped"})


if __name__ == '__main__':
    atexit.register(cleanup)
    app.run(debug=True, host='0.0.0.0', port=8081)
