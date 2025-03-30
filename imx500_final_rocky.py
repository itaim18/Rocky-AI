import argparse 
import sys
import time
import math
import numpy as np
import requests
import json
import subprocess
import os
import cv2

from inference_sdk import InferenceHTTPClient
from picamera2 import CompletedRequest, MappedArray, Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.devices.imx500.postprocess import COCODrawer
from picamera2.devices.imx500.postprocess_highernet import postprocess_higherhrnet

# ------------------ GLOBALS ------------------
last_boxes = None
last_scores = None
last_keypoints = None

WINDOW_SIZE_H_W = (480, 640)
bag_x = None  # punching bag x-axis

WRIST_IDX = [9, 10]
ANKLE_IDX = [15, 16]

AGGREGATOR = {
    "start_time": 0.0,
    "last_time": 0.0,
    "last_keypoints": None,
    "wrist_dist_sum": 0.0,
    "ankle_dist_sum": 0.0,
    "wrist_speed_sum": 0.0,
    "ankle_speed_sum": 0.0,
    "frame_count": 0,
    "no_movement_duration": 0.0
}

NO_MOVE_SPEED_THRESH = 5.0
NO_MOVE_TIME_LIMIT = 3.0

MAX_AVG_WRIST_SPEED = 3000.0
MAX_AVG_ANKLE_SPEED = 2000.0
MAX_WRIST_DIST = 30000.0
MAX_ANKLE_DIST = 10000.0

OPENAI_API_KEY =  "sk-$$$$$$$$$$$"
ELEVEN_API_KEY = "sk-$$$$$$$$$$$$"
ELEVEN_VOICE_ID = "##########"  # example voice ID

# ------------------ Roboflow Client ------------------
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="rf_$$$$$$$$$$$$$$$"  # Replace with your actual API key
)

# ------------------ DETECTION & MOVEMENT ------------------
def detect_bag_x(frame, save_path="frame.jpg"):
    cv2.imwrite(save_path, frame)
    try:
        result = CLIENT.infer(save_path, model_id="rocky-ai/2")
        predictions = result.get("predictions", [])
        if predictions:
            best = max(predictions, key=lambda p: p["confidence"])
            return int(best["x"])
    except Exception as e:
        print(f"[Roboflow ERROR]: {e}")

    return frame.shape[1] // 2
def scale_0_10(value, max_value):
    return min((value / max_value) * 10.0, 10.0) if value > 0 else 0.0

def euclidean_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def compute_movement(prev_kpts, curr_kpts, dt):
    if dt <= 0: return {"wrist_dist": 0, "ankle_dist": 0, "wrist_speed": 0, "ankle_speed": 0}

    wrist_dist = sum(euclidean_distance(prev_kpts[i][:2], curr_kpts[i][:2]) for i in WRIST_IDX)
    ankle_dist = sum(euclidean_distance(prev_kpts[i][:2], curr_kpts[i][:2]) for i in ANKLE_IDX)

    return {
        "wrist_dist": wrist_dist,
        "ankle_dist": ankle_dist,
        "wrist_speed": wrist_dist / dt,
        "ankle_speed": ankle_dist / dt
    }

def update_movement_aggregator(kpts, current_time):
    global AGGREGATOR
    if AGGREGATOR["last_keypoints"] is None:
        AGGREGATOR["last_keypoints"] = kpts
        AGGREGATOR["last_time"] = current_time
        return

    dt = current_time - AGGREGATOR["last_time"]
    movement = compute_movement(AGGREGATOR["last_keypoints"], kpts, dt)

    AGGREGATOR["wrist_dist_sum"] += movement["wrist_dist"]
    AGGREGATOR["ankle_dist_sum"] += movement["ankle_dist"]
    AGGREGATOR["wrist_speed_sum"] += movement["wrist_speed"]
    AGGREGATOR["ankle_speed_sum"] += movement["ankle_speed"]
    AGGREGATOR["frame_count"] += 1

    if movement["wrist_speed"] < NO_MOVE_SPEED_THRESH:
        AGGREGATOR["no_movement_duration"] += dt
    else:
        AGGREGATOR["no_movement_duration"] = 0.0

    if AGGREGATOR["no_movement_duration"] >= NO_MOVE_TIME_LIMIT:
        print("[INFO] No movement for 3s. Done.")
        finalize_metrics(force_finished=True)
        sys.exit(0)

    AGGREGATOR["last_time"] = current_time
    AGGREGATOR["last_keypoints"] = kpts

def reset_aggregator():
    now = time.time()
    AGGREGATOR.update({
        "start_time": now, "last_time": now, "last_keypoints": None,
        "wrist_dist_sum": 0.0, "ankle_dist_sum": 0.0,
        "wrist_speed_sum": 0.0, "ankle_speed_sum": 0.0,
        "frame_count": 0, "no_movement_duration": 0.0
    })

# ------------------ AI & AUDIO ------------------
def talk_like_rocky(metrics):
    prompt = f"""
I’m a boxer. My speed: {metrics['speed']}, power: {metrics['power']}, 
leg movement: {metrics['leg_movement']}, hand movement: {metrics['hand_movement']}. 
Finished: {metrics['finished']}
Give a Rocky-style note but be specific about legs work or hands or powe and speed. If not finished, max 12 words.
"""
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "system", "content": "You're Rocky Balboa."},
                         {"role": "user", "content": prompt}],
            "max_tokens": 150,
            "temperature": 0.7
        }
    ).json()
    note = response.get("choices", [{}])[0].get("message", {}).get("content", "Keep going champ!").strip()

    if not metrics["finished"] and len(note.split()) > 12:
        note = " ".join(note.split()[:12])

    print("💬 Rocky says:", note)

    audio = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}",
        headers={"xi-api-key": ELEVEN_API_KEY, "Content-Type": "application/json"},
        json={"text": note, "voice_settings": {"stability": 0.3, "similarity_boost": 0.75}}
    )

    if audio.status_code == 200:
        with open("output.mp3", "wb") as f:
            f.write(audio.content)
        subprocess.run(["cvlc", "--play-and-exit", "output.mp3"])
    else:
        print("[TTS ERROR]", audio.text)

def finalize_metrics(force_finished=False):
    fc = AGGREGATOR["frame_count"]
    if fc == 0:
        metrics = {"speed": 0, "power": 0, "leg_movement": 0, "hand_movement": 0, "finished": True}
    else:
        avg_ws = AGGREGATOR["wrist_speed_sum"] / fc
        avg_as = AGGREGATOR["ankle_speed_sum"] / fc
        total_wd = AGGREGATOR["wrist_dist_sum"]
        total_ad = AGGREGATOR["ankle_dist_sum"]

        metrics = {
            "speed": round(scale_0_10(avg_ws, MAX_AVG_WRIST_SPEED), 1),
            "power": round(scale_0_10((avg_ws + avg_as)/2, (MAX_AVG_WRIST_SPEED + MAX_AVG_ANKLE_SPEED)/2), 1),
            "hand_movement": round(scale_0_10(total_wd, MAX_WRIST_DIST), 1),
            "leg_movement": round(scale_0_10(total_ad, MAX_ANKLE_DIST), 1),
            "finished": force_finished
        }

    talk_like_rocky(metrics)
    reset_aggregator()

# ------------------ POSE CALLBACK ------------------
def ai_output_tensor_parse(metadata):
    global last_boxes, last_scores, last_keypoints
    np_outputs = imx500.get_outputs(metadata=metadata, add_batch=True)
    if np_outputs is not None:
        keypoints, scores, boxes = postprocess_higherhrnet(
            outputs=np_outputs,
            img_size=WINDOW_SIZE_H_W,
            img_w_pad=(0, 0),
            img_h_pad=(0, 0),
            detection_threshold=args.detection_threshold,
            network_postprocess=True
        )
        if scores is not None and len(scores) > 0:
            best = 0
            last_keypoints = np.reshape(keypoints[best], (17, 3))
            last_boxes = [np.array(boxes[best])]
            last_scores = np.array([scores[best]])
        else:
            last_keypoints = None
            last_boxes = None
            last_scores = None
    return last_boxes, last_scores, last_keypoints

def ai_output_tensor_draw(request, boxes, scores, keypoints, stream='main'):
    with MappedArray(request, stream) as m:
        if boxes:
            drawer.annotate_image(m.array, boxes, scores, np.zeros(scores.shape), keypoints[np.newaxis], args.detection_threshold, args.detection_threshold, request.get_metadata(), picam2, stream)

def picamera2_pre_callback(request):
    global bag_x
    boxes, scores, kpts = ai_output_tensor_parse(request.get_metadata())
    ai_output_tensor_draw(request, boxes, scores, kpts)

    if kpts is not None:
        current_time = time.time()
        update_movement_aggregator(kpts, current_time)

    with MappedArray(request, 'main') as m:
        frame = m.array
        if bag_x is None:
            bag_x = detect_bag_x(frame.copy()) or frame.shape[1] // 2
            print(f"[INFO] Punching bag x={bag_x}")

        # Check if either wrist crosses the bag line
        flash_red = any(abs(kpts[i][0] - bag_x) < 20 for i in WRIST_IDX) if kpts is not None else False
        color = (0, 0, 255) if flash_red else (255, 0, 0)
        cv2.line(frame, (bag_x, 0), (bag_x, frame.shape[0]), color, 3)

# ------------------ MAIN ------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk")
    parser.add_argument("--fps", type=int)
    parser.add_argument("--detection_threshold", type=float, default=0.3)
    parser.add_argument("--labels", type=str)
    parser.add_argument("--print_intrinsics", action="store_true")
    return parser.parse_args()

def get_drawer():
    cats = intrinsics.labels
    return COCODrawer([c for c in cats if c and c != "-"], imx500, needs_rescale_coords=False)

if __name__ == "__main__":
    args = get_args()

    imx500 = IMX500(args.model)
    intrinsics = imx500.network_intrinsics or NetworkIntrinsics()
    if not intrinsics.task:
        intrinsics.task = "pose estimation"
    elif intrinsics.task != "pose estimation":
        print("[ERROR] Not a pose model.")
        sys.exit(1)

    if args.labels:
        with open(args.labels) as f:
            intrinsics.labels = f.read().splitlines()
    if args.fps:
        intrinsics.inference_rate = args.fps
    if intrinsics.labels is None:
        with open("assets/coco_labels.txt") as f:
            intrinsics.labels = f.read().splitlines()
    intrinsics.update_with_defaults()
    if args.print_intrinsics:
        print(intrinsics)
        sys.exit(0)

    drawer = get_drawer()
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(controls={'FrameRate': intrinsics.inference_rate}, buffer_count=12)
    imx500.show_network_fw_progress_bar()
    picam2.start(config, show_preview=True)
    imx500.set_auto_aspect_ratio()
    picam2.pre_callback = picamera2_pre_callback

    reset_aggregator()
    print("[INFO] Boxing pose app started.")

    try:
        while True:
            time.sleep(0.2)
            if time.time() - AGGREGATOR["start_time"] >= 5.0:
                finalize_metrics(force_finished=False)
                print("[INFO] Starting new 5-second round...")

    except KeyboardInterrupt:
        print("[INFO] Ctrl+C pressed. Exiting...")
    finally:
        picam2.stop()
