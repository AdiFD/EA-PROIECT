"""
robot_video_control_stepper.py

VERSIUNEA FINALA:
- STEPPER + VIDEO + MQTT
- COORDONATE: Bounding Rect (ca la codul mic: x, y)
- VISUAL FEEDBACK: Deseneaza pe ecran
"""

import time
import cv2
import numpy as np
import json
import sys
import threading
import paho.mqtt.client as mqtt
from flask import Flask, Response

# Biblioteci Hardware
import RPi.GPIO as GPIO
try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except Exception:
    SERVOKIT_AVAILABLE = False

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

# --- CONFIGURARE STREAMING VIDEO ---
outputFrame = None
lock = threading.Lock()
app = Flask(__name__)

# --- CONFIGURARE MQTT ---
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_DETECTION = "robot/detection"
MQTT_TOPIC_BELT = "robot/belt"
MQTT_TOPIC_SERVO = "robot/servo"
MQTT_TOPIC_CONTROL = "robot/control"

mqtt_client = None
SYSTEM_ACTIVE = True 

# --- CONFIGURARE STEPPER ---
IN1 = 23
IN2 = 17 
IN3 = 24
IN4 = 27
ENA = 25
ENB = 5

STEP_WAIT = 0.0025

STEP_SEQUENCE = [
    [1, 0, 1, 0],
    [0, 1, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 0, 1]
]

# --- Configurare Servo ---
SERVO_LEFT_CH = 0
SERVO_RIGHT_CH = 1

# --- TIMPI DEPLASARE ---
TIME_TO_YELLOW = 1.5  
TIME_TO_GREEN = 1.5

# --- Configurare Imagine ---
CAMERA_SIZE = (640, 480)
MIN_AREA = 1200
HSV_STATE_FILE = 'hsv_values.json'


# Incarcare calibrare
try:
    with open(HSV_STATE_FILE, 'r') as f:
        vals = json.load(f)
        if 'green' in vals:
            g = vals['green']
            GREEN_LOWER = np.array([g['hmin'], g['smin'], g['vmin']])
            GREEN_UPPER = np.array([g['hmax'], g['smax'], g['vmax']])
        if 'yellow' in vals:
            y = vals['yellow']
            YELLOW_LOWER = np.array([y['hmin'], y['smin'], y['vmin']])
            YELLOW_UPPER = np.array([y['hmax'], y['smax'], y['vmax']])
except: pass

# ---------------- CLASE ----------------

class ConveyorBelt:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        self.pins = [IN1, IN2, IN3, IN4, ENA, ENB]
        for pin in self.pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, 0)
            
        self.is_running = False
        self.thread = None
    
    def _set_step(self, w1, w2, w3, w4):
        GPIO.output(IN1, w1)
        GPIO.output(IN2, w2)
        GPIO.output(IN3, w3)
        GPIO.output(IN4, w4)

    def _run_loop(self):
        GPIO.output(ENA, GPIO.HIGH)
        GPIO.output(ENB, GPIO.HIGH)
        while self.is_running:
            for seq in reversed(STEP_SEQUENCE):
                if not self.is_running: break
                self._set_step(seq[0], seq[1], seq[2], seq[3])
                time.sleep(STEP_WAIT)
                
        self._set_step(0, 0, 0, 0)
        GPIO.output(ENA, GPIO.LOW)
        GPIO.output(ENB, GPIO.LOW)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._run_loop)
            self.thread.start()
            self._publish("running")
        
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        self._set_step(0, 0, 0, 0)
        GPIO.output(ENA, GPIO.LOW)
        GPIO.output(ENB, GPIO.LOW)
        self._publish("stopped")
        
    def _publish(self, status):
        if mqtt_client and mqtt_client.is_connected():
            try:
                mqtt_client.publish(MQTT_TOPIC_BELT, json.dumps({"status": status, "timestamp": time.time()}))
            except: pass

class ServoManager:
    def __init__(self):
        self.kit = None
        self.current_angles = {SERVO_LEFT_CH: 0, SERVO_RIGHT_CH: 0}
        if SERVOKIT_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16)
                self.move(SERVO_LEFT_CH, 0)
                self.move(SERVO_RIGHT_CH, 0)
            except: pass

    def move(self, channel, angle):
        self.current_angles[channel] = angle
        if self.kit:
            try: self.kit.servo[channel].angle = angle
            except: pass
        else: print(f"[SIM] Servo {channel} -> {angle}")
        
        if mqtt_client and mqtt_client.is_connected():
            try:
                mqtt_client.publish(MQTT_TOPIC_SERVO, json.dumps({"channel": channel, "angle": angle, "timestamp": time.time()}))
            except: pass

    def get_last_angle(self, channel):
        return self.current_angles.get(channel, 0)

# ---------------- LOGICA (MODIFICATA AICI) ----------------

def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = {'green': cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER),
             'yellow': cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)}
    best_obj = None
    max_area = 0

    for color, mask in masks.items():
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > MIN_AREA:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                corners = len(approx)
                
                x, y, w, h = cv2.boundingRect(approx)
                
                shape = None
                
                # Detectie forma
                if corners == 3: 
                    shape = "triangle"
                elif corners == 4:
                    aspect = float(w)/h
                    if 0.90 <= aspect <= 1.15: 
                        shape = "square"
                
                if shape:
                    # --- FILTRU EXCLUDERE (MODIFICAREA E AICI) ---
                    # Daca detectez Green Triangle SAU Yellow Square, le ignor complet
                    if (color == 'green' and shape == 'triangle') or \
                       (color == 'yellow' and shape == 'square'):
                        continue 
                    # ---------------------------------------------

                    # 1. DESENARE PE FRAME (Doar daca nu au fost excluse mai sus)
                    cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                    
                    cv2.putText(frame, f"{color} {shape}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # 2. Logica 'Regele Dealului'
                    if area > max_area:
                        best_obj = (color, shape, x, area)
                        max_area = area
    
    return best_obj

def execute_on_demand_sequence(belt, servos, obj_data):
    if not SYSTEM_ACTIVE: return

    color, shape, _, area = obj_data
    print(f"\n[ACTIONEZ] {color.upper()} {shape.upper()}")

    target_servo = None
    if shape == "square": target_servo = SERVO_LEFT_CH
    elif shape == "triangle": target_servo = SERVO_RIGHT_CH
    else: return

    start_pos = 180   ; end_pos = 180; wait_time = 0
    if color == 'yellow':
        start_pos = 180; end_pos = 0; wait_time = TIME_TO_YELLOW
    elif color == 'green':
        start_pos = 0; end_pos = 180; wait_time = TIME_TO_GREEN

    if servos.get_last_angle(target_servo) != start_pos:
        servos.move(target_servo, start_pos)
        time.sleep(0.3) 

    belt.start()
    t_start = time.time()
    while time.time() - t_start < wait_time:
        if not SYSTEM_ACTIVE: 
            belt.stop()
            return 
        time.sleep(0.1)

    belt.stop()
    time.sleep(0.5)
    servos.move(target_servo, end_pos)
    time.sleep(0.8)
    print("   > Ciclu gata.")

# --- MQTT SETUP ---
def on_message(client, userdata, msg):
    global SYSTEM_ACTIVE
    try:
        command = msg.payload.decode().upper()
        if command == "START":
            SYSTEM_ACTIVE = True
            print("[CONTROL] SISTEM PORNIT")
        elif command == "STOP":
            SYSTEM_ACTIVE = False
            print("[CONTROL] SISTEM OPRIT (PAUZA)")
    except: pass

def mqtt_init():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = lambda c, u, f, rc: c.subscribe(MQTT_TOPIC_CONTROL)
        mqtt_client.on_message = on_message
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"MQTT Error: {e}")

# --- FLASK VIDEO SERVER ---
@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def run_flask():
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

# --- MAIN ---
def main():
    global outputFrame
    belt = ConveyorBelt()
    servos = ServoManager()
    mqtt_init()

    t = threading.Thread(target=run_flask)
    t.daemon = True
    t.start()

    picam2 = None
    cap = None
    if PICAMERA2_AVAILABLE:
        print("Picamera2 (Streaming Enabled)...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "XBGR8888"},
            controls={"FrameDurationLimits": (33333, 33333)})
        picam2.configure(config)
        picam2.start()
        time.sleep(10)
    else:
        cap = cv2.VideoCapture(0)

    print("\n--- ROBOT + VIDEO + CONTROL ---")
    print("Video disponibil la: http://<IP_RASPBERRY>:5000/video_feed")
    
    belt.stop()
    servos.move(SERVO_LEFT_CH, 0)
    servos.move(SERVO_RIGHT_CH, 0)

    last_detection_time = 10
    COOLDOWN = 10.0 

    try:
        while True:
            frame = None
            if PICAMERA2_AVAILABLE:
                frame = picam2.capture_array('main').copy()
            else:
                ret, frame = cap.read()
                if not ret: break

            obj = detect_object(frame) 

            with lock:
                outputFrame = frame.copy()

            if not SYSTEM_ACTIVE:
                time.sleep(0.05)
                continue

            if time.time() - last_detection_time < COOLDOWN:
                continue

            if obj:
                if mqtt_client:
                    c, s, x_coord, a = obj
                    mqtt_client.publish(MQTT_TOPIC_DETECTION, json.dumps({
                        "color": c, "shape": s, "center_x": x_coord, "timestamp": time.time()
                    }))
                
                execute_on_demand_sequence(belt, servos, obj)
                last_detection_time = time.time()

    except KeyboardInterrupt:
        print("Oprire...")
    finally:
        belt.stop()
        GPIO.cleanup()
        if picam2: picam2.stop()
        if mqtt_client: mqtt_client.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()