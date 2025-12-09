"""
robot_activare_la_cerere_simple.py

FARA MQTT. Doar logica mecanica pura.
Mod de functionare: START -> STOP -> ASTEPTARE
"""

import time
import cv2
import numpy as np
import json
import sys
import paho.mqtt.client as mqtt

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

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_DETECTION = "robot/detection"
MQTT_TOPIC_BELT = "robot/belt"
MQTT_TOPIC_SERVO = "robot/servo"
mqtt_client = None

# ---------------- CONFIGURARE ----------------

# --- Motor DC (Banda) ---
PIN_ENA = 25
PIN_IN1 = 23
PIN_IN2 = 24
MOTOR_SPEED = 100 

# --- Servo ---
SERVO_LEFT_CH = 0   # Cuburi
SERVO_RIGHT_CH = 1  # Piramide

# --- TIMPI DEPLASARE (CRITIC!) ---
TIME_TO_YELLOW = 2.5  
TIME_TO_GREEN = 4.5   

# --- Configurare Imagine ---
CAMERA_SIZE = (640, 480)
MIN_AREA = 1200
HSV_STATE_FILE = 'hsv_values.json'

GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([90, 255, 255])
YELLOW_LOWER = np.array([15, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

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
        GPIO.setup(PIN_ENA, GPIO.OUT)
        GPIO.setup(PIN_IN1, GPIO.OUT)
        GPIO.setup(PIN_IN2, GPIO.OUT)
        self.pwm = GPIO.PWM(PIN_ENA, 1000)
        self.pwm.start(0)
        self.stop() # Pornim cu ea oprita
    
    def start(self):
        GPIO.output(PIN_IN1, GPIO.HIGH)
        GPIO.output(PIN_IN2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(MOTOR_SPEED)
        
        # Publish MQTT event
        if mqtt_client and mqtt_client.is_connected():
            try:
                payload = {"status": "running", "timestamp": time.time()}
                mqtt_client.publish(MQTT_TOPIC_BELT, json.dumps(payload))
            except: pass
        
    def stop(self):
        GPIO.output(PIN_IN1, GPIO.LOW)
        GPIO.output(PIN_IN2, GPIO.LOW)
        self.pwm.ChangeDutyCycle(0)
        
        # Publish MQTT event
        if mqtt_client and mqtt_client.is_connected():
            try:
                payload = {"status": "stopped", "timestamp": time.time()}
                mqtt_client.publish(MQTT_TOPIC_BELT, json.dumps(payload))
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
            try:
                self.kit.servo[channel].angle = angle
            except: pass
        else:
            print(f"[SIM] Servo {channel} -> {angle}")
        
        # Publish MQTT event
        if mqtt_client and mqtt_client.is_connected():
            try:
                payload = {"channel": channel, "angle": angle, "timestamp": time.time()}
                mqtt_client.publish(MQTT_TOPIC_SERVO, json.dumps(payload))
            except: pass

    def get_last_angle(self, channel):
        return self.current_angles.get(channel, 0)

# ---------------- LOGICA ----------------

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
            if area > MIN_AREA and area > max_area:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                corners = len(approx)
                
                shape = None
                if corners == 3: shape = "triangle"
                elif corners == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    ar = float(w)/h
                    if 0.9 <= ar <= 1.1: shape = "square"
                
                if shape:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        best_obj = (color, shape, cX, area)
                        max_area = area
    return best_obj

def execute_on_demand_sequence(belt, servos, obj_data):
    color, shape, _, area = obj_data
    print(f"\n[NOU OBIECT] {color.upper()} {shape.upper()}")

    # 1. Selectie Servo
    target_servo = None
    if shape == "square":
        target_servo = SERVO_LEFT_CH
    elif shape == "triangle":
        target_servo = SERVO_RIGHT_CH
    else:
        return

    # 2. Selectie Pozitii si Timp (Pendul)
    start_pos = 0
    end_pos = 0
    wait_time = 0
    
    if color == 'yellow':
        start_pos = 0
        end_pos = 180
        wait_time = TIME_TO_YELLOW
    elif color == 'green':
        start_pos = 180
        end_pos = 0
        wait_time = TIME_TO_GREEN

    # --- EXECUTIE ---

    # A. Pregatire Servo (BANDA E OPRITA)
    current_angle = servos.get_last_angle(target_servo)
    if current_angle != start_pos:
        print(f"   > Pozitionez servo la start ({start_pos})...")
        servos.move(target_servo, start_pos)
        time.sleep(0.3) 

    # B. START BANDA (Acum pleaca obiectul)
    print(f"   > START BANDA! (Merge {wait_time}s)")
    belt.start()
    
    # C. Transport
    t_start = time.time()
    while time.time() - t_start < wait_time:
        time.sleep(0.1)

    # D. STOP BANDA (La destinatie)
    print("   > STOP BANDA. (Destinatie atinsa)")
    belt.stop()
    time.sleep(0.5)

    # E. IMPINGE SERVO
    print(f"   > SERVO IMPINGE la {end_pos}")
    servos.move(target_servo, end_pos)
    time.sleep(0.8)

    # F. GATA (Banda ramane oprita)
    print("   > Ciclu complet. Astept urmatorul obiect.")

def mqtt_init():
    """Initialize MQTT client and connect to broker"""
    global mqtt_client
    try:
        mqtt_client = mqtt.Client()
        mqtt_client.on_connect = on_mqtt_connect
        mqtt_client.on_disconnect = on_mqtt_disconnect
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        mqtt_client.loop_start()
        print(f"[MQTT] Connecting to {MQTT_BROKER}:{MQTT_PORT}...")
        time.sleep(1)
    except Exception as e:
        print(f"[MQTT] Failed to initialize: {e}")
        mqtt_client = None

def on_mqtt_connect(client, userdata, flags, rc):
    """MQTT connection callback"""
    if rc == 0:
        print(f"[MQTT] Connected successfully (code {rc})")
    else:
        print(f"[MQTT] Connection failed (code {rc})")

def on_mqtt_disconnect(client, userdata, rc):
    """MQTT disconnection callback"""
    if rc != 0:
        print(f"[MQTT] Unexpected disconnection (code {rc})")

def main():
    belt = ConveyorBelt()
    servos = ServoManager()
    
    # Initialize MQTT
    mqtt_init()

    picam2 = None
    cap = None
    if PICAMERA2_AVAILABLE:
        print("Pornire Picamera2...")
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": CAMERA_SIZE, "format": "XBGR8888"},
            controls={
                # Fortam 30 FPS:
                # 1 secunda = 1,000,000 microsecunde
                # 1,000,000 / 30 = 33333
                "FrameDurationLimits": (33333, 33333)
            })
        picam2.configure(config)
        picam2.start()
        stream_config = picam2.stream_configuration("main")
        print(f"Camera Configured: {stream_config['size']} format={stream_config['format']}")
        time.sleep(2)
    else:
        print("Pornire Webcam...")
        cap = cv2.VideoCapture(0)

    print("\n--- ROBOT ACTIVARE LA CERERE (FARA MQTT) ---")
    print("Pune obiectul -> Robotul porneste -> Sorteaza -> Se opreste.")
    
    belt.stop()
    servos.move(SERVO_LEFT_CH, 0)
    servos.move(SERVO_RIGHT_CH, 0)

    last_detection_time = 0
    COOLDOWN = 2.0 

    try:
        while True:
            frame = None
            if PICAMERA2_AVAILABLE:
                frame = picam2.capture_array('main').copy()
            else:
                ret, frame = cap.read()
                if not ret: break

            cv2.imshow("Robot Sortare", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if time.time() - last_detection_time < COOLDOWN:
                continue

            obj = detect_object(frame)
            if obj:
                # Publish detection event to MQTT
                if mqtt_client and mqtt_client.is_connected():
                    try:
                        color, shape, center, area = obj
                        payload = {
                            "color": color,
                            "shape": shape,
                            "center_x": center[0],
                            "center_y": center[1],
                            "area": area,
                            "timestamp": time.time()
                        }
                        mqtt_client.publish(MQTT_TOPIC_DETECTION, json.dumps(payload))
                    except: pass
                
                execute_on_demand_sequence(belt, servos, obj)
                last_detection_time = time.time()

    except KeyboardInterrupt:
        print("\nOprire...")
    finally:
        belt.stop()
        GPIO.cleanup()
        if picam2: picam2.stop()
        if cap: cap.release()
        
        # Cleanup MQTT
        if mqtt_client:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
                print("[MQTT] Disconnected")
            except: pass
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()