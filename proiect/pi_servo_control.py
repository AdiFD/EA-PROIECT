"""
pi_servo_control.py

Script complet pentru Raspberry Pi 4 + Camera (picamera2) + PCA9685 (adafruit_servokit).
- Capturează cadre cu Picamera2 (dacă nu e disponibil, încearcă camera v4l2 OpenCV)
- Detectează obiecte verzi și galbene (spațiul HSV)
- Găsește contururi, clasifică forma (triangle, square/rectangle, circle, polygon)
- Debounce temporal: necesită detecție stabilă pe N cadre înainte de a comanda servo
- Controlează servomotorul prin adafruit_servokit (PCA9685)

Configurabil în secțiunea CONFIG.

Rulare:
    sudo python3 pi_servo_control.py

Note:
- Asigură-te că ai instalat `python3-picamera2` (sau `libcamera` compat) și `adafruit-circuitpython-servokit`.
- Alimentare servo: folosește sursă externă dacă servo consumă mult; conectează GND comun.
- Da Richard la tine ma refer.
"""

import time
import sys
import math

import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json

# încearcă picamera2, altfel folosește VideoCapture(0)
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# Controlează servo prin PCA9685 (Adafruit ServoKit)
try:
    from adafruit_servokit import ServoKit
    SERVOKIT_AVAILABLE = True
except Exception:
    SERVOKIT_AVAILABLE = False
    # vom continua, dar funcțiile de mișcare servo vor fi no-ops

# ---------------- CONFIG ----------------
CAMERA_SIZE = (640, 480)
MIN_AREA = 800                 # aria minima pentru a considera un contur valid
STABLE_FRAMES_REQUIRED = 3     # numar de cadre consecutive pentru debouncing
DEBUG_SHOW = True

# Servo mapping (schimbă canalul/corect angle în funcție de set-up)
SERVO_CHANNEL = 0              # canalul PCA9685 folosit
NEUTRAL_ANGLE = 90
MOVE_DELAY = 0.25              # secunde, așteaptă după comanda servo
RESET_AFTER_MOVE = True        # dacă True, revenim la NEUTRAL_ANGLE după RESET_DELAY
RESET_DELAY = 2.0             # secunde după care revenim la neutral (dacă RESET_AFTER_MOVE)

# Mapare color+shape -> angle (adaptează la nevoile tale)
# ACTION_MAP maps (color, shape) -> (servo_channel, angle)
# Adjust channels and angles to match your hardware setup.
# Example mapping for 4 compartments:
#  (green, square) -> compartment 0 (cube green)
#  (yellow, square) -> compartment 1 (cube yellow)
#  (green, triangle) -> compartment 2 (pyramid green)
#  (yellow, triangle) -> compartment 3 (pyramid yellow)
ACTION_MAP = {
    ('green', 'square'): (0, 180),
    ('yellow', 'square'): (1, 180),
    ('green', 'triangle'): (0, 0),
    ('yellow', 'triangle'): (1, 0),
}

# set of servo channels used by mapping (used for cleanup)
SERVOS_USED = set(ch for (ch, _) in ACTION_MAP.values())

# HSV ranges (valori de start - ajustează cu script de calibrare dacă ai nevoie)
GREEN_LOWER = np.array([40, 50, 50])
GREEN_UPPER = np.array([85, 255, 255])
YELLOW_LOWER = np.array([18, 100, 100])
YELLOW_UPPER = np.array([35, 255, 255])

# Try to load calibrated HSV values from file (created by calibrate_hsv.py)
HSV_STATE_FILE = 'hsv_values.json'
try:
    with open(HSV_STATE_FILE, 'r') as _f:
        _vals = json.load(_f)
        if 'green' in _vals:
            g = _vals['green']
            GREEN_LOWER = np.array([g.get('hmin', 40), g.get('smin', 50), g.get('vmin', 50)])
            GREEN_UPPER = np.array([g.get('hmax', 85), g.get('smax', 255), g.get('vmax', 255)])
        if 'yellow' in _vals:
            y = _vals['yellow']
            YELLOW_LOWER = np.array([y.get('hmin', 18), y.get('smin', 100), y.get('vmin', 100)])
            YELLOW_UPPER = np.array([y.get('hmax', 35), y.get('smax', 255), y.get('vmax', 255)])
        print(f"Loaded HSV values from {HSV_STATE_FILE}")
except FileNotFoundError:
    # no saved calibration, continue with defaults
    pass
except Exception as e:
    print('Warning: failed to load hsv values:', e)

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# ----------------------------------------


class ServoController:
    def __init__(self, default_channel=SERVO_CHANNEL, used_channels=None):
        """Initialize ServoKit. If not available, commands are simulated.
        - default_channel: kept for backward compatibility
        - used_channels: iterable of channels that should be returned to neutral on cleanup
        """
        self.default_channel = default_channel
        self.kit = None
        self.used_channels = set(used_channels) if used_channels is not None else {default_channel}
        if SERVOKIT_AVAILABLE:
            try:
                self.kit = ServoKit(channels=16)
                # set default neutral for used channels
                for ch in self.used_channels:
                    try:
                        self.kit.servo[ch].angle = NEUTRAL_ANGLE
                    except Exception:
                        pass
                time.sleep(0.1)
            except Exception as e:
                print("Warning: failed to initialize ServoKit:", e)
                self.kit = None
        else:
            print("Warning: adafruit_servokit not available. Servo commands will be no-ops.")

    def move(self, channel: int, angle: float):
        """Move a specific servo channel to angle (0..180)."""
        angle = max(0, min(180, float(angle)))
        if self.kit is None:
            print(f"[SIM] Move servo ch{channel} to angle {angle}")
        else:
            try:
                self.kit.servo[channel].angle = angle
            except Exception as e:
                print(f"Servo move failed (ch={channel}):", e)

    def move_multiple(self, mapping: dict):
        """mapping: {channel: angle, ...} - move multiple servos."""
        for ch, angle in mapping.items():
            self.move(ch, angle)

    def cleanup(self):
        if self.kit is not None:
            for ch in self.used_channels:
                try:
                    self.kit.servo[ch].angle = NEUTRAL_ANGLE
                except Exception:
                    pass


def recognize_colors(frame):
    """
    STEP 2: Recognize colors (green and yellow)
    Returns: dict with color masks {'green': mask, 'yellow': mask}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create color masks
    green_mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    
    # Clean masks with morphology
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, KERNEL, iterations=1)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    
    return {'green': green_mask, 'yellow': yellow_mask}


def recognize_shapes(color_masks):
    """
    STEP 3: Recognize shapes (triangle and square) from color masks
    Returns: list of detections with color, shape, area, center, contour
    """
    results = []
    
    for color_name, mask in color_masks.items():
        # Find contours in the color mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_AREA:
                continue
            
            # Detect shape
            perim = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perim, True)
            shape = None
            
            if len(approx) == 3:
                shape = 'triangle'
            elif len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h) if h != 0 else 0
                if 0.9 <= ar <= 1.1:  # Only accept squares, skip rectangles
                    shape = 'square'
            
            # Only process triangle and square shapes
            if shape is None:
                continue

            # Calculate center point
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

            results.append({
                'color': color_name,
                'shape': shape,
                'area': area,
                'center': (cx, cy),
                'cnt': cnt
            })
    
    return results


def choose_best_detection(detections):
    """Alege detectia cu aria maxima. Returneaza None daca nu sunt detectii."""
    if not detections:
        return None
    return max(detections, key=lambda d: d['area'])


def start_camera():
    """
    STEP 1: Start Pi Camera
    Returns: (camera_object, use_picam2_flag)
    """
    if PICAMERA2_AVAILABLE:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration({"main": {"size": CAMERA_SIZE}})
        picam2.configure(config)
        picam2.start()  # Camera activated here
        time.sleep(0.4)
        print('STEP 1: Pi Camera started (Picamera2)')
        return picam2, True
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('Error: no camera available')
            return None, False
        print('STEP 1: Camera started (OpenCV VideoCapture)')
        return cap, False


def action_servo(servo_controller, color, shape, channel, angle):
    """
    STEP 4: Action servo motors based on color and shape
    """
    print(f"STEP 4: Moving servo ch{channel} to angle {angle} for {color} {shape}")
    servo_controller.move(channel, angle)


def main():
    # STEP 1: Start Pi Camera
    camera, use_picam2 = start_camera()
    if camera is None:
        return

    # initialize servo controller and ensure all mapped channels are set to neutral
    servo = ServoController(used_channels=SERVOS_USED)
    # pending resets: channel -> reset_timestamp
    pending_resets = {}
    
    # MQTT setup
    try:
        mqtt_broker = "localhost"
        mqtt_topic = "cuburi/detectie"
        mqtt_client = mqtt.Client()
        mqtt_client.connect(mqtt_broker, 1883, 60)
        mqtt_client.loop_start()
        print(f"Connected to MQTT broker at {mqtt_broker}")
    except Exception as e:
        print("Warning: MQTT connect failed:", e)
        mqtt_client = None
    stable_frame_count = 0
    last_action = None

    try:
        while True:
            # Capture frame from camera
            if use_picam2:
                frame = camera.capture_array('main')
            else:
                ret, frame = camera.read()
                if not ret:
                    print('Failed to read frame')
                    break

            # STEP 2: Recognize colors
            color_masks = recognize_colors(frame)
            
            # STEP 3: Recognize shapes
            detections = recognize_shapes(color_masks)
            best = choose_best_detection(detections)

            action = None
            if best is not None:
                color = best['color']
                shape = best['shape']
                # mapare la actiune (daca exista)
                pair = ACTION_MAP.get((color, shape), None)
                if pair is not None:
                    channel, angle = pair
                    action = {'color': color, 'shape': shape, 'angle': angle, 'channel': channel, 'area': best['area'], 'center': best.get('center')}

            # debounce temporal: necesita STABLE_FRAMES_REQUIRED cadre consecutiv
            if action is not None and last_action is not None and action['color'] == last_action['color'] and action['shape'] == last_action['shape']:
                stable_frame_count += 1
            elif action is not None and last_action is None:
                stable_frame_count = 1
            else:
                stable_frame_count = 0

            if stable_frame_count >= STABLE_FRAMES_REQUIRED:
                # comanda servo daca actiunea e diferita de ultima executata
                if last_action is None or action.get('channel') != last_action.get('channel') or action['angle'] != last_action.get('angle'):
                    # STEP 4: Action servo motors
                    action_servo(servo, action['color'], action['shape'], action['channel'], action['angle'])
                    # schedule reset to neutral if configured
                    if RESET_AFTER_MOVE:
                        pending_resets[action['channel']] = time.time() + RESET_DELAY
                    time.sleep(MOVE_DELAY)
                    # publish MQTT (if available)
                    if mqtt_client is not None:
                        try:
                            payload = {
                                'color': action['color'],
                                'shape': action['shape'],
                                'angle': action['angle'],
                                'channel': action['channel'],
                                'area': action['area'],
                                'center': action.get('center'),
                                'timestamp': time.time()
                            }
                            mqtt_client.publish(mqtt_topic, json.dumps(payload))
                        except Exception as e:
                            print('MQTT publish failed:', e)
                    # optional: opreste semnalul (depinde de hardware)
                last_action = action
                stable_frame_count = 0

            # Draw detections for debug
            if DEBUG_SHOW:
                vis = frame.copy()
                for d in detections:
                    cv2.drawContours(vis, [d['cnt']], -1, (0, 255, 0) if d['color'] == 'green' else (0, 255, 255), 2)
                    cx, cy = d['center']
                    cv2.putText(vis, f"{d['color']} {d['shape']}", (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow('pi_servo_control', vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # handle pending resets (non-blocking)
            if RESET_AFTER_MOVE and pending_resets:
                now = time.time()
                to_reset = [ch for ch, t in pending_resets.items() if now >= t]
                for ch in to_reset:
                    try:
                        servo.move(ch, NEUTRAL_ANGLE)
                    except Exception as e:
                        print('Failed to reset servo ch', ch, e)
                    pending_resets.pop(ch, None)

    finally:
        # cleanup
        try:
            servo.cleanup()
        except Exception:
            pass
        try:
            if mqtt_client is not None:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
        except Exception:
            pass
        if use_picam2:
            try:
                camera.stop()
            except Exception:
                pass
        else:
            try:
                camera.release()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
