import cv2
import numpy as np
from adafruit_servokit import ServoKit
import paho.mqtt.client as mqtt
import time

# Servo config (folosind PCA9685 cu 16 canale)
kit = ServoKit(channels=16)
servo_map = {'red':0, 'yellow':1, 'green':2}

# MQTT config
broker = "localhost"  # dacă brokerul Mosquitto rulează pe Raspberry Pi
topic = "cuburi/detectie"
client = mqtt.Client()
client.connect(broker, 1883, 60)

# Interval HSV culori
colors = {
    'red': ([0,100,100],[10,255,255]),
    'yellow': ([20,100,100],[30,255,255]),
    'green': ([40,50,50],[90,255,255])
}

cap = cv2.VideoCapture(0)

def move_servo(color):
    kit.servo[servo_map[color]].angle = 90
    time.sleep(0.5)
    kit.servo[servo_map[color]].angle = 0

counters = {'red':0, 'yellow':0, 'green':0}

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, (lower, upper) in colors.items():
        lower_np = np.array(lower)
        upper_np = np.array(upper)
        mask = cv2.inRange(hsv, lower_np, upper_np)
        
        if cv2.countNonZero(mask) > 500:  # prag detectie
            counters[color] += 1
            move_servo(color)

            # Trimitem mesaj MQTT
            payload = {
                "culoare": color,
                "numar": counters[color],
                "timestamp": time.time()
            }
            client.publish(topic, str(payload))
            print(payload)

    time.sleep(1)  # întârziere pentru stabilitate
