"""
calibrate_hsv.py

Script pentru calibrare HSV folosind trackbars OpenCV.
- Ruleaza pe Raspberry Pi cu Picamera2 sau pe orice cameră accesibilă la /dev/video0
- Permite ajustarea H/S/V min/max și salvarea valorilor pentru "green" si "yellow"

Rulare:
    python3 calibrate_hsv.py

Taste utile:
 - g: selecteaza culoarea 'green'
 - y: selecteaza culoarea 'yellow'
 - s: salveaza valorile curente intr-un fisier 'hsv_values.json'
 - q: iesire
"""

import json
import time
import cv2
import numpy as np

# incearca picamera2, fallback la VideoCapture
try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except Exception:
    USE_PICAMERA2 = False

# default values (plecare)
DEFAULTS = {
    'green': {'hmin': 40, 'hmax': 85, 'smin': 50, 'smax': 255, 'vmin': 50, 'vmax': 255},
    'yellow': {'hmin': 18, 'hmax': 35, 'smin': 100, 'smax': 255, 'vmin': 100, 'vmax': 255}
}

STATE_FILE = 'hsv_values.json'

# utilitati trackbar
def nothing(x):
    pass

def create_trackbars(window_name, initial):
    cv2.createTrackbar('Hmin', window_name, initial['hmin'], 179, nothing)
    cv2.createTrackbar('Hmax', window_name, initial['hmax'], 179, nothing)
    cv2.createTrackbar('Smin', window_name, initial['smin'], 255, nothing)
    cv2.createTrackbar('Smax', window_name, initial['smax'], 255, nothing)
    cv2.createTrackbar('Vmin', window_name, initial['vmin'], 255, nothing)
    cv2.createTrackbar('Vmax', window_name, initial['vmax'], 255, nothing)

def read_trackbars(window_name):
    hmin = cv2.getTrackbarPos('Hmin', window_name)
    hmax = cv2.getTrackbarPos('Hmax', window_name)
    smin = cv2.getTrackbarPos('Smin', window_name)
    smax = cv2.getTrackbarPos('Smax', window_name)
    vmin = cv2.getTrackbarPos('Vmin', window_name)
    vmax = cv2.getTrackbarPos('Vmax', window_name)
    return {'hmin': hmin, 'hmax': hmax, 'smin': smin, 'smax': smax, 'vmin': vmin, 'vmax': vmax}


def main():
    if USE_PICAMERA2:
        picam2 = Picamera2()
        picam2_config = picam2.create_preview_configuration({'main': {'size': (640, 480)}})
        picam2.configure(picam2_config)
        picam2.start()
        time.sleep(0.3)
        cap = None
        print('Using Picamera2 for calibration')
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print('No camera found')
            return
        picam2 = None
        print('Using OpenCV VideoCapture for calibration')

    current_color = 'green'
    values = DEFAULTS.copy()

    window = 'HSV Calibrator'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    create_trackbars(window, values[current_color])

    print("Controls: g=green, y=yellow, s=save, q=quit")

    try:
        while True:
            if USE_PICAMERA2:
                frame = picam2.capture_array('main')
            else:
                ret, frame = cap.read()
                if not ret:
                    print('Failed to grab frame')
                    break

            # read trackbars and build mask
            tv = read_trackbars(window)
            lower = np.array([tv['hmin'], tv['smin'], tv['vmin']])
            upper = np.array([tv['hmax'], tv['smax'], tv['vmax']])
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # show info overlay
            info = f"Color={current_color} H:[{tv['hmin']},{tv['hmax']}] S:[{tv['smin']},{tv['smax']}] V:[{tv['vmin']},{tv['vmax']}]"
            cv2.putText(res, info, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

            combined = np.hstack((cv2.resize(frame, (320,240)), cv2.resize(res, (320,240))))
            cv2.imshow(window, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('g'):
                current_color = 'green'
                # reinit trackbars to current color values
                vals = values[current_color]
                cv2.setTrackbarPos('Hmin', window, vals['hmin'])
                cv2.setTrackbarPos('Hmax', window, vals['hmax'])
                cv2.setTrackbarPos('Smin', window, vals['smin'])
                cv2.setTrackbarPos('Smax', window, vals['smax'])
                cv2.setTrackbarPos('Vmin', window, vals['vmin'])
                cv2.setTrackbarPos('Vmax', window, vals['vmax'])
                print('Switched to GREEN')
            elif key == ord('y'):
                current_color = 'yellow'
                vals = values[current_color]
                cv2.setTrackbarPos('Hmin', window, vals['hmin'])
                cv2.setTrackbarPos('Hmax', window, vals['hmax'])
                cv2.setTrackbarPos('Smin', window, vals['smin'])
                cv2.setTrackbarPos('Smax', window, vals['smax'])
                cv2.setTrackbarPos('Vmin', window, vals['vmin'])
                cv2.setTrackbarPos('Vmax', window, vals['vmax'])
                print('Switched to YELLOW')
            elif key == ord('s'):
                # save current trackbar values into file for the current color
                vals = read_trackbars(window)
                values[current_color] = vals
                with open(STATE_FILE, 'w') as f:
                    json.dump(values, f, indent=2)
                print(f"Saved {current_color} values to {STATE_FILE}:", vals)

    finally:
        if USE_PICAMERA2 and picam2 is not None:
            try:
                picam2.stop()
            except Exception:
                pass
        if not USE_PICAMERA2 and cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
