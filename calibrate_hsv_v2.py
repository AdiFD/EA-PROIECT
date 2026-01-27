"""
calibrate_hsv_v2.py

Versiune imbunatatita cu "Color Picker" in centrul ecranului.
Iti arata exact ce valori HSV vede camera in punctul central.
"""

import json
import time
import cv2
import numpy as np
import os

try:
    from picamera2 import Picamera2
    USE_PICAMERA2 = True
except ImportError:
    USE_PICAMERA2 = False

# Valori initiale mult mai largi pentru galben pentru a garanta detectia initiala
DEFAULTS = {
    'green': {'hmin': 40, 'hmax': 90, 'smin': 50, 'smax': 255, 'vmin': 50, 'vmax': 255},
    'yellow': {'hmin': 90, 'hmax': 110, 'smin': 15, 'smax': 100, 'vmin': 200, 'vmax': 255}
}

STATE_FILE = 'hsv_values.json'

if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, 'r') as f:
            saved = json.load(f)
            for c, v in saved.items():
                if c in DEFAULTS: DEFAULTS[c].update(v)
        print("Valori incarcate din fisier.")
    except: pass

def nothing(x): pass

def create_trackbars(w_name, init):
    cv2.createTrackbar('Hmin', w_name, init['hmin'], 179, nothing)
    cv2.createTrackbar('Hmax', w_name, init['hmax'], 179, nothing)
    cv2.createTrackbar('Smin', w_name, init['smin'], 255, nothing)
    cv2.createTrackbar('Smax', w_name, init['smax'], 255, nothing)
    cv2.createTrackbar('Vmin', w_name, init['vmin'], 255, nothing)
    cv2.createTrackbar('Vmax', w_name, init['vmax'], 255, nothing)

def read_trackbars(w_name):
    return {
        'hmin': cv2.getTrackbarPos('Hmin', w_name),
        'hmax': cv2.getTrackbarPos('Hmax', w_name),
        'smin': cv2.getTrackbarPos('Smin', w_name),
        'smax': cv2.getTrackbarPos('Smax', w_name),
        'vmin': cv2.getTrackbarPos('Vmin', w_name),
        'vmax': cv2.getTrackbarPos('Vmax', w_name)
    }

def update_trackbars(w_name, vals):
    for k, v in vals.items():
        name = k[0].upper() + k[1:]
        cv2.setTrackbarPos(name, w_name, v)

def main():
    global USE_PICAMERA2
    picam2 = None
 
    # Initializare camera
    if USE_PICAMERA2:
        try:
            picam2 = Picamera2()
            cfg = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XBGR8888"})
            picam2.configure(cfg)
            picam2.start()
            time.sleep(1.5)
            print("Picamera2 pornita.")
        except: USE_PICAMERA2 = False

    if not USE_PICAMERA2:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened(): return

    cur_color = 'yellow' # Pornim direct pe galben pentru test
    vals = DEFAULTS.copy()
    win = 'Calibrare HSV'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    create_trackbars(win, vals[cur_color])

    print("\nPOZITIONEAZA OBIECTUL IN CENTRUL ECRANULUI (CRUCE)")
    print("Taste: 'g'=Verde, 'y'=Galben, 's'=Salveaza, 'q'=Iesire\n")

    try:
        while True:
            if USE_PICAMERA2:
                frame = picam2.capture_array('main')[:,:,:3].copy()
            else:
                ret, frame = cap.read()
                if not ret: continue

            # --- PIPETA (COLOR PICKER) ---
            h, w = frame.shape[:2]
            cy, cx = h // 2, w // 2
            # Extragem HSV-ul pixelului central
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            pixel_hsv = hsv_frame[cy, cx]
            
            # Desenam o tinta in centru
            cv2.line(frame, (cx-20, cy), (cx+20, cy), (0, 0, 255), 2)
            cv2.line(frame, (cx, cy-20), (cx, cy+20), (0, 0, 255), 2)

            # Procesare masca curenta
            tv = read_trackbars(win)
            lower = np.array([tv['hmin'], tv['smin'], tv['vmin']])
            upper = np.array([tv['hmax'], tv['smax'], tv['vmax']])
            mask = cv2.inRange(hsv_frame, lower, upper)
            res = cv2.bitwise_and(frame, frame, mask=mask)

            # --- AFISARE INFO ---
            # Afisam ce vede camera exact in centru
            target_info = f"TINTA HSV: H={pixel_hsv[0]} S={pixel_hsv[1]} V={pixel_hsv[2]}"
            cv2.putText(frame, target_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            mode_info = f"MOD: {cur_color.upper()}"
            cv2.putText(res, mode_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Combinam cele doua ferestre
            combined = np.hstack((cv2.resize(frame, (480, 360)), cv2.resize(res, (480, 360))))
            cv2.imshow(win, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('g'):
                cur_color = 'green'
                update_trackbars(win, vals['green'])
            elif key == ord('y'):
                cur_color = 'yellow'
                update_trackbars(win, vals['yellow'])
            elif key == ord('s'):
                vals[cur_color] = read_trackbars(win)
                with open(STATE_FILE, 'w') as f: json.dump(vals, f, indent=4)
                print("Salvat!")

    except KeyboardInterrupt: pass
    finally:
        if picam2: picam2.stop()
        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()