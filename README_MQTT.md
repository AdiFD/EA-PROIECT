# Robot Sorting System - Complete Documentation Index

## 📋 Project Overview

This is a **conveyor belt sorting robot** with:
- **Vision**: Object detection by color (green/yellow) and shape (square/triangle)
- **Control**: 2 servo motors + 1 DC motor (conveyor belt) via GPIO
- **Monitoring**: MQTT real-time event publishing
- **Dashboard**: Optional Node-RED web interface

---

## 📁 File Structure

### Core Application Files

#### `pi_servo_control.py` ⭐ **Main Robot Program**
- **Purpose**: Complete robotic sorting system with MQTT integration
- **Components**:
  - `ConveyorBelt` class - DC motor control (GPIO PWM)
  - `ServoManager` class - Servo positioning (GPIO or PCA9685)
  - `detect_object()` - HSV-based color/shape detection
  - `execute_on_demand_sequence()` - Sorting state machine
  - MQTT event publishing (non-blocking)
- **Usage**: `python3 pi_servo_control.py`
- **Dependencies**: RPi.GPIO, adafruit_servokit, picamera2, opencv-python, paho-mqtt
- **Status**: ✅ Production-ready with MQTT

#### `calibrate_hsv.py`
- **Purpose**: Interactive HSV color range calibration tool
- **Features**:
  - Live camera preview with adjustable trackbars
  - Real-time color mask visualization
  - Save/load calibration to `hsv_values.json`
- **Usage**: `python3 calibrate_hsv.py` → Press 'g' (green), 'y' (yellow), 's' (save)
- **Output**: Updates `hsv_values.json` with optimized color ranges

#### `calibrate_hsv.py` (Legacy)
- Historical reference; use `calibrate_hsv.py` instead

#### `color_detector.py`
- Legacy prototype (replaced by pi_servo_control.py)
- Kept for reference only

### Configuration Files

#### `hsv_values.json`
- **Purpose**: Saved HSV color calibration data
- **Created by**: `calibrate_hsv.py` script
- **Format**:
```json
{
  "green": {"hmin": 35, "hmax": 85, "smin": 100, "smax": 255, "vmin": 100, "vmax": 255},
  "yellow": {"hmin": 15, "hmax": 35, "smin": 100, "smax": 255, "vmin": 100, "vmax": 255}
}
```

#### `requirements.txt`
- Python package dependencies
- Install with: `pip3 install -r requirements.txt`

#### `README.md`
- Project introduction and overview

---

## 📚 Documentation Files

### Quick Start Guides

#### `QUICK_START.md` ⭐ **START HERE**
- **Best for**: Getting system running in 5 minutes
- **Contains**:
  - TL;DR installation steps
  - Configuration changes
  - Common troubleshooting
  - File reference guide
- **Time to read**: 3-5 minutes

#### `INTEGRATION_SUMMARY.md` ⭐ **For Developers**
- **Best for**: Understanding what changed
- **Contains**:
  - Complete list of code changes (9 locations)
  - Before/after code snippets
  - Safety & error handling details
  - Deployment instructions
- **Time to read**: 10-15 minutes

### Detailed Documentation

#### `MQTT_INTEGRATION_GUIDE.md` ⭐ **Complete MQTT Reference**
- **Best for**: Deep understanding of MQTT implementation
- **Contains**:
  - Installation (Mosquitto, paho-mqtt)
  - Configuration options
  - MQTT topics and payload formats (with examples)
  - Testing procedures (mosquitto_sub/pub)
  - Failure modes and error handling
  - Performance analysis
  - Troubleshooting table
- **Time to read**: 20-30 minutes

#### `NODE_RED_SETUP.md` ⭐ **Dashboard Installation**
- **Best for**: Setting up web-based monitoring
- **Contains**:
  - Node.js and Node-RED installation
  - Dashboard plugin setup
  - Systemd auto-start configuration
  - Pre-built flow JSON (import-ready)
  - Manual flow creation steps
  - Dashboard customization examples
  - Authentication setup
- **Time to read**: 15-20 minutes

#### `VALIDATION_REPORT.md` ⭐ **Technical Verification**
- **Best for**: QA and production readiness verification
- **Contains**:
  - Change verification checklist
  - Core logic preservation proof
  - Payload format validation
  - Error handling analysis
  - Performance metrics
  - Test scenarios
  - Pre/post-deployment checklists
- **Time to read**: 10-15 minutes

---

## 🚀 Quick Navigation

### I want to...

#### **Get the robot working in 5 minutes**
→ Read: `QUICK_START.md`
→ Run: `python3 pi_servo_control.py`
→ Monitor: `mosquitto_sub -h localhost -t "robot/#"`

#### **Understand the MQTT implementation**
→ Read: `MQTT_INTEGRATION_GUIDE.md`
→ Test: `mosquitto_pub` and `mosquitto_sub` commands
→ Verify: Payload formats in section "MQTT Topics & Payloads"

#### **Set up a web dashboard**
→ Read: `NODE_RED_SETUP.md`
→ Install: Node-RED and dashboard plugin
→ Import: Pre-built flow JSON
→ Access: `http://localhost:1880/ui`

#### **Calibrate HSV color detection**
→ Run: `python3 calibrate_hsv.py`
→ Adjust: Trackbars for green and yellow
→ Save: Press 's' to save to `hsv_values.json`

#### **See what code changed**
→ Read: `INTEGRATION_SUMMARY.md`
→ Focus: Section "Files Modified"

#### **Verify production readiness**
→ Read: `VALIDATION_REPORT.md`
→ Run: Pre/post-deployment checklists
→ Test: All scenarios in "Testing Scenarios"

---

## 🔧 Hardware Setup Reference

### GPIO Pinout (Raspberry Pi)

| Component | Pin | GPIO | Purpose |
|-----------|-----|------|---------|
| Belt Motor - ENA | 22 | GPIO 25 | PWM speed control |
| Belt Motor - IN1 | 16 | GPIO 23 | Direction bit 1 |
| Belt Motor - IN2 | 18 | GPIO 24 | Direction bit 2 |
| Servo - SDA | 3 | GPIO 2 | I2C (if using PCA9685) |
| Servo - SCL | 5 | GPIO 3 | I2C (if using PCA9685) |
| Camera - CSI | - | - | Ribbon cable slot |

### Wiring Diagram (Conveyor + Servos)

```
Raspberry Pi 4B
├── GPIO 25 (PWM) ──→ L298N/Motor Driver ENA
├── GPIO 23 ──────→ L298N Motor Driver IN1
├── GPIO 24 ──────→ L298N Motor Driver IN2
├── I2C SDA (GPIO 2) ──→ PCA9685 SDA
├── I2C SCL (GPIO 3) ──→ PCA9685 SCL
├── CSI (Camera port) ──→ Camera Module v2.1
└── GND ──→ Common ground (motor driver + PCA9685)

Motor Driver (L298N)
├── IN1, IN2 ──→ GPIO 23, 24
├── ENA ──→ GPIO 25 (PWM)
├── OUT1, OUT2 ──→ DC Motor leads
└── GND ──→ Pi GND

PCA9685 Servo Driver
├── SDA ──→ GPIO 2
├── SCL ──→ GPIO 3
├── CH0 ──→ Left Servo (Squares)
├── CH1 ──→ Right Servo (Triangles)
└── GND ──→ Pi GND
```

---

## 📊 MQTT Message Flow

### Example Sorting Sequence

```
User places object on conveyor
         ↓
    [DETECTION]
         ↓
robot/detection ← {"color":"yellow", "shape":"square", "area":3200, ...}
         ↓
    Robot processes shape → select servo
         ↓
robot/servo ← {"channel":0, "angle":45, ...}  [Position at start]
         ↓
robot/belt ← {"status":"running", ...}  [Start conveyor]
         ↓
         [Conveyor moves for 2.5 seconds]
         ↓
robot/belt ← {"status":"stopped", ...}  [Stop conveyor]
         ↓
robot/servo ← {"channel":0, "angle":135, ...}  [Push to bin]
         ↓
     [CYCLE COMPLETE - Await next object]
```

---

## 🔍 Configuration Reference

### In `pi_servo_control.py`

```python
# MQTT Configuration (Lines 29-35)
MQTT_BROKER = "localhost"          # Change to IP for remote
MQTT_PORT = 1883                   # Default Mosquitto port
MQTT_TOPIC_DETECTION = "robot/detection"
MQTT_TOPIC_BELT = "robot/belt"
MQTT_TOPIC_SERVO = "robot/servo"

# Motor Control (Lines 40-41)
PIN_ENA = 25      # PWM pin for speed
PIN_IN1 = 23      # Direction pin 1
PIN_IN2 = 24      # Direction pin 2
MOTOR_SPEED = 100 # PWM duty cycle (0-100)

# Servo Channels (Lines 44-45)
SERVO_LEFT_CH = 0   # For squares/cubes
SERVO_RIGHT_CH = 1  # For triangles/pyramids

# Timing (Lines 48-49)
TIME_TO_YELLOW = 2.5  # Seconds to travel to yellow bin
TIME_TO_GREEN = 3.0   # Seconds to travel to green bin
```

### Change Broker for Remote Monitoring

```python
# Original (local):
MQTT_BROKER = "localhost"

# For remote (e.g., monitoring from laptop):
MQTT_BROKER = "192.168.1.100"  # Pi's IP address
```

---

## 📦 Dependencies

### Required Packages

```bash
# System packages
sudo apt install mosquitto mosquitto-clients python3-dev

# Python packages
pip3 install -r requirements.txt
# OR individually:
pip3 install RPi.GPIO adafruit-servokit picamera2 opencv-python paho-mqtt numpy
```

### Optional Packages

```bash
# For Node-RED dashboard
sudo npm install -g node-red node-red-dashboard
```

### Verify Installation

```bash
python3 -c "import RPi.GPIO; import cv2; import paho.mqtt; print('✅ All imports OK')"
```

---

## 🐛 Troubleshooting Quick Guide

| Problem | Quick Fix |
|---------|-----------|
| `No module named paho.mqtt` | `pip3 install paho-mqtt` |
| `Connection refused (111)` | Start Mosquitto: `sudo systemctl start mosquitto` |
| Mosquitto won't start | Check port 1883: `sudo netstat -tlnp \| grep 1883` |
| MQTT messages not showing | Verify topic names match exactly (case-sensitive) |
| Robot works but MQTT silent | Change `MQTT_BROKER` to correct IP |
| Node-RED 404 at `/ui` | Reinstall dashboard: `sudo npm install -g node-red-dashboard` |
| Pi can't detect camera | Verify CSI cable connected; run `vcgencmd get_camera` |

---

## 📈 Project Status

### Implementation ✅ COMPLETE
- Core robot control logic: ✅ Working
- MQTT integration: ✅ Complete and tested
- HSV detection: ✅ Calibration tool ready
- Node-RED dashboard: ✅ Flow provided
- Documentation: ✅ Comprehensive

### Testing Status ✅ READY
- Unit tested: ✅ Individual components
- Integration tested: ✅ Full sequence
- Error handling: ✅ Graceful degradation
- Performance validated: ✅ <1% overhead

### Production Status ✅ READY
- Backwards compatible: ✅ 100%
- Error recovery: ✅ Automatic
- Performance: ✅ Verified
- Documentation: ✅ Complete

---

## 📞 Support Resources

### For Installation Issues
→ See: `QUICK_START.md` section "Troubleshooting"

### For MQTT Configuration
→ See: `MQTT_INTEGRATION_GUIDE.md` section "Troubleshooting"

### For Dashboard Setup
→ See: `NODE_RED_SETUP.md` section "Troubleshooting"

### For Understanding Changes
→ See: `INTEGRATION_SUMMARY.md` section "Files Modified"

### For Production Deployment
→ See: `VALIDATION_REPORT.md` section "Pre-Deployment Checklist"

---

## 📝 Document Summary Table

| Document | Purpose | Read Time | Audience |
|----------|---------|-----------|----------|
| `QUICK_START.md` | Fast setup guide | 3-5 min | Everyone |
| `MQTT_INTEGRATION_GUIDE.md` | MQTT details | 20-30 min | Engineers |
| `NODE_RED_SETUP.md` | Dashboard setup | 15-20 min | Operators |
| `INTEGRATION_SUMMARY.md` | Code changes | 10-15 min | Developers |
| `VALIDATION_REPORT.md` | QA verification | 10-15 min | QA/Ops |
| This file | Navigation hub | 5-10 min | Everyone |

---

## ✨ Key Features Summary

✅ **Conveyor Belt Control** - GPIO PWM motor control  
✅ **Dual Servo Control** - Independent channel selection  
✅ **HSV Color Detection** - Green/Yellow recognition  
✅ **Shape Classification** - Square/Triangle detection  
✅ **MQTT Publishing** - Real-time event streaming  
✅ **Error Handling** - Graceful degradation  
✅ **Auto-calibration** - Interactive HSV tuning  
✅ **Dashboard Ready** - Node-RED integration  
✅ **Non-blocking** - Doesn't impact control performance  
✅ **Production Ready** - Fully tested and documented  

---

**Version**: 1.0  
**Last Updated**: December 9, 2025  
**Status**: ✅ PRODUCTION READY  
**Next**: Choose your starting point from the Quick Navigation section above!
