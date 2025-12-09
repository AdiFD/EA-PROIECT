# Quick Start - MQTT + Robot Sorting System

## TL;DR - Get Running in 5 Minutes

### On Raspberry Pi:

```bash
# 1. Install MQTT broker
sudo apt install mosquitto mosquitto-clients -y
sudo systemctl start mosquitto

# 2. Install Python MQTT client
pip3 install paho-mqtt

# 3. Run the robot
cd ~/EA-PROIECT
python3 pi_servo_control.py
```

### In another terminal (monitor MQTT):

```bash
mosquitto_sub -h localhost -t "robot/#"
```

You should see JSON events appearing as objects are detected and sorted.

---

## What MQTT Integration Does

Your robot now publishes real-time events to three topics:

| Topic | Event | When |
|-------|-------|------|
| `robot/detection` | Object detected | After detection, before sorting |
| `robot/belt` | Belt state change | When belt starts/stops |
| `robot/servo` | Servo movement | When servo rotates |

**Example output:**
```json
{"color":"yellow","shape":"square","center_x":310,"area":3200,"timestamp":1701000012.456}
{"status":"running","timestamp":1701000012.5}
{"channel":0,"angle":45,"timestamp":1701000012.6}
```

---

## Configuration

Edit `pi_servo_control.py` line 28-32 to change broker:

```python
MQTT_BROKER = "localhost"          # Change to IP if broker on different machine
MQTT_PORT = 1883                   # Default Mosquitto port
MQTT_TOPIC_DETECTION = "robot/detection"
MQTT_TOPIC_BELT = "robot/belt"
MQTT_TOPIC_SERVO = "robot/servo"
```

---

## Optional: Node-RED Dashboard

For a web-based monitoring UI:

```bash
# Install Node-RED
sudo npm install -g node-red node-red-dashboard

# Run it (or use systemd)
node-red
```

Then open: **http://localhost:1880/ui**

See `NODE_RED_SETUP.md` for full dashboard setup.

---

## Testing MQTT

### Test 1: Listen to all events
```bash
mosquitto_sub -h localhost -t "robot/#" -v
```

### Test 2: Send test message
```bash
mosquitto_pub -h localhost -t "robot/detection" -m '{"color":"green","shape":"square"}'
```

### Test 3: Check broker is running
```bash
sudo systemctl status mosquitto
```

---

## Troubleshooting

**"Connection refused"**
→ Start Mosquitto: `sudo systemctl start mosquitto`

**"No module named paho.mqtt"**
→ Install it: `pip3 install paho-mqtt`

**No messages appearing**
→ Check robot is running and objects are detected

**Remote monitoring from another computer**
1. Find Pi's IP: `hostname -I`
2. Change `MQTT_BROKER = "PI_IP_HERE"` in code
3. Subscribe from remote: `mosquitto_sub -h PI_IP_HERE -t "robot/#"`

---

## Files Reference

| File | Purpose |
|------|---------|
| `pi_servo_control.py` | Main robot code (MQTT integrated) |
| `calibrate_hsv.py` | HSV color calibration tool |
| `MQTT_INTEGRATION_GUIDE.md` | Full MQTT documentation |
| `NODE_RED_SETUP.md` | Dashboard setup guide |
| `hsv_values.json` | Saved color calibration values |

---

## Core Logic Unchanged

✅ **Belt control** - ConveyorBelt class (start/stop) unchanged  
✅ **Servo control** - ServoManager class (move) unchanged  
✅ **Detection** - detect_object() function unchanged  
✅ **Sorting sequence** - execute_on_demand_sequence() unchanged  

MQTT only **adds** event publishing. If Mosquitto isn't running, robot still works normally.

---

**Status**: ✅ Production Ready  
**MQTT Client**: paho-mqtt 1.6+  
**Broker**: Mosquitto  
**Dashboard**: Node-RED + node-red-dashboard  
