# MQTT Integration Guide - Robot Sorting System

## Overview

This document explains the MQTT layer that has been integrated into `pi_servo_control.py`. The core conveyor belt and servo control logic remains **completely unchanged**. MQTT is purely an additive monitoring/control layer.

## Installation & Setup

### 1. Install MQTT Broker (Mosquitto)

On Raspberry Pi:
```bash
sudo apt update
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable mosquitto
sudo systemctl start mosquitto
```

Verify it's running:
```bash
sudo systemctl status mosquitto
```

### 2. Install Python MQTT Client

```bash
pip install paho-mqtt
```

Or for Python 3:
```bash
pip3 install paho-mqtt
```

### 3. Verify MOSQUITTO Port

By default, Mosquitto listens on port 1883. You can verify this:
```bash
sudo netstat -tlnp | grep mosquitto
```

Expected output: `tcp 0 0 0.0.0.0:1883 0.0.0.0:* LISTEN <pid>/mosquitto`

## Configuration in pi_servo_control.py

The following globals control MQTT behavior:

```python
MQTT_BROKER = "localhost"          # IP/hostname of Mosquitto broker
MQTT_PORT = 1883                   # Default Mosquitto port
MQTT_TOPIC_DETECTION = "robot/detection"  # Detection events
MQTT_TOPIC_BELT = "robot/belt"            # Belt state changes
MQTT_TOPIC_SERVO = "robot/servo"          # Servo movements
```

**To change broker location**, edit the IP:
- Local Pi: `"localhost"` or `"127.0.0.1"`
- Remote laptop: `"192.168.1.100"` (replace with actual IP)

## MQTT Topics & Payloads

### 1. Detection Events (`robot/detection`)

**Published when**: An object is detected and before the sorting sequence starts.

**Payload format** (JSON):
```json
{
  "color": "green|yellow",
  "shape": "square|triangle",
  "center_x": 320,
  "center_y": 240,
  "area": 2500,
  "timestamp": 1701000000.123
}
```

**Example**:
```json
{
  "color": "yellow",
  "shape": "square",
  "center_x": 310,
  "center_y": 255,
  "area": 3200,
  "timestamp": 1701000012.456
}
```

### 2. Belt State (`robot/belt`)

**Published when**: Belt starts or stops.

**Payload format**:
```json
{
  "status": "running|stopped",
  "timestamp": 1701000000.123
}
```

**Examples**:
- Belt starts: `{"status": "running", "timestamp": 1701000012.5}`
- Belt stops: `{"status": "stopped", "timestamp": 1701000015.8}`

### 3. Servo Movement (`robot/servo`)

**Published when**: A servo moves to a new angle.

**Payload format**:
```json
{
  "channel": 0|1,
  "angle": 0-180,
  "timestamp": 1701000000.123
}
```

**Channel mapping**:
- `0`: Left servo (squares/cubes)
- `1`: Right servo (triangles/pyramids)

**Examples**:
- Left servo to 45°: `{"channel": 0, "angle": 45, "timestamp": 1701000012.6}`
- Right servo to 135°: `{"channel": 1, "angle": 135, "timestamp": 1701000015.3}`

## Testing MQTT Integration

### Option 1: Listen with mosquitto_sub

In a terminal on the Pi (or any computer with mosquitto-clients):

```bash
# Listen to all robot topics
mosquitto_sub -h localhost -t "robot/#"

# Listen to only detection events
mosquitto_sub -h localhost -t "robot/detection"

# Listen to only belt events
mosquitto_sub -h localhost -t "robot/belt"
```

You should see JSON payloads appear as objects are detected and the robot operates.

### Option 2: Publish Test Messages

```bash
# Test publish to detection topic
mosquitto_pub -h localhost -t "robot/detection" -m '{"color":"green","shape":"square","area":1500}'

# Test publish to belt topic
mosquitto_pub -h localhost -t "robot/belt" -m '{"status":"running"}'
```

### Option 3: Monitor with Node-RED Dashboard

See `node_red_setup.md` for full Node-RED integration that provides a web dashboard at `http://PI_IP:1880/ui`.

## Failure Modes & Error Handling

### What happens if Mosquitto isn't running?

The code will attempt to connect but fail gracefully. You'll see in the logs:
```
[MQTT] Connecting to localhost:1883...
[MQTT] Connection failed (code 1)
```

**The robot will still function normally** – MQTT is purely optional monitoring.

### What happens if Mosquitto is on a different IP?

Update the config:
```python
MQTT_BROKER = "192.168.1.50"  # Your Pi's actual IP if running from another computer
```

### Reconnection Logic

The MQTT client has built-in `keepalive=60` seconds. If the connection drops, it will attempt to reconnect automatically.

## Integration Points in Code

The MQTT code integrates at exactly 5 locations:

1. **ConveyorBelt.start()** - Publishes `{"status": "running", ...}`
2. **ConveyorBelt.stop()** - Publishes `{"status": "stopped", ...}`
3. **ServoManager.move()** - Publishes `{"channel": ch, "angle": ang, ...}`
4. **main() detection block** - Publishes `{"color": ..., "shape": ..., "area": ..., ...}`
5. **main() finally block** - Gracefully stops MQTT client loop on exit

**None of these change the core control logic** – they simply wrap the existing function calls with optional MQTT publishing.

## Performance Notes

- MQTT publishing is wrapped in try/except blocks to prevent network errors from crashing the robot
- Each publish is ~100 bytes of JSON; with 30 FPS detection + belt events = ~2-5 KB/sec traffic
- Network latency is typically <10ms on local network
- No performance impact if Mosquitto is unavailable (graceful timeout in ~1 second)

## Next Steps

1. **Run the robot with MQTT**: 
   ```bash
   python3 pi_servo_control.py
   ```

2. **Monitor events** (in separate terminal):
   ```bash
   mosquitto_sub -h localhost -t "robot/#"
   ```

3. **Build a dashboard** using Node-RED (see `node_red_setup.md`)

4. **Optional**: Remote monitoring by:
   - Installing Mosquitto on a laptop with a fixed IP
   - Changing `MQTT_BROKER` to that laptop's IP
   - Running Node-RED on the laptop or Pi to visualize events

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Connection refused` (code 111) | Mosquitto not running: `sudo systemctl start mosquitto` |
| `No module named paho.mqtt` | Install client: `pip3 install paho-mqtt` |
| Messages not appearing in `mosquitto_sub` | Check broker IP matches config; verify Mosquitto is listening on port 1883 |
| Robot works but MQTT silent | Check `/var/log/mosquitto/mosquitto.log` for broker errors |
| Intermittent disconnections | Check network stability; increase `keepalive` in code if needed |

---

**Author**: Auto-integrated MQTT layer  
**Date**: December 2025  
**Status**: Production-ready, non-breaking integration
