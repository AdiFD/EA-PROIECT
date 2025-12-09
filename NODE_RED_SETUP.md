# Node-RED Setup Guide - Robot Monitoring Dashboard

## Installation

### 1. Install Node.js and npm (if not already installed)

On Raspberry Pi:
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
node --version  # Verify installation
npm --version
```

### 2. Install Node-RED

```bash
sudo npm install -g node-red
```

### 3. Install Dashboard Plugin

```bash
sudo npm install -g node-red-dashboard
```

### 4. Configure Mosquitto (if needed)

Ensure Mosquitto allows anonymous connections (default in most setups):

```bash
sudo nano /etc/mosquitto/mosquitto.conf
```

Add these lines if not present:
```
listener 1883
protocol mqtt
allow_anonymous true
```

Then restart:
```bash
sudo systemctl restart mosquitto
```

## Running Node-RED

### As a standalone service:

```bash
node-red
```

You'll see:
```
Welcome to Node-RED
...
[info] Started flows
[info] Server now running at http://127.0.0.1:1880/
```

Access the editor at: **http://localhost:1880/**

### As a systemd service (auto-start on boot):

Create service file:
```bash
sudo nano /etc/systemd/system/node-red.service
```

Add:
```ini
[Unit]
Description=Node-RED
After=network.target

[Service]
ExecStart=/usr/bin/node-red
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=node-red

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable node-red
sudo systemctl start node-red
sudo systemctl status node-red
```

## Dashboard Access

Once running, access the dashboard at:
- **Local**: `http://localhost:1880/ui`
- **Remote**: `http://PI_IP:1880/ui` (replace PI_IP with actual Raspberry Pi IP)

Find your Pi's IP:
```bash
hostname -I
```

## Import Pre-built Flow

The following flow provides a complete monitoring dashboard. Copy this JSON:

```json
{
  "id": "robot_monitoring_flow",
  "label": "Robot Monitoring",
  "nodes": [
    {
      "id": "mqtt_detection",
      "type": "mqtt in",
      "z": "robot_monitoring_flow",
      "name": "Detection Events",
      "topic": "robot/detection",
      "qos": "2",
      "broker": "mosquitto_broker",
      "x": 100,
      "y": 100
    },
    {
      "id": "mqtt_belt",
      "type": "mqtt in",
      "z": "robot_monitoring_flow",
      "name": "Belt Events",
      "topic": "robot/belt",
      "qos": "2",
      "broker": "mosquitto_broker",
      "x": 100,
      "y": 200
    },
    {
      "id": "mqtt_servo",
      "type": "mqtt in",
      "z": "robot_monitoring_flow",
      "name": "Servo Events",
      "topic": "robot/servo",
      "qos": "2",
      "broker": "mosquitto_broker",
      "x": 100,
      "y": 300
    },
    {
      "id": "detection_display",
      "type": "ui_text",
      "z": "robot_monitoring_flow",
      "name": "Last Detection",
      "label": "Last Detection",
      "format": "{{msg.payload}}",
      "layout": "row-center",
      "className": "",
      "x": 500,
      "y": 100,
      "group": "dashboard_group"
    },
    {
      "id": "belt_display",
      "type": "ui_text",
      "z": "robot_monitoring_flow",
      "name": "Belt Status",
      "label": "Belt Status",
      "format": "{{msg.payload.status}}",
      "layout": "row-center",
      "className": "",
      "x": 500,
      "y": 200,
      "group": "dashboard_group"
    },
    {
      "id": "servo_display",
      "type": "ui_text",
      "z": "robot_monitoring_flow",
      "name": "Servo Info",
      "label": "Last Servo Movement",
      "format": "Channel {{msg.payload.channel}} → {{msg.payload.angle}}°",
      "layout": "row-center",
      "className": "",
      "x": 500,
      "y": 300,
      "group": "dashboard_group"
    },
    {
      "id": "detection_parser",
      "type": "json",
      "z": "robot_monitoring_flow",
      "name": "Parse Detection JSON",
      "property": "payload",
      "action": "obj",
      "x": 300,
      "y": 100
    },
    {
      "id": "belt_parser",
      "type": "json",
      "z": "robot_monitoring_flow",
      "name": "Parse Belt JSON",
      "property": "payload",
      "action": "obj",
      "x": 300,
      "y": 200
    },
    {
      "id": "servo_parser",
      "type": "json",
      "z": "robot_monitoring_flow",
      "name": "Parse Servo JSON",
      "property": "payload",
      "action": "obj",
      "x": 300,
      "y": 300
    },
    {
      "id": "history_inject",
      "type": "debug",
      "z": "robot_monitoring_flow",
      "name": "Event Log",
      "active": true,
      "tosidebar": true,
      "console": false,
      "tostatus": false,
      "complete": "payload",
      "targetType": "msg",
      "statusVal": "",
      "statusType": "auto",
      "x": 500,
      "y": 400
    }
  ],
  "configs": [
    {
      "id": "mosquitto_broker",
      "type": "mqtt-broker",
      "name": "Mosquitto",
      "broker": "127.0.0.1",
      "port": "1883",
      "clientid": "node-red-robot",
      "usetls": false,
      "protocolVersion": "4",
      "keepalive": "60",
      "cleansession": true,
      "birthTopic": "",
      "birthQos": "0",
      "birthPayload": "",
      "closeTopic": "",
      "closeQos": "0",
      "closePayload": "",
      "willTopic": "",
      "willQos": "0",
      "willPayload": ""
    },
    {
      "id": "dashboard_group",
      "type": "ui_group",
      "name": "Robot Status",
      "tab": "robot_tab"
    },
    {
      "id": "robot_tab",
      "type": "ui_tab",
      "name": "Robot Monitoring",
      "icon": "fa-heartbeat"
    }
  ]
}
```

### How to import:

1. Open Node-RED editor: `http://localhost:1880/`
2. Click ☰ menu → **Import**
3. Paste the JSON above
4. Click **Import**
5. Click **Deploy** (red button, top right)
6. Access dashboard: `http://localhost:1880/ui`

## Manual Flow Creation (Alternative)

If you prefer to build it manually:

### Step 1: Add MQTT Input Nodes

1. Search for "mqtt in" in left panel → Drag 3 nodes to canvas
2. Configure each:
   - **Node 1**: Topic = `robot/detection`
   - **Node 2**: Topic = `robot/belt`
   - **Node 3**: Topic = `robot/servo`
3. For each node, set Broker = Add new broker → Configure:
   - Server: `127.0.0.1` (or your Pi's hostname)
   - Port: `1883`
   - Click **Add**

### Step 2: Add JSON Parser Nodes

1. Search for "json" → Add 3 JSON nodes
2. Set Property: `msg.payload` for each
3. Connect MQTT nodes → JSON parser nodes

### Step 3: Add Display Widgets

1. Search "ui_text" → Add 3 text display nodes
2. Configure:
   - **Detection**: Format = `Color: {{msg.payload.color}}, Shape: {{msg.payload.shape}}, Area: {{msg.payload.area}}`
   - **Belt**: Format = `Status: {{msg.payload.status}}`
   - **Servo**: Format = `Channel: {{msg.payload.channel}}, Angle: {{msg.payload.angle}}°`
3. For each widget, assign to a new Group (create "Robot Status" group)

### Step 4: Deploy

Click the red **Deploy** button (top right)

## Dashboard Features

Once imported/created, your dashboard will show:

1. **Detection Display** - Latest detected object color, shape, and area
2. **Belt Status** - Current belt state (running/stopped)
3. **Servo Info** - Last servo movement details
4. **Event Log** - Full debug log of all MQTT messages

## Customization

### Add a Chart for Detection History

1. Add "ui_chart" node
2. Set Title: "Detection History"
3. Connect JSON parser → Chart
4. In the chart config, set Label to `{{msg.payload.color}}`

### Add a Gauge for Motor Speed

1. Add "ui_gauge" node
2. Set Min: 0, Max: 100
3. Subscribe to `robot/belt` and calculate expected duty cycle

### Add Counters

1. Add "change" nodes to count events
2. Use `msg.count++` in JSON mode
3. Display with "ui_text" nodes

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Cannot find module node-red-dashboard" | Run `sudo npm install -g node-red-dashboard` and restart Node-RED |
| Dashboard not loading at `/ui` | Ensure node-red-dashboard is installed; access error log with `node-red` in terminal |
| MQTT broker connection fails | Check Mosquitto is running: `sudo systemctl status mosquitto`; verify IP in config |
| No messages appearing | Check MQTT topics match exactly (case-sensitive); verify robot is publishing with `mosquitto_sub -h localhost -t "robot/#"` |
| Port 1880 already in use | Kill existing Node-RED: `pkill -f node-red` or use different port with `--port 1881` |

## Advanced: Authentication

To add password protection to Mosquitto:

```bash
# Create user
sudo mosquitto_passwd -c /etc/mosquitto/passwd.txt username

# Edit config
sudo nano /etc/mosquitto/mosquitto.conf
# Add: password_file /etc/mosquitto/passwd.txt
# Change: allow_anonymous false

sudo systemctl restart mosquitto
```

Then update Node-RED MQTT config with username/password.

---

**Dashboard Features**: Real-time monitoring, JSON visualization, event logging  
**Status**: Production-ready  
**Last Updated**: December 2025
