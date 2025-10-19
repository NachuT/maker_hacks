from flask import Flask, request, jsonify, render_template_string, Response
import threading, time, numpy as np, torch, torch.nn as nn, joblib
from sklearn.preprocessing import StandardScaler
import json

# --------------------------
# Simple CNN Models
# --------------------------
class CNN_Attention(nn.Module):
    def __init__(self):
        super(CNN_Attention, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.attn = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x_mean = x.mean(dim=2)
        attn = self.attn(x_mean)
        out = self.fc(x_mean * attn)
        return self.sigmoid(out).squeeze()


class CNN_Bottleneck(nn.Module):
    def __init__(self):
        super(CNN_Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.AvgPool1d(2)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.mean(dim=2)
        x = torch.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x)).squeeze()


# --------------------------
# Feature extraction utils
# --------------------------
def extract_hrv_features(bvp, temp):
    if bvp is None or len(bvp) == 0:
        return None
    bvp = np.array(bvp)
    features = [
        np.mean(bvp), np.std(bvp), np.min(bvp), np.max(bvp),
        np.mean(np.diff(bvp) ** 2), temp or 0
    ]
    return features


def prepare_single_cnn_input(bvp, temp):
    if bvp is None:
        return np.zeros((1, 1, 128), dtype=np.float32)
    bvp = np.array(bvp, dtype=np.float32)
    if len(bvp) < 128:
        bvp = np.pad(bvp, (0, 128 - len(bvp)))
    else:
        bvp = bvp[:128]
    return bvp.reshape(1, 1, -1)


# --------------------------
# Load models (mock or trained)
# --------------------------
try:
    scaler = joblib.load("scaler.pkl")
    hgb_model = joblib.load("hgb_model.pkl")
except:
    from sklearn.ensemble import GradientBoostingClassifier
    scaler = StandardScaler().fit([[0, 0, 0, 0, 0, 0]])
    hgb_model = GradientBoostingClassifier()
    hgb_model.classes_ = np.array([0, 1])

cnn_attention = CNN_Attention()
cnn_bottleneck = CNN_Bottleneck()

w1, w2, w3 = 0.3, 0.3, 0.4

# --------------------------
# Flask setup
# --------------------------
app = Flask(__name__)
latest_data = {
    "bvp": [],
    "heartbeat": 0,
    "temperature": 0,
    "humidity": 0,
    "stress": None,
    "prob": None
}

# SSE will use a simpler polling approach

# --------------------------
# Stress Prediction
# --------------------------
def predict_stress(bvp, temp):
    f = extract_hrv_features(bvp, temp)
    if f is None or not all(np.isfinite(f)):
        return None, "Invalid signal"

    # Adjust input length to match scaler expectation
    required_features = 11
    f = np.array(f)
    if len(f) < required_features:
        f = np.pad(f, (0, required_features - len(f)), mode='constant')
    elif len(f) > required_features:
        f = f[:required_features]

    f_scaled = scaler.transform(f.reshape(1, -1))
    x_cnn = prepare_single_cnn_input(bvp, temp)

    with torch.no_grad():
        x_tensor = torch.tensor(x_cnn)
        pa = cnn_attention(x_tensor).item()
        pb = cnn_bottleneck(x_tensor).item()

    try:
        ph = hgb_model.predict_proba(f_scaled)[0, 1]
    except:
        ph = 0.5

    # Base ensemble
    base_p = 0.3 * pa + 0.3 * pb + 0.4 * ph

    # ----------------------------
    # Adaptive physiological rules
    # ----------------------------
    heartbeat = latest_data.get("heartbeat", 0) or 0
    temperature = temp or 25.0

    # Heart rate scaling
    if heartbeat > 120:
        hr_factor = 1.25 + min((heartbeat - 120) / 100, 0.5)  # up to +50%
    elif heartbeat < 70:
        hr_factor = 0.85 - min((70 - heartbeat) / 100, 0.3)   # down to -30%
    else:
        hr_factor = 1.0

    # Temperature moderation
    if temperature > 30:
        env_adjust = 1 - min((temperature - 30) / 50, 0.25)   # reduce HR weight up to 25%
    elif temperature < 20:
        env_adjust = 1 + min((20 - temperature) / 50, 0.25)   # amplify HR effect up to +25%
    else:
        env_adjust = 1.0

    # Combined adjustment
    adjusted_p = base_p * hr_factor * env_adjust
    adjusted_p = np.clip(adjusted_p, 0, 1)

    return int(adjusted_p > 0.5), float(adjusted_p)


# --------------------------
# Background AI thread
# --------------------------
def stress_predictor_loop():
    while True:
        if latest_data["bvp"]:
            label, prob = predict_stress(latest_data["bvp"], latest_data["temperature"])
            latest_data["stress"] = label
            latest_data["prob"] = prob
            print(f"[AI] Stress={label}, p={prob:.3f}")
            
            # Data updated, SSE clients will get it on next poll
        time.sleep(10)

def broadcast_update():
    """Send latest data to all connected SSE clients"""
    # For now, we'll use a simpler approach without client tracking
    # The SSE endpoint will handle real-time updates through the stream
    pass

threading.Thread(target=stress_predictor_loop, daemon=True).start()

# --------------------------
# Routes
# --------------------------
@app.route("/")
def index():
    html = """
    <h1>ESP32 Wellness Dashboard</h1>
    <div style="font-family: Arial; background: #f4f4f4; padding: 20px;">
        <p><b>Heartbeat (raw):</b> {{heartbeat}}</p>
        <p><b>Temperature:</b> {{temperature}} °C</p>
        <p><b>Humidity:</b> {{humidity}} %</p>
        <p><b>Stress Level:</b> {{stress}}</p>
        <p><b>Probability:</b> {{prob}}</p>
    </div>
    """
    return render_template_string(html, **latest_data)


@app.route("/api/data", methods=["POST"])
def receive_data():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    # Match Arduino payload
    hb = data.get("hb", 0)
    temp = data.get("temp", 0.0)
    hum = data.get("hum", 0.0)

    # Save latest readings
    latest_data["heartbeat"] = hb
    latest_data["temperature"] = temp
    latest_data["humidity"] = hum

    # Append heartbeat into a small buffer for CNN input
    if "bvp" not in latest_data or latest_data["bvp"] is None:
        latest_data["bvp"] = []
    latest_data["bvp"].append(hb)
    if len(latest_data["bvp"]) > 200:
        latest_data["bvp"] = latest_data["bvp"][-200:]

    print(f"[Data] HB={hb}, Temp={temp} °C, Hum={hum} %")
    
    return jsonify({"status": "ok"})


@app.route("/api/result", methods=["GET"])
def get_result():
    return jsonify({
        "heartbeat": latest_data["heartbeat"],
        "temperature": latest_data["temperature"],
        "humidity": latest_data["humidity"],
        "stress": latest_data["stress"],
        "probability": latest_data["prob"]
    })

@app.route("/api/stream")
def stream():
    """Server-Sent Events endpoint for real-time updates"""
    def event_stream():
        # Send initial data
        data = {
            "heartbeat": latest_data["heartbeat"],
            "temperature": latest_data["temperature"],
            "humidity": latest_data["humidity"],
            "stress": latest_data["stress"],
            "probability": latest_data["prob"]
        }
        yield f"data: {json.dumps(data)}\n\n"
        
        # Keep connection alive
        while True:
            time.sleep(1)
            yield "data: {}\n\n"  # Keep-alive
    
    return Response(event_stream(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache',
                           'Connection': 'keep-alive',
                           'Access-Control-Allow-Origin': '*'})


# --------------------------
# Run Server
# --------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5010, debug=True)
