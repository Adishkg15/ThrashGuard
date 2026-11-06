from flask import Flask, request, jsonify, render_template
from pathlib import Path
import pickle
import numpy as np
import torch
from transformers import PatchTSTForClassification


# ============================================================
# THRASHGUARD CONFIGURATION
# ============================================================

MODEL_DIR = Path("models/patchtst")
SCALER_PATH = Path("models/scaler.pkl")

SEQUENCE_LENGTH = 16

FEATURES = [
    "page_fault_rate",
    "swap_io",
    "free_memory_mb",
    "cpu_usage",
    "run_queue_length"
]

CLASS_NAMES = [
    "Normal",
    "Memory Pressure",
    "Approaching Thrashing",
    "Thrashing"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# LOAD SCALER
# ============================================================

if not SCALER_PATH.exists():
    raise FileNotFoundError(
        "Scaler not found. Please run train.py first."
    )

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)


# ============================================================
# LOAD PATCHTST MODEL
# ============================================================

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        "PatchTST model not found. Please run train.py first."
    )

model = PatchTSTForClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()


# ============================================================
# METRIC BUFFER
# ============================================================

metric_buffer = []


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_sequence(sequence):

    data = np.array(
        sequence,
        dtype=float
    )

    expected_shape = (
        SEQUENCE_LENGTH,
        len(FEATURES)
    )

    if data.shape != expected_shape:
        raise ValueError(
            f"Expected {expected_shape}, "
            f"received {data.shape}"
        )

    # Scale using training scaler
    scaled_data = scaler.transform(data)

    # Convert to PyTorch tensor
    tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32
    )

    # Add batch dimension
    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    # Model prediction
    with torch.no_grad():

        outputs = model(
            past_values=tensor
        )

        logits = outputs.prediction_logits

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()

    return {
        "label": predicted_class,

        "label_name": CLASS_NAMES[
            predicted_class
        ],

        "confidence": round(
            confidence * 100,
            2
        ),

        "probabilities": {
            CLASS_NAMES[i]: round(
                probabilities[0, i].item() * 100,
                2
            )
            for i in range(len(CLASS_NAMES))
        }
    }


# ============================================================
# HOME / DASHBOARD
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "model": "PatchTST",
        "device": str(DEVICE),
        "num_classes": len(CLASS_NAMES)
    })


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route("/info")
def info():

    return jsonify({

        "application": "ThrashGuard",

        "model": "PatchTST",

        "device": str(DEVICE),

        "sequence_length": SEQUENCE_LENGTH,

        "features": FEATURES,

        "classes": {
            "0": "Normal",
            "1": "Memory Pressure",
            "2": "Approaching Thrashing",
            "3": "Thrashing"
        }

    })


# ============================================================
# DIRECT SEQUENCE PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        sequence = data.get(
            "sequence"
        )

        if sequence is None:
            return jsonify({
                "error": "Missing 'sequence'."
            }), 400

        result = predict_sequence(
            sequence
        )

        return jsonify(result)

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 400


# ============================================================
# ADD ONE METRIC SAMPLE
# ============================================================

@app.route(
    "/metrics",
    methods=["POST"]
)
def add_metric():

    global metric_buffer

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No JSON data received."
            }), 400

        # --------------------------------------------
        # Extract five metrics
        # --------------------------------------------

        sample = [

            float(
                data["page_fault_rate"]
            ),

            float(
                data["swap_io"]
            ),

            float(
                data["free_memory_mb"]
            ),

            float(
                data["cpu_usage"]
            ),

            float(
                data["run_queue_length"]
            )

        ]

        # --------------------------------------------
        # Add to buffer
        # --------------------------------------------

        metric_buffer.append(
            sample
        )

        # Keep only latest 16 measurements

        if len(metric_buffer) > SEQUENCE_LENGTH:

            metric_buffer = metric_buffer[
                -SEQUENCE_LENGTH:
            ]

        response = {

            "status": "metric_added",

            "buffer_size": len(
                metric_buffer
            ),

            "required": SEQUENCE_LENGTH,

            "prediction": None

        }

        # --------------------------------------------
        # Predict after 16 samples
        # --------------------------------------------

        if len(metric_buffer) == SEQUENCE_LENGTH:

            result = predict_sequence(
                metric_buffer
            )

            response["prediction"] = result

        return jsonify(
            response
        )

    except KeyError as error:

        return jsonify({

            "error":
            f"Missing metric: {error}"

        }), 400

    except ValueError as error:

        return jsonify({

            "error":
            f"Invalid metric value: {error}"

        }), 400

    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 400


# ============================================================
# GET CURRENT METRICS
# ============================================================

@app.route(
    "/metrics",
    methods=["GET"]
)
def get_metrics():

    return jsonify({

        "buffer_size":
        len(metric_buffer),

        "required":
        SEQUENCE_LENGTH,

        "metrics":
        metric_buffer

    })


# ============================================================
# RESET MONITORING
# ============================================================

@app.route(
    "/reset",
    methods=["POST"]
)
def reset():

    global metric_buffer

    metric_buffer = []

    return jsonify({

        "status":
        "buffer_reset",

        "buffer_size":
        0

    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 55)
    print("                 THRASHGUARD")
    print("=" * 55)

    print()
    print("Model          :", "PatchTST")
    print("Device         :", DEVICE)
    print("Sequence Length:", SEQUENCE_LENGTH)

    print()
    print("Features:")

    for feature in FEATURES:
        print("  -", feature)

    print()
    print("Classes:")

    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {i} = {name}"
        )

    print()
    print("Model loaded successfully.")
    print("Scaler loaded successfully.")

    print()
    print("Starting Flask server...")
    print()
    print("Dashboard:")
    print("http://127.0.0.1:5000")

    print()
    print("=" * 55)
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )