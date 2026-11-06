import pickle
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import PatchTSTForClassification

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    create_sequences,
    SEQUENCE_LENGTH
)


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_DIR = Path("models/patchtst")
SCALER_PATH = Path("models/scaler.pkl")

BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# FEATURES
# ==========================================

FEATURES = [
    "page_fault_rate",
    "swap_io",
    "free_memory_mb",
    "cpu_usage",
    "run_queue_length"
]


# ==========================================
# 4 CLASS CONFIGURATION
# ==========================================

CLASS_NAMES = [
    "Normal",
    "Memory Pressure",
    "Approaching Thrashing",
    "Thrashing"
]

CLASS_LABELS = [
    0,
    1,
    2,
    3
]


# ==========================================
# CHECK FILES
# ==========================================

if not MODEL_DIR.exists():

    raise FileNotFoundError(
        "Trained model not found. "
        "Run train.py first."
    )


if not SCALER_PATH.exists():

    raise FileNotFoundError(
        "Scaler not found. "
        "Run train.py first."
    )


# ==========================================
# HEADER
# ==========================================

print("\n========================================")
print("THRASHGUARD MODEL EVALUATION")
print("========================================")

print("Device:", DEVICE)

print("\nClasses:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{i} = {name}"
    )


# ==========================================
# LOAD DATA
# ==========================================

df = load_data()

X, y = prepare_features(df)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

(
    X_train,
    X_test,
    y_train,
    y_test
) = split_data(
    X,
    y
)


# ==========================================
# LOAD SCALER
# ==========================================

with open(
    SCALER_PATH,
    "rb"
) as f:

    scaler = pickle.load(f)


# ==========================================
# SCALE TEST DATA
# ==========================================

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================
# CREATE TEST SEQUENCES
# ==========================================

(
    X_test_seq,
    y_test_seq
) = create_sequences(
    X_test_scaled,
    y_test.to_numpy(),
    SEQUENCE_LENGTH
)


# ==========================================
# CONVERT TO TENSORS
# ==========================================

X_test_tensor = torch.tensor(
    X_test_seq,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test_seq,
    dtype=torch.long
)


# ==========================================
# TEST DATASET
# ==========================================

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


print(
    "\nTest sequences:",
    len(test_dataset)
)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("\nLoading PatchTST model...")

model = PatchTSTForClassification.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)

model.eval()

print("Model loaded successfully.")


# ==========================================
# MAKE PREDICTIONS
# ==========================================

all_predictions = []

all_labels = []


with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(
            DEVICE
        )

        outputs = model(
            past_values=batch_X
        )

        logits = outputs.prediction_logits

        predictions = torch.argmax(
            logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            batch_y.numpy()
        )


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print("\n========================================")
print("RESULTS")
print("========================================")

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

report = classification_report(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=CLASS_LABELS
)

print(cm)

print("\nRows = Actual")
print("Columns = Predicted")


# ==========================================
# CLASS-WISE ACCURACY
# ==========================================

print("\n========================================")
print("CLASS-WISE ACCURACY")
print("========================================")


for i, class_name in enumerate(
    CLASS_NAMES
):

    total = cm[i].sum()

    if total > 0:

        class_accuracy = (
            cm[i, i] / total
        )

    else:

        class_accuracy = 0.0


    print(
        f"{class_name}: "
        f"{class_accuracy * 100:.2f}%"
    )


# ==========================================
# PREDICTION DISTRIBUTION
# ==========================================

print("\n========================================")
print("PREDICTION DISTRIBUTION")
print("========================================")


for i, class_name in enumerate(
    CLASS_NAMES
):

    count = (
        np.sum(
            np.array(all_predictions) == i
        )
    )

    print(
        f"{class_name}: {count}"
    )


# ==========================================
# ACTUAL DISTRIBUTION
# ==========================================

print("\n========================================")
print("ACTUAL TEST DISTRIBUTION")
print("========================================")


for i, class_name in enumerate(
    CLASS_NAMES
):

    count = (
        np.sum(
            np.array(all_labels) == i
        )
    )

    print(
        f"{class_name}: {count}"
    )


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "\nModel:",
    "PatchTST"
)

print(
    "Number of classes:",
    len(CLASS_NAMES)
)

print(
    "Sequence length:",
    SEQUENCE_LENGTH
)

print(
    "Test accuracy:",
    f"{accuracy * 100:.2f}%"
)

print("\nClasses:")

for i, name in enumerate(
    CLASS_NAMES
):

    print(
        f"{i} = {name}"
    )

print("\nReady for inference!")