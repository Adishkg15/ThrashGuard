import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import (
    load_data,
    validate_data,
    prepare_features,
    split_data,
    scale_features,
    create_sequences,
    convert_to_tensors,
    SEQUENCE_LENGTH,
    FEATURES
)


# ==========================================
# CONFIGURATION
# ==========================================

NUM_CLASSES = 4
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.0005

MODEL_DIR = Path("models/patchtst")
SCALER_PATH = Path("models/scaler.pkl")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD AND PREPARE DATA
# ==========================================

print("\n========================================")
print("THRASHGUARD PATCHTST TRAINING")
print("========================================")

print("Device:", DEVICE)

df = load_data()

validate_data(df)

X, y = prepare_features(df)

X_train, X_test, y_train, y_test = split_data(
    X,
    y
)

X_train_scaled, X_test_scaled, scaler = scale_features(
    X_train,
    X_test
)


# ==========================================
# CREATE SEQUENCES
# ==========================================

X_train_seq, y_train_seq = create_sequences(
    X_train_scaled,
    y_train.to_numpy(),
    SEQUENCE_LENGTH
)

X_test_seq, y_test_seq = create_sequences(
    X_test_scaled,
    y_test.to_numpy(),
    SEQUENCE_LENGTH
)


# ==========================================
# CONVERT TO TENSORS
# ==========================================

(
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor
) = convert_to_tensors(
    X_train_seq,
    y_train_seq,
    X_test_seq,
    y_test_seq
)


print("\n========================================")
print("DATA READY")
print("========================================")

print("Training:", X_train_tensor.shape)
print("Testing :", X_test_tensor.shape)


# ==========================================
# DATA LOADERS
# ==========================================

train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ==========================================
# PATCHTST MODEL
# ==========================================

config = PatchTSTConfig(
    num_input_channels=len(FEATURES),
    context_length=SEQUENCE_LENGTH,

    patch_length=4,
    patch_stride=4,

    num_hidden_layers=2,
    d_model=64,
    num_attention_heads=4,
    ffn_dim=128,

    dropout=0.1,
    head_dropout=0.1,

    pooling_type="mean",

    num_targets=NUM_CLASSES,

    scaling=None
)

model = PatchTSTForClassification(
    config=config
)

model.to(DEVICE)


print("\n========================================")
print("MODEL")
print("========================================")

print("Architecture: PatchTST")
print("Input features:", len(FEATURES))
print("Sequence length:", SEQUENCE_LENGTH)
print("Patch length:", 4)
print("Patch stride:", 4)
print("Classes:", NUM_CLASSES)


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ==========================================
# TRAINING
# ==========================================

print("\n========================================")
print("TRAINING STARTED")
print("========================================")

best_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            past_values=batch_X
        )

        logits = outputs.prediction_logits

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(
                past_values=batch_X
            )

            logits = outputs.prediction_logits

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Test Acc: {val_accuracy:.4f}"
    )


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_accuracy > best_accuracy:

        best_accuracy = val_accuracy

        MODEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model.save_pretrained(
            MODEL_DIR
        )

        print(
            f"  ✓ Best model saved "
            f"(accuracy: {best_accuracy:.4f})"
        )


# ==========================================
# SAVE SCALER
# ==========================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

with open(
    SCALER_PATH,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


# ==========================================
# FINAL EVALUATION
# ==========================================

print("\n========================================")
print("FINAL EVALUATION")
print("========================================")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(DEVICE)

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


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(
    f"\nFinal Test Accuracy: {accuracy:.4f}"
)

print(
    f"Best Test Accuracy: {best_accuracy:.4f}"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

class_names = [
    "Normal",
    "Memory Pressure",
    "Approaching Thrashing",
    "Thrashing"
]

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names,
        zero_division=0
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        all_labels,
        all_predictions
    )
)


# ==========================================
# FINAL MESSAGE
# ==========================================

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Model saved to:",
    MODEL_DIR
)

print(
    "Scaler saved to:",
    SCALER_PATH
)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i} = {name}")

print("\nReady for inference!")