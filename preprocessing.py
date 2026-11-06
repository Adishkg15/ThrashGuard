import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =========================
# CONFIGURATION
# =========================

DATA_PATH = Path("data/thrashguard_augmented_800.csv")

FEATURES = [
    "page_fault_rate",
    "swap_io",
    "free_memory_mb",
    "cpu_usage",
    "run_queue_length"
]

TARGET = "label"

SEQUENCE_LENGTH = 16

NUM_CLASSES = 4


# =========================
# LOAD DATA
# =========================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n================================")
    print("DATASET LOADED")
    print("================================")

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nClass distribution:")
    print(df["label_name"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


# =========================
# VALIDATE DATA
# =========================

def validate_data(df):

    print("\n================================")
    print("DATA VALIDATION")
    print("================================")

    print("\nFeature statistics:")

    print(
        df[FEATURES]
        .describe()
        .round(2)
    )

    print("\nInvalid values:")

    print(
        "Negative page faults:",
        (df["page_fault_rate"] < 0).sum()
    )

    print(
        "Negative swap I/O:",
        (df["swap_io"] < 0).sum()
    )

    print(
        "Negative memory:",
        (df["free_memory_mb"] < 0).sum()
    )

    print(
        "CPU outside 0-100:",
        (
            (df["cpu_usage"] < 0) |
            (df["cpu_usage"] > 100)
        ).sum()
    )

    print(
        "Negative run queue:",
        (df["run_queue_length"] < 0).sum()
    )

    # Check that only the four expected classes exist
    expected_labels = {
        0: "Normal",
        1: "Memory Pressure",
        2: "Approaching Thrashing",
        3: "Thrashing"
    }

    actual_labels = dict(
        zip(
            df["label"],
            df["label_name"]
        )
    )

    if actual_labels != expected_labels:
        raise ValueError(
            f"Unexpected class mapping.\n"
            f"Expected: {expected_labels}\n"
            f"Found: {actual_labels}"
        )

    print("\nClass validation: PASSED")
    print("Number of classes:", NUM_CLASSES)


# =========================
# PREPARE FEATURES
# =========================

def prepare_features(df):

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


# =========================
# TRAIN / TEST SPLIT
# =========================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\n================================")
    print("TRAIN / TEST SPLIT")
    print("================================")

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index())

    print("\nTesting class distribution:")
    print(y_test.value_counts().sort_index())

    return X_train, X_test, y_train, y_test


# =========================
# SCALE FEATURES
# =========================

def scale_features(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n================================")
    print("FEATURE SCALING")
    print("================================")

    print("Scaler: StandardScaler")
    print("Features scaled:", len(FEATURES))

    return X_train_scaled, X_test_scaled, scaler


# =========================
# CREATE SEQUENCES
# =========================

def create_sequences(
    X,
    y,
    sequence_length
):

    sequences = []
    labels = []

    for i in range(
        len(X) - sequence_length + 1
    ):

        sequence = X[
            i:i + sequence_length
        ]

        label = y[
            i + sequence_length - 1
        ]

        sequences.append(sequence)
        labels.append(label)

    return (
        np.array(sequences),
        np.array(labels)
    )


# =========================
# CONVERT TO PYTORCH TENSORS
# =========================

def convert_to_tensors(
    X_train,
    y_train,
    X_test,
    y_test
):

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.long
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_test_tensor = torch.tensor(
        y_test,
        dtype=torch.long
    )

    return (
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor
    )


# =========================
# MAIN PIPELINE
# =========================

if __name__ == "__main__":

    # 1. Load
    df = load_data()

    # 2. Validate
    validate_data(df)

    # 3. Prepare features
    X, y = prepare_features(df)

    # 4. Train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # 5. Scale
    (
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = scale_features(
        X_train,
        X_test
    )

    # 6. Create time sequences
    (
        X_train_seq,
        y_train_seq
    ) = create_sequences(
        X_train_scaled,
        y_train.to_numpy(),
        SEQUENCE_LENGTH
    )

    (
        X_test_seq,
        y_test_seq
    ) = create_sequences(
        X_test_scaled,
        y_test.to_numpy(),
        SEQUENCE_LENGTH
    )

    # 7. Convert to tensors
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

    # =========================
    # FINAL INFORMATION
    # =========================

    print("\n================================")
    print("PREPROCESSING COMPLETE")
    print("================================")

    print(
        "Training tensor shape:",
        X_train_tensor.shape
    )

    print(
        "Training labels shape:",
        y_train_tensor.shape
    )

    print(
        "Testing tensor shape:",
        X_test_tensor.shape
    )

    print(
        "Testing labels shape:",
        y_test_tensor.shape
    )

    print(
        "\nSequence length:",
        SEQUENCE_LENGTH
    )

    print(
        "Number of features:",
        len(FEATURES)
    )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    print("\nClasses:")

    print("0 = Normal")
    print("1 = Memory Pressure")
    print("2 = Approaching Thrashing")
    print("3 = Thrashing")

    print("\nEverything is ready for model training!")