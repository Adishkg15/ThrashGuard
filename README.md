# ThrashGuard

**Deep Learning-Based Memory Pressure and Thrashing Classification using PatchTST**

ThrashGuard is a prototype system that uses a PatchTST-based Transformer model to classify operating system memory conditions into four categories: Normal, Memory Pressure, Approaching Thrashing, and Thrashing.

## Overview

Thrashing occurs when an operating system spends excessive time swapping pages between memory and storage, reducing overall system performance.

ThrashGuard analyzes sequential system-performance metrics to identify memory pressure patterns and classify the current memory state.

## Key Features

- **PatchTST Model:** Transformer-based time-series classification.
- **Multivariate Analysis:** Uses five system metrics:
  - Page fault rate
  - Swap I/O
  - Free memory
  - CPU usage
  - Run queue length
- **Sequence Processing:** Processes input sequences of 16 time steps.
- **Four-Class Classification:** Identifies different levels of memory pressure.
- **Flask Backend:** Handles model loading and prediction requests.
- **Dashboard:** Provides a visual interface for exploring classification results.

## Model Architecture

The model uses PatchTST with:
- Sequence length: 16
- Patch size: 4
- Patch stride: 4
- Transformer encoder layers: 2
- Attention heads: 4
- Model dimension: 64
- Optimizer: AdamW
- Learning rate: 0.0005

## Dataset

The prototype uses an augmented dataset containing **800 samples**, equally distributed across four classes (200 samples per class).

The dataset contains five system metrics along with timestamps and class labels.

## Experimental Results

The final evaluation on the test sequences produced:

| Metric | Result |
|---|---:|
| Test accuracy | 68.97% |
| Training accuracy | 82.40% |
| Test sequences | 145 |
| Classes | 4 |

These results represent the current prototype evaluation and are not a benchmark of production-system performance.

## Tech Stack

- Python
- PyTorch
- PatchTST
- Flask
- HTML, CSS and JavaScript
- Pandas, NumPy and Scikit-learn

## Run Locally

Clone the repository:

```bash
git clone https://github.com/Adishkg15/ThrashGuard.git
cd ThrashGuard
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
python app.py
```

Open the local dashboard at:

```text
http://127.0.0.1:5000
```

## Current Limitations

- The dataset is small and augmented; broader evaluation on real-world system traces is required.
- Live OS metric collection is not yet integrated end-to-end.
- Dashboard metrics are currently simulated.
- The model performs classification only; it does not automatically terminate processes or reallocate system resources.

## Future Scope

- Integrate real-time Linux system metric collection.
- Evaluate performance on larger, independently collected datasets.
- Improve classification accuracy and generalization.
- Explore early-warning mechanisms and automated recovery strategies.

---

**ThrashGuard — Intelligent Detection of Operating System Memory Pressure and Thrashing.**
