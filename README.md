# MNIST Transformer – Tiled Digit Sequence Model

This project implements an **Encoder-Decoder Transformer** that processes **tiled MNIST digit images** and predicts the sequence of digits represented in the image, including support for **blanks and variable-length sequences**.

---

## What the Model Does

- Inputs a 2×2 **tiled MNIST image** (up to 4 digits, blanks allowed)
- Encodes it using a **Vision Transformer-style encoder**
- Decodes the sequence of digits using an **autoregressive decoder**
- Handles `<start>`, `<end>`, and `<pad>` tokens for training
- Achieves **98%+ test accuracy** on tiled digit sequences

---

## Accomplishments

- Built an end-to-end Transformer for image-to-sequence generation
- Supports **blank tiles** and variable-length outputs
- Achieved **robust generalisation** on unseen test tiles
- Clean modular code using PyTorch and TorchVision
- Integrated **greedy decoding**, visualisation, and model saving
- Logging via **Weights & Biases (wandb)**

---

## Tools and Libraries Used

- **Python 3.10+**
- **PyTorch**
- **TorchVision**
- **Matplotlib**
- **TQDM**
- **Weights & Biases** (`wandb`)
- **Hugging Face Datasets** (for loading MNIST)

---

## Prerequisites

Ensure you have a working Python environment (preferably Conda).  
Install the dependencies:

```bash
pip install torch torchvision matplotlib tqdm wandb datasets
```

Or use the provided `environment.yml`

---

## How to Run

**Train the model:**
```bash
python training.py
```

**Evaluate the model and save predictions:**
```bash
python evaluate.py
```

Saved predictions will appear in the `predictions/` folder.

---

## Project Structure

```
.
├── model.py              # All model components: encoder, decoder, attention
├── mnist_generator.py    # Custom dataset with tiling and blanks (scattered for future      exploration)
├── training.py           # Training loop with logging & checkpointing
├── evaluate.py           # Evaluation + greedy decoding & visualisation
├── predictions/          # Saved PNGs of model predictions
├── model/...             # Model checkpoints by wandb run name
```

---

## Training Progress (Logged with wandb)

Below are example visualisations of the training performance:

### 🔹 Train Loss | Test Loss | Test accuracy
![](logs.png)

These charts reflect the model's steady improvement across 8 epochs of training.

---

## Sample Model Prediction

Below is a sample output showing how the model predicts a digit sequence from a 2×2 tiled MNIST image:

![Sample Prediction](model/predictions/sample_0.png)

---
