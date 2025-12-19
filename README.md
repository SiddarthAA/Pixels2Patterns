<div align="center">

# 🧠 Pixels2Patterns  
### A Hands-On Workshop on Convolutional Neural Networks

**Organized by NeuralHive**  
**PES University – Electronic City Campus**

*Where raw pixels evolve into meaningful patterns, and patterns into intelligence.*

</div>

---

## 📌 About the Workshop

**Pixels2Patterns** is a hands-on, beginner-friendly workshop designed to build strong intuition and practical competence in **Convolutional Neural Networks (CNNs)** using **PyTorch**.

Rather than treating CNNs as black boxes, this workshop walks participants through the complete lifecycle of a vision model — from loading raw image data to training, evaluating, and visualizing what the network actually learns.

The notebook is structured for clarity, experimentation, and exploration, making it suitable for both classroom and self-study use.

---

## 🎯 Learning Outcomes

By the end of this workshop, participants will be able to:

- Design and implement a CNN from scratch using PyTorch  
- Understand the role of:
  - Convolutional layers  
  - Pooling layers  
  - Fully connected layers  
  - Output logits and loss functions  
- Train neural networks using backpropagation and modern optimizers  
- Evaluate model performance on unseen test data  
- Visualize training dynamics and internal feature representations  
- Confidently experiment with architectures and hyperparameters  

---

## 🧱 Notebook Structure

### ⚙️ 1. Environment Setup
- PyTorch and torchvision imports  
- Automatic CPU / GPU detection  
- Reproducibility via fixed random seeds  
- Utility functions for image visualization  

---

### 🗂️ 2. Dataset Loading & Exploration
- MNIST handwritten digits dataset  
- Image preprocessing using `transforms.ToTensor()`  
- Efficient batching with `DataLoader`  
- Visual inspection of sample digits  

---

### 🧠 3. CNN Architecture

The workshop uses a clean and readable **Sequential CNN architecture**, designed to be easy to understand and modify.

#### Model Architecture (PyTorch)

```python
class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            # Convolutional Feature Extraction
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),

            # Spatial Downsampling
            nn.MaxPool2d(2, 2),

            # Flatten for Fully Connected Layers
            nn.Flatten(),

            # Classification Head
            nn.Linear(4096, 1024),
            nn.ReLU(),

            nn.Linear(1024, 10)  # Output logits for digits 0–9
        )

    def forward(self, x):
        return self.stack(x)
````

**Design highlights:**

* Progressive feature extraction from low-level edges to high-level patterns
* ReLU activations for non-linearity
* MaxPooling for spatial compression
* Fully connected layers for final classification

This structure encourages experimentation with depth, width, kernel sizes, and regularization.

---

### 📉 4. Loss Function & Optimizer

* **Loss:** `CrossEntropyLoss` for multi-class classification
* **Optimizer:** `Adam` for stable and efficient training

Participants are encouraged to experiment with:

* SGD and RMSprop
* Learning rate tuning
* Momentum and regularization

---

### 🔁 5. Training Loop

* Standard PyTorch training pipeline:

  * Forward pass
  * Loss computation
  * Backpropagation
  * Parameter updates
* Epoch-wise loss tracking
* Clean logging for learning progression

---

### 🔍 6. Model Evaluation

* Evaluation on the MNIST test set
* Accuracy computation
* Visualization of true vs predicted labels
* Analysis of generalization performance

---

### 📊 7. Visualizations & Interpretability

* **Training Loss Curve**

  * Observe convergence behavior
* **Feature Map Visualization**

  * Inspect intermediate activations
  * Understand what patterns the CNN learns internally

This section connects CNN theory with interpretability and intuition.

---

## 👨‍🏫 Workshop Mentors

* **Siddartha A Y**
* **Kshirin Shetty**

The mentors guide participants through both conceptual understanding and hands-on implementation, encouraging experimentation and critical thinking.

---

## ✨ Why This Workshop Matters

Convolutional Neural Networks form the backbone of:

* Image classification
* Object detection
* Medical imaging
* Autonomous systems
* Remote sensing and security

This workshop equips learners with **foundational, transferable skills** that scale naturally into advanced computer vision, research, and real-world deployment.

---

## 👥 Intended Audience

* Undergraduate students exploring AI and Computer Vision
* Beginners in Deep Learning using PyTorch
* Workshop and classroom participants
* Anyone seeking an intuitive, hands-on CNN introduction

---

## 🚀 Suggested Extensions

Participants are encouraged to explore further by:

* Adding Dropout or Batch Normalization
* Modifying network depth or width
* Visualizing misclassified samples
* Switching to Fashion-MNIST
* Logging metrics using TensorBoard

---

## 📜 Usage & License

This notebook is intended for educational and workshop use.
Feel free to fork, modify, and adapt it for learning, teaching, or experimentation.

---

<div align="center">

**Pixels2Patterns**

</div>