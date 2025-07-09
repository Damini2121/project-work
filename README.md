# Pneumonia Detection from Chest X-Rays

[cite_start]This project fine-tunes a pre-trained Inception-V3 model to classify chest X-ray images as 'normal' or 'pneumonia' using the PneumoniaMNIST dataset. [cite: 3, 5]

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-link>
    cd Pneumonia-Detection
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the training and evaluation script:**
    ```bash
    python train_evaluate.py
    ```
    The script will download the dataset, train the model, and print the evaluation metrics.  It will also save a `confusion_matrix.png` image.

## [cite_start]Hyper-parameter Choices [cite: 17]

* **Learning Rate (0.001)**: A common starting point for the Adam optimizer. It's small enough to ensure the model doesn't overshoot the optimal weights during fine-tuning.
* **Batch Size (32)**: This size offers a balance between computational efficiency and the stability of the gradient estimate. It's a standard choice that fits well in memory for most GPUs.
* **Epochs (10)**: Chosen as a sufficient number to allow the model to learn from the data without significant overfitting, especially with data augmentation in place.
