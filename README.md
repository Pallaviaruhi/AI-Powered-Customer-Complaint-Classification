# AI-Powered-Customer-Complaint-Classification

## Overview
This project focuses on building an AI-driven text classification system that automatically categorizes customer complaints into predefined categories such as mortgage, credit card, money transfers, debt collection, etc. The model is built using PyTorch and employs deep learning techniques to achieve high accuracy in classifying complaints.

## Features
- **Preprocessing:** Tokenization, word-to-index mapping, and padding for uniform input length.
- **Model Architecture:** A convolutional neural network (CNN)-based classifier with embedding layers.
- **Training & Optimization:** Model trained using cross-entropy loss and optimized with Adam.
- **Evaluation Metrics:** Accuracy, Precision, and Recall for multi-class classification.

## Dataset
The dataset consists of customer complaints that have been tokenized and converted into numerical format. It includes:
- `words.json`: Vocabulary dictionary mapping words to indices.
- `text.json`: Tokenized text data.
- `labels.npy`: Complaint category labels.

## Installation
To run this project, install the necessary dependencies using:
```sh
pip install torch torchmetrics nltk numpy pandas scikit-learn
```

## Model Architecture
The model consists of the following layers:
1. **Embedding Layer**: Converts word indices into dense vector representations.
2. **1D Convolutional Layer**: Extracts meaningful patterns from text.
3. **Fully Connected Layer**: Maps extracted features to complaint categories.

## Training
The model is trained for 3 epochs with a batch size of 400. Loss and accuracy are tracked during training to ensure performance improvement.
```python
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
```

## Evaluation
After training, the model is tested using accuracy, precision, and recall metrics:
```python
accuracy_metric = Accuracy(task='multiclass', num_classes=5)
precision_metric = Precision(task='multiclass', num_classes=5, average=None)
recall_metric = Recall(task='multiclass', num_classes=5, average=None)
```

### Final Model Performance
- **Accuracy:** 80%
- **Precision (per class):** [0.69, 0.76, 0.86, 0.81, 0.87]
- **Recall (per class):** [0.75, 0.75, 0.81, 0.81, 0.87]

## Usage
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/customer-complaint-classifier.git
   ```
2. Navigate to the project directory:
   ```sh
   cd customer-complaint-classifier
   ```
3. Run the training script:
   ```sh
   python train.py
   ```

## Future Improvements
- Implementing LSTM or Transformer-based models for better performance.
- Fine-tuning hyperparameters for improved accuracy.
- Deploying as an API for real-time classification.



