


# Machine Learning Project

## Project Overview
This project involves implementing various components of neural networks from scratch and training them on different datasets using both custom and PyTorch implementations. The project covers fundamental aspects of deep learning, including feed-forward fully-connected networks, convolutional neural networks (CNNs), and training using mini-batch gradient descent.

## Implemented Components

### 1. Feed-Forward Fully-Connected Neural Networks
- Implemented the forward and backward passes for the fully-connected layer.
- Implemented the ReLU activation function.
- Implemented the softmax activation function and derived its Jacobian.
- Implemented the cross-entropy loss function.
- Trained a two-layer fully-connected neural network on the Iris dataset, exploring different hyperparameter configurations (learning rate, hidden layer size).

### 2. Convolutional Neural Networks (CNNs)
- Implemented the forward and backward passes for the convolutional layer, including gradient computations for weights and biases.
- Implemented the forward and backward passes for the pooling layer (both max pooling and average pooling).
- Debugged and validated the implementation using provided test cases and gradient checking.

### 3. Neural Network Package Structure
- Utilized a modular codebase to construct neural networks, including layers, activations, losses, and optimizers.
- Ensured that all layers and components were compatible with mini-batch operations.
- Organized code into classes for layers, activation functions, optimizers, and other components.

### 4. Testing and Debugging
- Used the `unittest` module to test the forward and backward methods of all implemented layers.
- Implemented additional tests to ensure the correctness of gradient computations.
- Debugged gradient implementations using numerical gradient checking in a provided Jupyter notebook.

### 5. Training and Evaluation
- Trained a two-layer neural network on the Iris dataset, achieving optimal performance by tuning hyperparameters.
- Implemented and trained a CNN on the CIFAR-10 dataset using PyTorch, achieving competitive accuracy on the Kaggle competition.
- Visualized training and validation loss/accuracy curves to evaluate model performance.

### 6. PyTorch Implementation
- Implemented and trained a Multi-Layer Perceptron (MLP) on the Fashion MNIST dataset using PyTorch.
- Achieved a validation accuracy of at least 82% by training for at least 8 epochs.
- Submitted test set predictions for the CIFAR-10 dataset to the Kaggle competition, achieving a test accuracy of 74.8%.

## How to Reproduce Results
1. **Set Up Environment**:
   - Ensure you have Python and the necessary libraries installed (`numpy`, `matplotlib`, `torch`, etc.).
   - Install any additional dependencies listed in `requirements.txt`.

2. **Run Local Tests**:
   - Navigate to the project root directory.
   - Run the local tests using `python -m unittest -v` to ensure all components are working correctly.

3. **Train Models**:
   - To train the feed-forward neural network on the Iris dataset, run `python train_ffnn.py`.
   - To train the CNN on the CIFAR-10 dataset, run `python train_conv.py`.

4. **Submit to Kaggle**:
   - Generate predictions using the trained CNN model and submit them to the Kaggle competition.

5. **Evaluate Results**:
   - Review the training and validation curves generated during the training process.
   - Compare the Kaggle competition results to assess model performance.

## Code Structure
- `hw6_release/code/neural_networks/`: Contains implementations for layers, activations, losses, and optimizers.
- `hw6_release/code/tests/`: Contains test cases for verifying the correctness of implemented components.
- `hw6_release/code/train_ffnn.py`: Script for training the feed-forward neural network on the Iris dataset.
- `hw6_release/code/train_conv.py`: Script for training the CNN on the CIFAR-10 dataset.
- `hw6_release/code/check_gradients.ipynb`: Jupyter notebook for debugging gradient computations.


---

