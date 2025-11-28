import os
import numpy as np

from typing import List, Tuple, Union, Dict, Callable

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

"""
The following class `Layer` is a super-class from which subsequent layer, activation, and loss classes will be inherited.
Nothing needs to be changed here, but it's good to get familiar with the stucture and the instuctions.
"""

class Layer:
    """
    Base class for all layers.
    """
    def __init__(self):
        """
        Initialize layer parameters (if any) and auxiliary data that is needed for,
        usually, variables like input_size, output_size, are initialized here.
        """

    def init_weights(self):
        """
        Initialize the weights of the layer, if applicable.
        """
        pass

    def forward(self, input):
        """
        Forward pass of the layer.
        
        Hint: you may need to store the input and/or the output for use in the backward pass.

        Parameter:
            input: input data
        Returns:
            output: output of the layer
        """
        pass

    def backward(self, output_grad):
        """
        Backward pass of the layer.

        Each backward call should perform the following two things:

        1) Compute the gradient of the loss w.r.t. the input of this layer, and return it.
        2) Compute and store the gradients of the layer parameters (if any) as attributes of the layer.
           For example, if the layer has weights and bias, you may store their gradients as
           self.weights_grad and self.bias_grad respectively.

        more formally if the loss value is L then output_grad is dL/dy where y is the output of this layer,
        and you need to compute and return dL/dx where x is the input of this layer.

        Parameter:
            output_grad: gradient of the output of the layer (dLdy)
        Returns:
            input_grad: gradient of the input of the layer (dLdx)
        """
        pass

    def update(self, learning_rate):
        """
        Update the layer parameters, if applicable.
        Parameter:
            learning_rate: learning rate used for updating
        """
        pass

    def __call__(self, input, *args, **kwargs):
        """
        Call the forward pass of the layer.
        Parameter:
            input: input data
        Returns:
            output: output of the layer
        """
        return self.forward(input, *args, **kwargs)
    

"""### Q1.1: Layer Class
Follow the instuctions in the comments to implement a fully-connected layer (`Dense`) capable
of working with any generic input and output sizes. It should define its weight and bias matrices inside.
"""
class Dense(Layer):
    """
    Fully connected layer.
    """
    def __init__(
        self,
        input_size,
        output_size,
    ):
        """
        Initialize the layer.
        Parameters:
            input_size (int): input size of the layer
            output_size (int): output size of the layer
            weights (np.ndarray): weights of the layer
            bias (np.ndarray): bias of the layer
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.weights = None  # shape: (input_size, output_size)
        self.bias = None     # shape: (1, output_size)
        self.weights_grad = None  # shape: (input_size, output_size)
        self.bias_grad = None     # shape: (1, output_size)
        self.init_weights()

    def init_weights(self):
        """
        Initializes the weights of the layer.
        By default, the weights and biases are initialized using the
        uniform Xavier initialization.
        Suggested shapes:
            weights (np.ndarray): weights of the layer, shape: (self.input_size, self.output_size)
            bias (np.ndarray): bias of the layer, shape: (1, self.output_size)
        """
        ## RESET GLOBAL SEED
        ## Do not modify this for reproducibility
        np.random.seed(33)
        border_limit = np.sqrt(6.0 / (self.input_size + self.output_size))
        self.weights = np.random.uniform(-border_limit, border_limit, (self.input_size, self.output_size))
        self.bias = np.zeros((1, self.output_size))

    def forward(self, x):
        """
        Forward pass of the layer.
        Parameters:
            x (np.ndarray): input of the layer, shape: (batch_size, self.input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, self.output_size)
        """
        assert self.weights is not None, "Weights must be initialized before forward pass."
        assert self.bias is not None, "Bias must be initialized before forward pass."
        self.x_cache = x
        return x @ self.weights + self.bias

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        
        if gradients are not initialized, initialize them initialize them in this function.
        Compute the gradients w.r.t. weights, bias, and the input, and update the current gradients
        correspondingly. For simplicity of the implementation, overwrite the gradients at each call
        to backward(). i.e. do not accumulate them.

        Parameters:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, self.output_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, self.input_size)
        """
        assert self.weights is not None, "Weights must be initialized before backward pass."
        assert self.bias is not None, "Bias must be initialized before backward pass."
        x = self.x_cache
        self.weights_grad = x.T @ output_grad
        self.bias_grad = np.sum(output_grad, axis=0, keepdims=True)
        return output_grad @ self.weights.T

    def update(self, learning_rate):
        """
        Update the layer parameters. Normally, this is done by using the
        gradients computed in the backward pass; therefore, backward() must
        be called before update(). In other words, this function assumes that
        the gradients are already computed via backward pass.

        This function implements SGD (stochastic gradient descent)
        
        Parameter:
            learning_rate (float): learning rate used for updating
        """
        assert self.weights is not None, "Weights must be initialized before update."
        assert self.bias is not None, "Bias must be initialized before update."
        assert self.weights_grad is not None, "Weights gradients must be computed before update."
        assert self.bias_grad is not None, "Bias gradients must be computed before update."
        self.weights = self.weights - learning_rate * self.weights_grad
        self.bias = self.bias - learning_rate * self.bias_grad

"""
## **Question 1.2: Activation and Loss Layers (30 points)**
"""

class SoftmaxLayer(Layer):
    """
    Softmax layer.
    """
    def forward(self, x):
        """
        Forward pass of the layer.
        The output's sum along the second axis should be 1.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        shifted_inputs = x - np.max(x, axis=1, keepdims=True)
        exponentiated_inputs = np.exp(shifted_inputs)
        normalization_factors = np.sum(exponentiated_inputs, axis=1, keepdims=True)
        self.out = exponentiated_inputs / normalization_factors
        return self.out

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        softmax_outputs = self.out
        batch_size, vector_dim = softmax_outputs.shape
        input_grad = np.zeros_like(output_grad)
        for i in range(batch_size):
            output_vector = softmax_outputs[i].reshape(-1, 1)
            jacobian_mat = np.diagflat(output_vector) - output_vector @ output_vector.T
            input_grad[i] = jacobian_mat @ output_grad[i]
        return input_grad


class TanhLayer(Layer):
    """
    Tanh layer.
    """
    def forward(self, x):
        """
        Forward pass of the layer. Note that tanh is applied element-wise.
        Also you cannot use np.tanh directly.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        exp_x = np.exp(x)
        exp_negative_x = np.exp(-x)
        self.out = (exp_x - exp_negative_x) / (exp_x + exp_negative_x)
        return self.out

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        return output_grad * (1.0 - self.out * self.out)


class ReLULayer(Layer):
    """
    ReLU layer.
    """
    def forward(self, x):
        """
        Forward pass of the layer.
        Parameter:
            x (np.ndarray): input of the layer, shape: (batch_size, input_size)
        Returns:
            output (np.ndarray): output of the layer, shape: (batch_size, input_size)
        """
        self.activation_mask = (x > 0).astype(float)
        return x * self.activation_mask

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the layer (dy), shape: (batch_size, input_size)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, input_size)
        """
        return output_grad * self.activation_mask

"""
## Q1.3: Cross-Entropy Loss Layer

The forward pass again receives the predicted class probabilities (output from a previous activation layer)
and the ground truth labels (target). It computes the cross-entropy loss using the predicted probabilities and the ground truth labels.
The loss function measures the dissimilarity between the predicted probabilities and the actual labels.

In the backward pass, you need to compute the gradient of the loss with respect to the predicted probabilities. This gradient will be used to update the weights and biases of the preceding layers during backpropagation.

The equation for the cross-entropy loss is given by:
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} y_i \log(p_i)
$$
where $y_i$ is the ground truth label, $p_i$ is the predicted probability for class $i$, and $N$ is the batch size.

Remember to handle numerical stability, it is a good practice to add a small value (e.g. $10^{-10}$) to the predicted probabilities before taking the logarithm.
"""

class CrossEntropyLossLayer(Layer):
    """
    Cross entropy loss layer.
    """

    def forward(self, prediction, target):
        """
        Forward pass of the layer.
        Note that prediction input is assumed to be a probability distribution (e.g., softmax output).
        More formally, the last axis of `prediction` is guaranteed to sum to 1.
        Parameters:
            prediction (np.ndarray): prediction of the model, shape: (batch_size, num_classes)
            target (np.ndarray): target, shape: (batch_size,)
        Returns:
            output (float): cross entropy loss, averaged over the batch
        """
        # preconditions:
        assert prediction.ndim == 2, "Prediction must be a 2D array."
        assert target.ndim == 1, "Target must be a 1D array."
        assert prediction.shape[0] == target.shape[0], "Batch size of prediction and target must match."
        assert ((prediction.sum(1) - 1.0) < 1e-6).all(), "Predictions must sum to 1 along the last axis."
        self.prediction_cache = prediction
        self.target_cache = target
        batch_size = prediction.shape[0]
        stable_prediction = prediction + 1e-10
        chosen_class_probabilities = stable_prediction[np.arange(batch_size), target]
        negative_log_likelihood = -np.log(chosen_class_probabilities)
        return np.mean(negative_log_likelihood)

    def backward(self, output_grad):
        """
        Backward pass of the layer.
        Parameter:
            output_grad (float): gradient of the output of the layer (dy)
        Returns:
            input_grad (np.ndarray): gradient of the input of the layer (dx), shape: (batch_size, num_classes)
                in particular, this is the gradient of the loss w.r.t. the `prediction` input of the forward pass.
        """
        prediction = self.prediction_cache
        target = self.target_cache
        batch_size, num_classes = prediction.shape
        input_grad = np.zeros_like(prediction)
        input_grad[np.arange(batch_size), target] = -1.0 / (prediction[np.arange(batch_size), target] + 1e-10)
        return (output_grad / batch_size) * input_grad

