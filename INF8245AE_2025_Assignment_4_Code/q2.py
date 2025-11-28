import numpy as np

from typing import List
from q1 import Layer, Dense

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

"""## **Question 2: Putting it all together: MLP**

Now, we will put everything together and implement an MLP (multi-layer perceptron) class which is capable enough of stacking multiple layers.
"""

class MLP(Layer):
    """
    Multi-layer perceptron.
    """
    def __init__(self, layers: List[Layer]):
        """
        Initialize the MLP object. The passed list of layers usually
        follows the order: [Dense, Activation, Dense, Activation, ...]
        Parameters:
            layers (list): list of layers of the MLP
        """
        super().__init__()
        self.layers = layers
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the MLP.
        By default, each Dense layer will use the Kaiming initialization.
        Parameters:
            seed (int): seed for random number generation
        """
        for layer in self.layers:
            if isinstance(layer, Dense):
                fan_in = layer.input_size
                fan_out = layer.output_size
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                layer.weights = np.random.uniform(-limit, limit, (fan_in, fan_out))
                layer.bias = np.zeros((1, fan_out))

            
    def forward(self, input):
        """
        Forward pass of the MLP.
        Parameter:
            input (np.ndarray): input of the MLP, shape: (batch_size, input_size)
                                (NOTE: input_size is the size of the input of the first layer)
        Returns:
            output (np.ndarray): output of the MLP, shape: (batch_size, output_size)
                                 (NOTE: output_size is the size of the output of the last layer)
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, output_grad):
        """
        Backward pass of the MLP.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the MLP (dy)
        Returns:
            input_grad (np.ndarray): gradient of the input of the MLP (dx)
        """
        grad = output_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update(self, learning_rate):
        """
        Update the MLP parameters. Normally, this is done by using the
        gradients computed in the backward pass; therefore, .backward() must
        be called before update(). In other words, this function assumes that
        the gradients are already computed via backward pass.
        Parameter:
            learning_rate (float): learning rate used for updating
        """
        # assumes self.backward() function has been called before
        for layer in self.layers:
            if isinstance(layer, Dense):
                layer.update(learning_rate)