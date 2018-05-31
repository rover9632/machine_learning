## Artificial Neural Network
The Network consists of multiple layers, each layer consists of a set of units (neurons). The layer that receives input features as its units is called input layer, the layer that output the final results is called output layer, the layers between the input layer and the output layer is called hidden layers. Adjacent layers are connected, mostly, the units transform the values from the previous layer with a weighted linear summation plus a bias, followed by a activation function, that is the form ![](https://latex.codecogs.com/gif.latex?out%20%3D%20g%28w_1%20x_1%20&plus;%20w_2%20x_2%20&plus;%20...%20&plus;%20w_m%20x_m%20&plus;%20b%29), where ![](https://latex.codecogs.com/gif.latex?%5C%7Bx_1%2Cx_2%2C...%2Cx_m%5C%7D) is the values from the previous layer, ![](https://latex.codecogs.com/gif.latex?%5C%7Bw_1%2Cw_2%2C...%2Cw_m%5C%7D) is the weights, ![](https://latex.codecogs.com/gif.latex?b) is the bias, ![](https://latex.codecogs.com/gif.latex?g%28%5Ccdot%29) is the activation function.

### Activation Function
ReLU, Sigmoid / Logistic, Binary, Tanh, Softplus, Softmax, Maxout, Linear (i.e. identity)

### Backpropagation
**Backpropagation** is a method that calculate the gradient of the objective/loss function to adjust the weights of neurons in the gradient descent optimization process.

### Initializer
Initialize weights, bias, etc.
- **Zeros** :  initialized to 0.
- **Ones** :  initialized to 1.
- **Random Normal** : initialized with a normal distribution.
- **Random Uniform** : initialized with a uniform distribution.
- **Truncated Normal** : initialized with a truncated normal distribution.

## Architectures

### Multi-layer Perceptron (MLP)
MLP is the basic neural network, which adjacent layers are fully connected, that is, between them every neuron in one layer is connected to every neuron in the other layer.

### Convolutional Neural Network (CNN)
CNNs are good at extracting information about spatial correlation, most commonly applied to analyzing visual imagery.

- **Convolutional layer** : apply a convolution operation, that is using filters to extract features.
  - **1D convolution** : the filters are one dimensional
  - **2D convolution** : the filters are two dimensional
- **Pooling layer** : combine the outputs of neuron clusters at prior layer into a single neuron in this layer, to downsampling the features.
  - **max pooling** : uses the maximum value from each of a cluster of neurons at the prior layer.
  - **average pooling** : uses the average value from each of a cluster of neurons at the prior layer.
- **Fully connected layer** : connect every neuron in prior layer to every neuron in this layer.

### Recurrent Neural Network (RNN)
A RNN is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit dynamic temporal behavior for a time sequence. Commonly used in *Natural Language Processing*.

- **Simple RNN**
- **Gated recurrent unit (GRU)** : using a gating mechanism.
- **Long short-term memory (LSTM)** : normally augmented by recurrent gates called "forget" gates to prevents backpropagated errors from vanishing or exploding.
- **Bi-directional RNN** : This is done by concatenating the outputs of two RNNs, one processing the sequence from left to right, the other one from right to left.
- **Deep RNNs** : stack multiple layers of RNNs together.

### Generative Adversarial Network (GAN)
GAN is a class of artificial intelligence algorithms used in unsupervised machine learning, implemented by a system of two neural networks contesting with each other in a zero-sum game framework.
