import numpy as np 
from numpy import random
import matplotlib.pyplot

def sigmoid(x):
    return 1/(1 + np.exp(-x))
  
def derivative_sigmoid(x):
    f = sigmoid(x)
    return f * (1-f)

def initialize_weights(architecture):
    weights = []
    for l in range(1,len(architecture),1):
        w_layer = np.random.randn(architecture[l],architecture [l-1])
        weights.append(w_layer)
    return weights

class neuralNetwork:

    def __init__(self,architecture,weights,activation="sigmoid"):
        self.architecture = architecture
        self.weights = weights
        if activation == "sigmoid":
            self.activation = sigmoid
            self.derivative_activation = derivative_sigmoid
        elif activation == "sigmoid":
            #Implementar reLU , tanh... 
            self.activation = sigmoid
            self.derivative_activation = derivative_sigmoid


    """def initialize_weights(self):
        weights = []
        for l in range(1,len(self.architecture),1):
            w_layer = np.random.randn(self.architecture[l],self.architecture [l-1])
            weights.append(w_layer)
        return weights"""
    
    def feed_forward(self,X):
        A_layer = list()
        for layer in range(len(self.weights)):
            Z = np.dot(self.weights[layer],X)
            A = self.activation(Z)
            A_layer.append(A)
            X = Z
        return A_layer
    
    def get_output_error(self,A_output,Y):
        error = Y - A_output
        return [error * self.derivative_activation(A_output)]

    def get_hidden_errors(self,A,g_errors):   
        for l in range(len(A) - 2, 0, -1): 
            g_errors.append(g_errors[-1].dot(self.weights[l].T) * self.derivative_activation(A[l]))
        g_errors.reverse()
        return g_errors

    def update_wights(self,A,g_errors,learning_rate):
        for i in range(len(self.weights)):
            layer = np.atleast_2d(A[i])
            grad = np.atleast_2d(g_errors[i])
            self.weights[i] += learning_rate * layer.T.dot(grad)

    def backpropagation(self,A,Y,learning_rate):
        output_error = self.get_output_error(A[-1],Y)
        g_errors = self.get_hidden_errors(A, output_error)
        self.update_wights(A,g_errors,learning_rate)
        return np.sum(np.power(output_error,2) / 2)

    def train(self, data, Y,learning_rate = 0.05,threshold = 0.01, max_epochs = 20000):
        square_error = threshold * 2

        for epoch in range(max_epochs):
            for i in range(len(data)):
                A = self.feed_forward(data[i])
                square_error +=  self.backpropagation(A, Y[i] , learning_rate)

            square_error /= len(data)
            if epoch % 100 == 0:
                print(f"Épocas: {epoch} | Erro quad. médio: {square_error}")
            
            if(square_error < threshold):
                break

    """def print_graph(self,errors_network):
        matplotlib.pyplot.ylim(0, max(errors_network))
        matplotlib.pyplot.xlim(0, len(errors_network))

        matplotlib.pyplot.title('Aprendizado da rede')
        matplotlib.pyplot.xlabel('Iteração')
        matplotlib.pyplot.ylabel('Erro da rede')

        matplotlib.pyplot.plot(list(range(len(errors_network))) , errors_network)
        matplotlib.pyplot.show()  """          

if __name__ == '__main__':
    #XOR
    input_train = np.array(
    [
        [0,0],
        [0,1],
        [1,0],
        [1,1]          
    ])
    expected_train = np.array([0 , 1 , 0 , 1])



    mlp = neuralNetwork(architecture = [2,2,2], 
                        weights = initialize_weights(architecture = [2,2,2]),
                        activation="sigmoid")
    mlp.train(data = input_train,Y = expected_train)
