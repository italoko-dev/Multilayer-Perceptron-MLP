import numpy as np
import matplotlib.pyplot

class Mlp:
  def __init__(self):
      self.parameters = {}
      self.eta = 1

  def initialize_parameters(self):
    parameters = {
      "W1": np.random.rand(2,2),
      "W2": np.random.rand(2,2)
    }
    self.parameters = parameters

  def sigmoid(self,x):
    return 1/(1 + np.exp(-x))
  
  def de_sigmoid(self,x):
    f = self.sigmoid(x)
    return f * (1-f)

  def net(self,input,weights):
    return  np.sum(np.multiply(input,weights))

  def forward(self,input):
    i_result = {
      "iC" : self.sigmoid(np.sum(np.multiply(input,self.parameters["W1"][0,:]))),
      "iD" : self.sigmoid(np.sum(np.multiply(input,self.parameters["W1"][1,:])))}

    i_result["iS1" ] = self.sigmoid(np.sum(np.multiply(i_result["iC"],self.parameters["W2"][0,:])))
    i_result["iS2" ] = self.sigmoid(np.sum(np.multiply(i_result["iD"],self.parameters["W2"][1,:])))
    
    return i_result

  def calc_error(self,i_result,y,expected):
    error_s1 = (expected[y,0] - i_result["iS1"]) * self.de_sigmoid(i_result["iS1"])
    error_s2 = (expected[y,1] - i_result["iS2"]) * self.de_sigmoid(i_result["iS2"])

    gradient_errors = {
      "gradient_s1" : error_s1,
      "gradient_s2" : error_s2,
      "gradient_c"  : error_s1 * self.parameters["W2"][:,0] + error_s2 * self.parameters["W2"][:,0],
      "gradient_d"  : error_s1 * self.parameters["W2"][:,1] + error_s2 * self.parameters["W2"][:,1]       
    }  
    return gradient_errors

  def setting_weights(self,errors,i_results):
    self.parameters["W2"][0,:] = self.parameters["W2"][0,:] + self.eta * errors["gradient_s1"] * i_results["iS1"]
    self.parameters["W2"][1,:] = self.parameters["W2"][1,:] + self.eta * errors["gradient_s2"] * i_results["iS2"]
    
    self.parameters["W1"][0,:] = self.parameters["W1"][0,:] + self.eta * errors["gradient_c"] * i_results["iC"]
    self.parameters["W1"][1,:] = self.parameters["W1"][1,:] + self.eta * errors["gradient_d"] * i_results["iD"]

  def calc_error_network(self,s_errors):
    return 0.5 * (np.sum(s_errors))**2

  def backpropagation(self,i_result,y,expected):
    errors = self.calc_error(i_result,y,expected)
    s_errors = np.array([errors["gradient_s1"], errors["gradient_s2"]])
    self.setting_weights(errors,i_result)
    return s_errors

  def train(self,input,expected,max_epoch,limiar):
    h_erros = []
    self.initialize_parameters()
    for epoch in range(max_epoch):
      for i in input:
        x = i[:len(i[:])-1]
        y = i[-1]
        i_result = self.forward(x)
        s_errors = self.backpropagation(i_result,y,expected)
        error_network =  self.calc_error_network(s_errors)
        h_erros.append(error_network)
        if error_network > limiar:
          return h_erros
    return h_erros
  
  def print_graph(self,errors_network):
    matplotlib.pyplot.ylim(0, max(errors_network))
    matplotlib.pyplot.xlim(0, len(errors_network))

    matplotlib.pyplot.title('Aprendizado da rede')
    matplotlib.pyplot.xlabel('Iteração')
    matplotlib.pyplot.ylabel('Erro da rede')

    matplotlib.pyplot.plot(list(range(len(errors_network))) , errors_network)
    matplotlib.pyplot.show()

  def prediction(self,input):
    result = self.forward(input)
    map = {"iS1":"classe 0", "iS2":"classe 1"} 
    if result["iS1"] > result["iS2"]:
      print(f'Entrada:{input} Pertente à: {map["iS1"]}')
    else:
      print(f'Entrada:{input} Pertente à: {map["iS2"]}')  

#-----------------------------------------------
#Inputs for training 
'''
#AND
input_train = np.array(
    [
        [0,0,0],
        [0,1,0],
        [1,0,0],
        [1,1,1]          
    ])

#OR
input_train = np.array(
    [
        [0,0,0],
        [0,1,1],
        [1,0,1],
        [1,1,1]          
    ])

'''
#XOR
input_train = np.array(
[
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]          
])
'''
#XNOR
input_train = np.array(
[
    [0,0,1],
    [0,1,0],
    [1,0,0],
    [1,1,1]          
])
'''

mlp = Mlp()
expected = np.array([[1,0],[0,1]])
mlp.print_graph(mlp.train(input_train,expected,max_epoch=25,limiar=1))

#Input for prediction 
input_prediction = [1,1]
mlp.prediction(input_prediction)
input("Aperte qlq tecla para sair..")
