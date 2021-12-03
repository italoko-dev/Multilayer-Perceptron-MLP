import numpy as np
from numpy.core.numeric import isclose

class Mlp:
  def __init__(self):
      self.parameters = {}
      self.eta = 1

  #Inicializa os pesos (valores aleatórios)
  def initialize_parameters(self):
    parameters = {
      "W1": np.random.rand(2,2),
      "W2": np.random.rand(2,2)
    }
    self.parameters = parameters

  def sigmoid(self,x):
    return 1/(1 + np.exp(-x))

  #Derivate
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
    self.parameters["W2"] = self.parameters["W2"][0,:] + self.eta * errors["gradient_s1"] * i_results["iS1"]
    self.parameters["W2"] = self.parameters["W2"][1,:] + self.eta * errors["gradient_s2"] * i_results["iS2"]
    
    self.parameters["W1"] = self.parameters["W1"][0,:] + self.eta * errors["gradient_c"] * i_results["iC"]
    self.parameters["W1"] = self.parameters["W1"][1,:] + self.eta * errors["gradient_d"] * i_results["iD"]

  def calc_error_network(s_errors):
    return 0.5 * (np.sum(s_errors))**2


  def backpropagation():
    ##Implementar

"""
pesos = np.array([
        #w1
        [0.5, 1.0],  # [(a,c) , (a,d)]
        [-0.7, 1.2] # [(b,c) , (b,d)]  
        ])
"""

#input = np.array([1,0])
#print(pesos[:,1])


#

mlp = Mlp()
params = mlp.initialize_parameters()
#print(params["W1"])
#print("----------------------")
#print(params["W1"][0,:])
#print(params["W1"][1,:])

#print(len(params["W1"][1]))

expected = np.array(
  [
    [1,0],
    [0,1]
  ])
print(expected[1,1])

"""
netC = np.sum(np.multiply(input,params["W1"][0,:]))
iC = mlp.sigmoid(netC)
netD = np.sum(np.multiply(input,params["W1"][1,:]))
iD = mlp.sigmoid(netD)

netS1 = np.sum(np.multiply(netC,params["W2"][0,:]))
iS1 = mlp.sigmoid(netS1)
netS2 = np.sum(np.multiply(netD,params["W2"][0,:]))
iS2 = mlp.sigmoid(netS2)

print (f"NetC: {netC}")
print (f"NetD: {netD}")
print (f"NetS1: {netS1}")
print (f"NetS2: {netS2}")

print (f"iC: {iC}")
print (f"iD: {iD}")
print (f"iS1: {iS1}")
print (f"iS2: {iS2}")

"""
#test = np.sum(np.multiply(entrada,pesos[:1])) 

#print(entrada) 
#print(oculta)
#print(netC)
#print(test)

#1 - Aplica as entradas X1, X2, ..., Xn 
#2 - Calcula os nets da camada oculta
#3 – Aplica função de transferência na camada de saída
#4 – Calcula os nets da camada oculta
#5 - Calcula as saídas Ok da camada de saída
#6 - Calcula os erros da camada de saída
#7 - Calculam os erros da camada oculta
#8 - Atualiza os pesos da camada de saída
#9 - Atualiza os pesos da camada oculta
#10 - Calcula o erro da rede fazendo
