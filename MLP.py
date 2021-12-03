import numpy as np

#Inicializa os pesos (valores aleatórios)
def initialize_parameters(layer_dims):
    parameters = {}
    for i in range(1,len(layer_dims)):
        parameters['W' + str(i)] = np.random.rand(layer_dims[i],layer_dims[i])
        #parameters['b' + str(i)] = np.random.rand(layer_dims[i],layer_dims[i]) 
    return parameters


def sigmoid(x):
  return 1/(1 + np.exp(-x))

#Derivate
def de_sigmoid(x):
    f = sigmoid(x)
    return f * (1-f)

def net(input,weights):
    return  np.sum(np.multiply(input,weights))

def l_layer_forwardProp(input, parameters, layer_dims):
  i = input
 
  for layer in range(1,layer_dims[]):
    i_prev = i
    i = sigmoid(net(i_prev,parameters['W' + str(layer)]))


  return A




   





def teste(layer_dims):
    print (initialize_parameters(layer_dims))


#teste([2,2,2])

params = initialize_parameters([2,2,2])


entrada = np.array([1,0])
oculta = params["W1"]

netC = np.sum(np.multiply(entrada,oculta))
netD = np.sum(np.multiply(entrada,oculta))



print(entrada) 
print(oculta)
print(netC)
print(len(params))

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
