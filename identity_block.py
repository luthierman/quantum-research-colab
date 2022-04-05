import pennylane as qml
from pennylane import numpy as np 
import random
L = 2
M = 2
n = 4
dev = qml.device("default.qubit", wires=n)
np.random.seed(1)
def initialize(n,L,M):
  parameters = np.zeros((M,2*L,n))
  
  for m in range(M):
    stack = []
    for l in range(L):
      for i in range(n):
        theta = np.random.uniform(0,2*np.pi)
        stack.append(theta)
        parameters[m,l,i] = theta
    for l in range(L,2*L):
      for i in range(n):
        parameters[m,l,n-i-1] = stack.pop()
  return qml.math.concatenate(parameters)

def block(parameters,n, L, M):
  U_m = []
  ops = [qml.RX, qml.RY, qml.RZ]
  for l in range(L):
    for i in range(n):
      U = random.choice(ops)
      U_m.append(U)
      U(parameters[l,i], wires=i)
    for i in range(n-1):
      qml.CZ(wires=[(i)%n,(i+1)%n])
  for l in range(L, 2*L):
    for i in range(n-1):
      qml.CZ(wires=[n-i-2,n-i-1])
    for i in range(n):
      U = U_m.pop()
      U(parameters[l,n-i-1], wires=n-i-1).inv()
@qml.qnode(dev)

def circuit(inputs, weights):
  L = 2
  M = 2
  n = 4

  qml.AngleEmbedding(inputs*(np.pi/4), 
                     wires = range(n))
  for i in range(0,2*M*L,2*L):
    block(weights[i:i+2*L],n, L, M)
  
  return [qml.expval(qml.PauliZ(i)) for i in range(n)]
thetas = initialize(n,L,M)
print(qml.draw(circuit)(np.array([0.0,0.0,0.0,0.0]), thetas))
weight_shapes = {"weights": thetas.shape}
qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)
model = torch.nn.Sequential(qlayer)
new_state_dict = model.state_dict()
new_state_dict['0.weights'] = torch.tensor(thetas)
model.load_state_dict(new_state_dict)