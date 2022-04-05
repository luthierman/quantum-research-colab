import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from pennylane import numpy as np 
import random
import torch


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
thetas = initialize(n,L,M)
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
@qml.qnode(dev, interface='torch')
def circuit(inputs, weights):
  qml.AngleEmbedding(inputs*(np.pi/4), 
                     wires = range(n))
  for i in range(0,2*M*L,2*L):
    block(weights[i:i+2*L],n, L, M)
  
  return [qml.expval(qml.PauliZ(i)) for i in range(n)]

class Quantum_Net(nn.Module):
    def __init__(self, thetas, use_cuda=False):
        super(Quantum_Net, self).__init__()
        if use_cuda and not torch.cuda.is_available():
            raise Exception("Asked for CUDA but GPU not found")
            
        self.use_cuda = use_cuda
        weight_shapes = {"weights": thetas.shape}
        self.circuit = circuit
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes).to('cuda' if use_cuda else 'cpu')
        self.linear1 = nn.Linear(n, 2).to('cuda' if use_cuda else 'cpu')
        nn.init.xavier_normal_(self.linear1.weight)
        self.linear1.bias.data.zero_()
        new_state_dict = self.qlayer.state_dict()
        new_state_dict['weights'] = torch.tensor(thetas)
        self.qlayer.load_state_dict(new_state_dict)

    def forward(self, x):
        x = x.float().to('cuda' if self.use_cuda else 'cpu')
        x.requires_grad = True
        x = F.relu(self.qlayer(x))
        return self.linear1(x)

    def print_circuit(self):
        drawer = qml.draw(self.circuit, show_all_wires=True)(torch.tensor([0,0,0,0]),
                                        thetas)
        print(drawer)

    def print_circuit_mpl(self):
        drawer = qml.draw_mpl(self.circuit, show_all_wires=True)(torch.tensor([0,0,0,0]),
                                        thetas)
        print(drawer)
thetas = initialize(n,L,M)
qnet = Quantum_Net(thetas, True)

