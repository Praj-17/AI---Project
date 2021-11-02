import torch.nn as nn
class NeuralNet(nn.Module):
    """
  Preparing 3 layers of neural networks, 
  There is only one hidden layer created while respecting the complexity of the problem

 
  
  This class is basically taking three types of inputes and fusing them into one
  this is similar to what our brain does , it takes stimulas from sensory organs and then provides an command or an ouput
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) #input layer
        self.l2 = nn.Linear(hidden_size, hidden_size) #single hidden layer
        self.l3 = nn.Linear(hidden_size, num_classes) # Output layer
        self.relu = nn.ReLU()
    def forward(self, x):      #Forward propagation
        out = self.l1(x) 
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out