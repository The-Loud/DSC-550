import numpy as np
from matplotlib import pyplot as plt

# lots of variables
w = np.array([2, 1])
b = np.array([2, -1])
# b = b.reshape(-1, 1)
p = np.linspace(-3, 3, 4)


# Create a neuron
class Neuron():
    def __init__(self, w, x, b):
        self.w = w
        self.x = x
        self.b = b

    def comb(self):
        '''
        returns a linear combination of variables to be used within an
        activation function.
        '''
        return self.w * self.x + self.b

    def activate(self, n, a="s"):
        '''
        activation portion of the transfer function.
        satlin and linear are both represented in a single nested if
        statement
        '''
        #n = self.comb(self.w, self.x, self.b)
        if a == "s":
            if n < 0:
                return 0
            elif 0 <= n and n <= 1:
                return n
            else:
                return 1
        else:
            return n  # for the linear activation


# main function

y_out = []

for i in p:
    n = Neuron(w, i, b)
    out = n.comb()
for j in out:
    a = n.activate(j)
    y_out.append(a)
print(y_out)
