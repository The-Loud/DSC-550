import numpy as np
from matplotlib import pyplot as plt

# lots of variables
w1 = np.array([2, 1])
w2 = np.array([1, -1])
b = np.array([2, -1, 0])
p = np.linspace(-3, 3, 10)
p = p.reshape(-1, 1)


# Create a neuron
class Neuron():
    def __init__(self, w, b):
        self.a = 0
        self.w = w
        self.b = b

    def comb(self, x):
        '''
        returns a linear combination of variables to be used within an
        activation function.
        '''
        z = np.dot(self.w, x) + self.b
        return z

    def activate(self, z, a="s"):
        '''
        activation portion of the transfer function.
        satlin and linear are both represented in a single nested if
        statement
        '''
        if a == "s":
            if z < 0:
                return 0
            elif 0 <= z and z <= 1:
                return z
            else:
                return 1
        else:
            return z  # for the linear activation


# main function
# glad somebody already figured out how to build libraries for this
# because this is some *ugly* code.

z1_out = []
z2_out = []
a1_out = []
a2_out = []
z3_out = []
a3_out = []


satnet1 = Neuron(2, 2)
satnet2 = Neuron(1, -1)
linnet = Neuron([1, -1], 0)

for item in p:
    # First layer
    z1 = satnet1.comb(item)
    z1_out.append(z1)
    z2 = satnet2.comb(item)
    z2_out.append(z2)

    a1, a2 = satnet1.activate(z1), satnet2.activate(z2)

    a1_out.append(a1)
    a2_out.append(a2)

    # Second layer
    z3 = a1 * linnet.w[0] + a2 * linnet.w[1] + linnet.b
    z3_out.append(z3)
    out = linnet.activate(z3)

    a3_out.append(out)
    print(out)
# Plot the results

fig = plt.figure()
ax = plt.subplot(111)
ax.plot(z1_out, label='n_1_1')
ax.plot(a1_out, label='a_1_1')
ax.plot(z2_out, label='n_1_2')
ax.plot(a2_out, label='a_1_2')
ax.plot(z3_out, label='n_2_1')
ax.plot(a3_out, label='a_2_1')
ax.plot(p, label='input')
plt.title('Legend inside')
ax.legend()
plt.show()
