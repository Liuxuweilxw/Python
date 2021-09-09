import numpy as np
a = 1.0
w = 0.6
b = 0.9
output = 1/(1+np.exp(-(a*w+b)))
print(output)