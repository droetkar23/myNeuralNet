import timeit

import cupy as cp
x = cp.array(cp.array([cp.random.randn(100000)]))
x = x.sum()
print(x)
setup = '''
from neural_net_cupy import NeuralNet
import cupy as cp
training_inputs = cp.array([cp.random.randn(100000)])
training_outputs = cp.array([[0,0,0,1,0,0,0,0,0,0]])
training_data = [(ti,to) for ti, to in zip(training_inputs,training_outputs)]
big_net = NeuralNet([100000,1000,100,50,10])
'''
print(timeit.timeit(stmt='for i in range(10): big_net.make_step(batch=training_data, eta=1.0)', setup=setup, number=1000))