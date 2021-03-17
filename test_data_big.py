import numpy as np



training_inputs = np.array([np.random.randn(100000)])

training_outputs = np.array([[0,0,0,1,0,0,0,0,0,0]])

training_data = [(ti,to) for ti, to in zip(training_inputs,training_outputs)]