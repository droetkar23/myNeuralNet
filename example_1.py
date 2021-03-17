import numpy as np



training_inputs = np.array([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1],])

training_outputs = np.array([[0],
                             [1],
                             [1],
                             [0]])

training_data = [(ti,to) for ti, to in zip(training_inputs,training_outputs)]