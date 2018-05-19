import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf



from deeplearning.con_neural_networks.cnn_utils import *
from deeplearning.con_neural_networks.cnn_tensorflow import *

# print(create_placeholders(64,64,3,6))

# tf.reset_default_graph()
#
# with tf.Session() as sess:
#     np.random.seed(1)
#     X, Y = create_placeholders(64, 64, 3, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3,Y)
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
#     print("Z3 = " + str(a))

