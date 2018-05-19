import numpy as np
import matplotlib.pyplot as plt

from deeplearning.con_neural_networks.pad.zero_pad import zero_pad

# plt.rcParams["figure.figsize"] = (5.0,4.0)
# plt.rcParams["image.interpolation"] = "nearest"
# plt.rcParams["image.cmap"] = "gray"

x = np.random.randn(4,3,3,2)
x_pad = zero_pad(x,2)
fig,axarr = plt.subplots(1,2)
axarr[0].set_title("x")
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title("x_pad")
axarr[1].imshow(x_pad[0,:,:,0])
plt.show()
