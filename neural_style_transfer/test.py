import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow

plt.subplot(1, 3, 1)
content_image = scipy.misc.imread("images/You.jpg")
imshow(content_image)
plt.subplot(1, 3, 2)
style_image = scipy.misc.imread("images/CXstyle.jpg")
imshow(style_image)
plt.subplot(1, 3, 3)
out_image = scipy.misc.imread("output/200.png")
imshow(out_image)
plt.show()