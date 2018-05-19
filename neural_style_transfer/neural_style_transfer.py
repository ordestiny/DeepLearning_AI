import os
import sys
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from  matplotlib.pyplot import imshow

from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf


def compute_content_cost(a_C,a_G):
    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    a_C_unrolled =  tf.reshape(a_C,[n_H * n_W,n_C])
    a_G_unrolled =  tf.reshape(a_G,[n_H * n_W,n_C])

    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled))) / (4 * n_H * n_W * n_C)
    return J_content

#  GA = A * A.T
def gram_matrix(A):
    GA = tf.matmul(A,tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S,a_G):

    m,n_H,n_W,n_C = a_G.get_shape().as_list()

    a_S = tf.reshape(a_S,[n_H * n_W,n_C])
    a_G = tf.reshape(a_G,[n_H * n_W,n_C])

    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))

    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS,GG))) / (4 * tf.square(tf.to_float(n_C)) * tf.square(tf.to_float(n_H * n_W)))
    return  J_style_layer



def compute_style_cost(model,STYLE_LAYERS,sess):
    J_style = 0.0
    for layer_name,coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out

        J_style_layer =  compute_layer_style_cost(a_S,a_G)
        J_style += coeff * J_style_layer
    return J_style


def total_cost(J_content,J_style,alpha = 10,beta = 40):
    J = alpha * J_content + beta * J_style
    return J

def model_nn(sess, input_image, num_iterations = 200):

    # Initialize global variables (you need to run the session on the initializer)
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model. Use assign().
    sess.run(model["input"].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model["input"])

        # Print every 20 iteration.
        if i%100 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            # save current generated image in the "/output" directory
            save_image("output/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image('output/generated_image.jpg', generated_image)

    return generated_image



if __name__ == '__main__':

    # model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")
    # print(model)

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    content_image = scipy.misc.imread("images/You.jpg")
    content_image = reshape_and_normalize_image(content_image)

    style_image = scipy.misc.imread("images/CXstyle.jpg")
    style_image = reshape_and_normalize_image(style_image)

    generated_image = generate_noise_image(content_image)
    # load the VGG16 model
    model = load_vgg_model("pretrained_model/imagenet-vgg-verydeep-19.mat")

    # Assign the content image to be the input of the VGG model.
    sess.run(model['input'].assign(content_image))

    # Select the output tensor of layer conv4_2
    out = model['conv4_2']

    # Set a_C to be the hidden layer activation from the layer we have selected
    a_C = sess.run(out)

    # Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2']
    # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
    # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
    a_G = out

    # Compute the content cost
    J_content = compute_content_cost(a_C, a_G)
    # Assign the input of the model to be the "style" image
    sess.run(model['input'].assign(style_image))
    STYLE_LAYERS = [("conv1_1", 0.2), ("conv2_1", 0.2), ("conv3_1", 0.2), ("conv4_1", 0.2), ("conv5_1", 0.2)]
    # Compute the style cost
    J_style = compute_style_cost(model, STYLE_LAYERS,sess)
    J = total_cost(J_content, J_style, alpha=10, beta=40)

    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(J)

    model_nn(sess, generated_image,num_iterations=1000)