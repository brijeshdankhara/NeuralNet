# Dankhara, Brijesh
# 1000-127-7373
# 2016-11-02
# Assignment_05

import numpy as np
import theano
import scipy.misc
import os
from random import shuffle
from IPython.display import Image
from IPython.display import IFrame
import pydot_ng as pd
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix
from theano.d3viz import d3viz
# theano.tensor contains most of the 'symbols' as well as the numpy-style operations
import theano.tensor as T
from theano.printing import pydotprint

def load_samples_targets():
    global t, i, samples, targets, s1, t1
    ## LOAD TRAIN SAMPLES AND TARGETS
    content_list = []
    imgvec = []
    dir = "/Users/brijeshdankhara/PycharmProjects/NeuralNet/train/"
    for content in os.listdir(dir):  # "." means current directory
        content_list.append(content)
        img = scipy.misc.imread(dir + content).astype(np.float32)  # read image and convert to float
        img = img.reshape(-1, 1)
        a = img
        imgvec.append(img)
    s = np.reshape(np.array(imgvec), (1000, 3072))
    tarvec = []
    for content in os.listdir("./train/"):
        if content[:1] == '0':
            temptar = [1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '1':
            temptar = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '2':
            temptar = [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '3':
            temptar = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '4':
            temptar = [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '5':
            temptar = [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '6':
            temptar = [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '7':
            temptar = [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1]
            tarvec.append(temptar)
        elif content[:1] == '8':
            temptar = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
            tarvec.append(temptar)
        elif content[:1] == '9':
            temptar = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
            tarvec.append(temptar)
    t = np.array(tarvec)
    list1_shuf = []
    list2_shuf = []
    index_shuf = range(len(s))
    shuffle(index_shuf)
    for i in index_shuf:
        list1_shuf.append(s[i])
        list2_shuf.append(t[i])
    samples = np.array(list1_shuf) / 225
    samples = samples.T / 255
    targets = np.array(list2_shuf)
    targets = targets.T
    ## LOAD TEST SAMPLES
    content_list1 = []
    imgvec1 = []
    dir = "/Users/brijeshdankhara/PycharmProjects/NeuralNet/test/"
    for content in os.listdir(dir):  # "." means current directory
        content_list1.append(content)
        img = scipy.misc.imread(dir + content).astype(np.float32)  # read image and convert to float
        img = img.reshape(-1, 1)
        a = img
        imgvec1.append(img)
    s1 = np.reshape(np.array(imgvec1), (100, 3072))
    tarvec1 = []
    for content in os.listdir("./test/"):
        if content[:1] == '0':
            temptar1 = [1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '1':
            temptar1 = [-1, 1, -1, -1, -1, -1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '2':
            temptar1 = [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '3':
            temptar1 = [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '4':
            temptar1 = [-1, -1, -1, -1, 1, -1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '5':
            temptar1 = [-1, -1, -1, -1, -1, 1, -1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '6':
            temptar1 = [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '7':
            temptar1 = [-1, -1, -1, -1, -1, -1, -1, 1, -1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '8':
            temptar1 = [-1, -1, -1, -1, -1, -1, -1, -1, 1, -1]
            tarvec1.append(temptar1)
        elif content[:1] == '9':
            temptar1 = [-1, -1, -1, -1, -1, -1, -1, -1, -1, 1]
            tarvec1.append(temptar1)
    t1 = np.array(tarvec1)


load_samples_targets()


list1_shuf1 = []
list2_shuf1 = []
index_shuf1 = range(len(s1))
shuffle(index_shuf1)
for i in index_shuf1:
    list1_shuf1.append(s1[i])
    list2_shuf1.append(t1[i])
test_samples = np.array(list1_shuf1)/225
test_samples = test_samples.T/255
test_targets = np.array(list2_shuf1)
test_targets = test_targets.T

## THEANO TRAIN

alpha = 0.0003
lambda_val = 0.5

first_W = theano.shared(np.random.uniform(-0.0001, 0.0001, (3072, 100)), name='Layer1weights')
second_W = theano.shared(np.random.uniform(-0.0001, 0.0001, (100, 10)), name='Layer2weights')

b1 = theano.shared(0.0, name='bias1') # separate bias because we can
b2 = theano.shared(0.0, name='bias2')

# Define some placeholders for our inputs/targets
p = T.dmatrix('samples')
t = T.dmatrix('targets')

# Compute the units outputs
net1 = T.dot(T.transpose(p),first_W) + b1
a1 = T.nnet.relu(net1) # relu
net2 = T.dot(a1,second_W) + b2
a2 = T.nnet.softmax(net2) # softmax

L2_sqr = (
    (first_W ** 2).sum()
    + (second_W ** 2).sum()
)

# Define our performance function
loss = abs(T.mean(T.nnet.categorical_crossentropy(a2, t))) + L2_sqr*lambda_val

# Compute gradients
dW1, dW2, db1, db2 = T.grad(loss, wrt=[first_W, second_W, b1, b2])

# Define a function which, for a given {p, t} pair, computes the output/error and performs a weight update
train = theano.function([p, t], [loss, a2, net2, first_W, second_W],
                        updates=[[first_W, first_W - alpha*dW1],[b1, b1 - alpha*db1],
                                 [second_W, second_W - alpha*dW2],[b2, b2 - alpha*db2]])
plt.title("NNet")
mean_err = 0
mean_list = []
j_list = []
for j in range(100): ## EPOCH
    for i in range(1000): ## number of images in the dataset
        err, output, net2, W1, W2 = train(samples[:,i:i+1], targets[:,i:i+1].T)
        if j % 5 == 0:
            mean_err = err + mean_err
    if j % 5 == 0:
        j_list.append(j)
        mean_list.append(mean_err/((j+1)*1000))

print j_list, mean_list






## THEANO TEST
max_val = T.argmax(a2)
predict = theano.function([p],max_val)


pred_list = []
for i in range(100):
    pred = predict(test_samples[:,i:i+1])
    pred_list.append(pred)

pred_array = np.array(pred_list)
test_target_array = np.argmax(np.array(test_targets),axis=0)
cm = ConfusionMatrix(test_target_array, pred_array)
cm.print_stats()

plt.plot(j_list, mean_list)
plt.show()