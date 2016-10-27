import tensorflow as tf
import numpy as np
import math
import textloader
import time
import sys

def weight(shape):
    stddev = 1.0 / math.sqrt(shape[0])
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def sampleBatch(data, steps, batchSize):
    indices = np.arange(steps).reshape([-1, steps]).repeat(batchSize, axis=0)
    indices += np.random.randint(0, data.shape[0] - steps - 1, (batchSize, 1)).repeat(steps, axis=1)
    return data[indices], data[indices+1]

def softmax(x, temperature=1.0):
    ex = np.exp(x / temperature)
    return ex / ex.sum()


np.random.seed(int(time.time()))

batchSize = 64
seqLength = 64
lstmSize = 256
lstmLayers = 2
numChars = len(textloader.acceptedChars())

sess = tf.Session()

inputData = tf.placeholder(tf.float32, shape=[batchSize, seqLength, numChars])
targetData = tf.placeholder(tf.float32, shape=[batchSize, seqLength, numChars])

lstmCell = tf.nn.rnn_cell.GRUCell(lstmSize)
stackedLstm = tf.nn.rnn_cell.MultiRNNCell([lstmCell] * lstmLayers)

initialState = stackedLstm.zero_state(batchSize, tf.float32)

embedW = weight([numChars, lstmSize])
embedB = bias([lstmSize])

W1 = weight([lstmSize, lstmSize])
B1 = bias([lstmSize])

outputW = weight([lstmSize, numChars])
outputB = bias([numChars])

probabilities = []
states = []

loss = 0.0
inState = initialState

with tf.variable_scope("RNN"):
    for i in range(seqLength):
        if i > 0:
            tf.get_variable_scope().reuse_variables()

        embedA = tf.nn.elu(tf.matmul(inputData[:, i], embedW) + embedB)

        lstmOut, outState = stackedLstm(embedA, inState)

        preOutput = tf.nn.elu(tf.matmul(lstmOut, W1) + B1)
        logits = tf.matmul(preOutput, outputW) + outputB

        probabilities.append(logits)
        states.append(outState)

        inState = outState
        loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, targetData[:,i,:]))

trainStep =  tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.initialize_all_variables())

data = textloader.load("data/shakespeare.txt")
print(data.shape)
for i in xrange(100000):
    id, td = sampleBatch(data, seqLength, batchSize)

    feedDict = {
        inputData : id,
        targetData : td,
    }

    _, loss_value = sess.run([trainStep, loss], feed_dict=feedDict)
    print("iter: %d = %f" % (i, loss_value))


seed = np.zeros((batchSize, seqLength, numChars))
seed[0, 0] = textloader.charVector('t', numChars)

state = sess.run([initialState]) # tf.convert_to_tensor(initialState).eval(sess)

for i in xrange(10000):
    feedDict = {
        inputData: seed,
        initialState: state
    }

    output, nextState = sess.run((probabilities[0], states[0]), feed_dict=feedDict)

    r = np.random.choice(np.arange(output[0].shape[0]), 1, p=softmax(output[0], 0.7))[0]
    sys.stdout.write(textloader.acceptedChars()[r])

    # print(output[0])
    # print(nextState)
    # print(output[0].shape)
    # print(seed[:,0,:].shape)

    thisChar = np.zeros(numChars)
    thisChar[r] = 1.0

    seed[0,0] = thisChar
    state = nextState

sys.stdout.write('\n')
sys.stdout.flush()
print("hello world")
