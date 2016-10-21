import tensorflow as tf
import numpy as np
import math
import textloader


def weight(shape):
    stddev = 1.0 / math.sqrt(shape[0])
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

def sampleBatch(data, steps, batchSize):
    indices = np.arange(steps).reshape([-1, steps]).repeat(batchSize, axis=0)
    indices += np.random.randint(0, data.shape[1] - steps - 1, (batchSize, 1)).repeat(steps, axis=1)
    return data[indices], data[indices+1]

batchSize = 16
seqLength = 24
lstmSize = 32
numChars = len(textloader.acceptedChars())

print("wooo:")
print(textloader.charVector('c', numChars))

sess = tf.Session()

inputData = tf.placeholder(tf.float32, shape=[batchSize, seqLength, numChars])
targetData = tf.placeholder(tf.float32, shape=[batchSize, seqLength, numChars])

lstmCell = tf.nn.rnn_cell.BasicLSTMCell(lstmSize)
initialState = lstmCell.zero_state(batchSize, tf.float32)

lstmOuts, _ = tf.nn.dynamic_rnn(lstmCell, inputData, initial_state=initialState, dtype=tf.float32)

outputW = weight([lstmSize, numChars])
outputB = bias([numChars])

probabilities = []
loss = 0.0

for i in range(seqLength):
    logits = tf.matmul(lstmOuts[:, i, :], outputW) + outputB
    # probabilities.append(tf.nn.softmax(logits))
    probabilities.append(logits)
    loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, targetData[:,i,:]))

trainStep =  tf.train.AdamOptimizer().minimize(loss)
sess.run(tf.initialize_all_variables())

data = textloader.load("data/shakespeare.txt")
for i in xrange(1000):
    id, td = sampleBatch(data, seqLength, batchSize)

    feedDict = {
        inputData : id,
        targetData : td,
    }

    _, loss_value = sess.run([trainStep, loss], feed_dict=feedDict)
    print("iter: %d = %f" % (i, loss_value))


print("hello world")
