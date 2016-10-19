import tensorflow as tf
import numpy as np
import math
import string

def acceptedChars():
    r = []
    r += string.lowercase
    r += string.digits
    r += [' ', '.', '!', '?', '\"', '\'', '(', ')', '[', ']', '{', '}', '-',
         '@', '#', '$', '%', '&',  '*',  '<', '>', ':', ';', '/', '\\']
    return r

def acceptedCharsMap(chars):
    r = {}
    for i in xrange(0, len(chars)):
        r[chars[i]] = i
    return r

def normalised(c):
    if c.isspace():
        return ' '
    else:
        return c

def loadInputOutput(inputsPath):
    chars = acceptedChars()
    charsMap = acceptedCharsMap(chars)

    fileChars = []
    with open(inputsPath) as f:
        for line in f:
            line = line.lower()
            for c in line:
                fileChars.append(normalised(c))

    indices = []
    isPrevSpace = False

    for c in fileChars:
        if c.isspace() and isPrevSpace:
            continue

        isPrevSpace = c.isspace()
        if c in charsMap:
            indices.append(charsMap[c])

    numExamples = len(indices)
    exampleDim = len(chars)

    r = np.zeros((numExamples, exampleDim))
    indices = np.array(indices)

    r[np.arange(numExamples), indices] = 1.0
    return r

def weight(shape):
    stddev = 1.0 / math.sqrt(shape[0])
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

print("hello world")
print(len(acceptedChars()))
loadInputOutput("data/shakespeare.txt")
