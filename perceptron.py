# coding: UTF-8

#
# Perceptron
#
# Lets teach the computer how to understand an "OR" decision
#
# Based on the tutorial by: Danilo Bargen
#

# From Wikipedia:

# In machine learning, the perceptron is an algorithm for supervised learning of
# binary classifiers: functions that can decide whether an input (represented by
# a vector of numbers) belongs to one class or another.
# The algorithm allows for online learning, in that it processes #elements in
# the training set one at a time.

from random import choice
from numpy import array, dot, random

# First we need to represent the unit step function
# where the result of the step will be equal to 0 for x < 0
# equal to and 1 for x >= 0 .
unit_step = lambda x: 0 if x < 0 else 1

# Next we need to map the possible input to the expected output. The first two
# entries of the NumPy array in each tuple are the two input values. The second
# element of the tuple is the expected result.

# And the third entry of the array is a "dummy" input (also called the bias)
# which is needed to move the threshold (also known as the decision boundary)
# up or down as needed by the step function. Its value is always 1, so that
# its influence on the result can be controlled by its weight.

# x y = input values, z = dummy (bias / decision boundary), e = expected
training_data = [
    #       x y z    e
    (array([0,0,1]), 0),
    (array([0,1,1]), 1),
    (array([1,0,1]), 1),
    (array([1,1,1]), 1),
]

# As you can see, this training sequence maps exactly to the definition of the
# OR function:

# A B   A OR B
# 0 0 	  0
# 0 1     1
# 1 0 	  1
# 1 1 	  1

# Next we'll choose three random numbers between 0 and 1 as the initial weights:
w = random.rand(3)

# Sample Values
# w = [ 0.66389729  0.55242224  0.38784826]

# learning rate
eta = 0.2

# learning iterations
n = 100

# Begin the training process
for i in xrange(n):

    # x = array([0,0,1])
    # expected = 0
    x, expected = choice(training_data)

    # Sample values
    # (0 * 0.6) + (0 * 0.5) + (1 * 0.3)

    result = dot(w, x)
    # result = 0.3

    error = expected - unit_step(result)   # lambda x: 0 if x < 0 else 1
    # error  = 0 - 1

    weight = eta * error * x
    # 0.2 * -1 * array([0,0,1])
    # weight = array([-0. , -0. , -0.2])

    w +=weight
    # w = array([ 0.66389729,  0.55242224,  0.18784826])

# And that's already everything we need in order to train the perceptron!
# It has now "learned" to act like a logical OR function:


for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))

# Sample result
# [0 0]: -0.165974062742 -> 0
# [0 1]: 0.595905222132  -> 1
# [1 0]: 0.468941108414  -> 1
# [1 1]: 1.23082039329   -> 1
