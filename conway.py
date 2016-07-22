# coding: utf-8

"""Game of life - Conway.
Based on the tutorial of: Juan Luis Cano <juanlu001@gmail.com>
The tableboard is an NumPy Array, where 0 means dead cell and 1 live.
Matplotlib for the animation
"""

from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import animation


def neighbor(b):

    """Count how many live cells are around the neighborhood"""
    neighborhood = (
        np.roll(np.roll(b, 1, 1), 1, 0) +  # Down-right
        np.roll(b, 1, 0) +  # Down
        np.roll(np.roll(b, -1, 1), 1, 0) +  # Down-Left
        np.roll(b, -1, 1) +  # Left
        np.roll(np.roll(b, -1, 1), -1, 0) +  # Up-Left
        np.roll(b, -1, 0) +  # Up
        np.roll(np.roll(b, 1, 1), -1, 0) +  # Up-Right
        np.roll(b, 1, 1)  # Right
    )

    return neighborhood


def step(b):
    """Step by the game of life."""
    # Lets check the neighbor for the current cells
    v = neighbor(b)
    # First step
    # >>> neighborhood
    # array([[1, 2, 3, 2, 1, 0, 0, 0],
    #        [2, 2, 3, 1, 1, 0, 0, 0],
    #        [2, 3, 5, 3, 1, 0, 0, 0],
    #        [1, 2, 1, 1, 0, 0, 0, 0],
    #        [0, 1, 1, 1, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0],
    #        [0, 0, 0, 0, 0, 0, 0, 0]])

    # Then just make a copy of the current set
    buffer_b = b.copy()

    # Lets check the size of our matrix, shape[0] = N
    for i in range(buffer_b.shape[0]):
        #  shape[1] = M
        for j in range(buffer_b.shape[1]):

            # If around the current position there is 3 live cells
            # A new cell can be born here

            # If there are 2 live cells, and also the current position
            # is a live cell, this cell will survive the next gen

            if v[i, j] == 3 or (v[i, j] == 2 and buffer_b[i, j]):
                buffer_b[i, j] = 1
            else:
                buffer_b[i, j] = 0

                # If there are more of 3 cells
                #    Will die on sobrepoblation
                # If there are less of two cells
                #    Will die of lonelyness

    return buffer_b

# ---------------------------------------------------------------------------- #

# Params of the problem
GENERATIONS = 50
N = 8
M = 8

# Build the board
board = np.zeros((N, M), dtype=int)

# >>> board
# array([[0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0]])

# Lets add a first gen
board[1, 1:4] = 1
board[2, 1] = 1
board[3, 2] = 1

# >>> board
# array([[0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 1, 1, 1, 0, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0, 0, 0],
#        [0, 0, 1, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0],
#        [0, 0, 0, 0, 0, 0, 0, 0]])

# ---------------------------------------------------------------------------- #

# Matplot fun: Create the figure
fig = plt.figure(figsize=(4, 4))

ax = fig.add_subplot(111)
ax.axis('off')

b = board
imag = ax.imshow(b, interpolation="none", cmap=cm.gray_r)

# Lets create the animation of each step, this is the place where the game
# starts. Each Gen will be a frame of our final animation.
def animate(i):
    global b
    b = step(b)
    imag.set_data(b)


# We need FFMPEG or MENCODER
anim = animation.FuncAnimation(fig, animate, frames=GENERATIONS)
anim.save('gameoflife.mp4', fps=5)
