#!/usr/local/bin/python3
#
# Authors: knikharg, vrmath, anipatil
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio
import random
import copy

# calculate "Edge strength map" of an image
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))
    filtered_y = zeros(grayscale.shape)
    filtered_y_vertical = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    filters.sobel(grayscale, -1, filtered_y_vertical)
    return sqrt(filtered_y**2)
    #return sqrt(((filtered_y + filtered_y_vertical)/2)**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels

# maximum pixel per column
def bayes_net(norm_edge_strength):
    return argmax(norm_edge_strength, axis=0)

# tried gaussian normal distribution for transition probabilty
def gaussian(x, mu):
    sigma = 1
    num = -1* ((x-mu)**2)
    den = (2*(sigma**2))
    expo = exp(num / den)
    expo /=sum(expo)
    return expo

# transition probabilty with uses the pixel intensities, the probabilty of being in the current row given the previous row. 
# some elemantary probability added to 5 pixels above and below the previous pixel on the ridge 
def distance(prev_row, col):
    temp = col/sum(col)
    temp[prev_row-5:prev_row+6] = 0.2*col[prev_row-5:prev_row+6]
    dist = [abs(prev_row - row) for row in range(len(col))]
    max_dist = max(dist)
    dist = abs(asarray(dist) -  max_dist)
    temp *= dist
    temp = [t/sum(temp) for t in temp]
    return asarray(reshape(temp, (-1,1)), dtype='float32')


def emission_probability(col):
     e = [c/sum(col) for c in col]
     return asarray(e).reshape(len(e), 1)


def viterbi(norm_edge_strength):
    vt1 = [0]*norm_edge_strength.shape[1]
    for col in range(initial_col,norm_edge_strength.shape[1]):
        if col == initial_col:
            vt1[initial_col] = initial_row
        else:
            #pij = gaussian((norm_edge_strength[:, col].reshape(norm_edge_strength.shape[0], 1)),(norm_edge_strength[vt1[col - 1]][col - 1]))
            pij = distance(vt1[col-1], norm_edge_strength[:,col])
            vt = norm_edge_strength[vt1[col - 1], col-1]/sum(norm_edge_strength[:,col-1])
            vt1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vt*pij)

    for col in range(initial_col - 1, -1, -1):
        #pij = gaussian((norm_edge_strength[:, col].reshape(norm_edge_strength.shape[0], 1)),(norm_edge_strength[vt1[col + 1]][col + 1]))
        pij = distance(vt1[col+1],norm_edge_strength[:,col])
        vt = norm_edge_strength[vt1[col + 1], col + 1]/sum(norm_edge_strength[:,col+1])
        vt1[col] = argmax(emission_probability(norm_edge_strength[:,col])*vt*pij)
    return vt1



def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):
        for t in range( int(max(y-int(thickness/2), 0)), int(min(y+int(thickness/2), image.size[1]-1 )) ):
            image.putpixel((x, t), color)
    return image

# main program

(input_filename, gt_row, gt_col) = sys.argv[1:]


# load in image 

input_image = Image.open(input_filename)


map_image = copy.deepcopy(input_image)
simple_image = copy.deepcopy(input_image)
human_image = copy.deepcopy(input_image)

# compute edge strength mask
edge_strength = edge_strength(input_image)


#normalize the image
norm_edge_strength = uint8(255 * edge_strength / (amax(edge_strength)))
imageio.imwrite('edges.jpg', norm_edge_strength)


ridgeLength = bayes_net(norm_edge_strength)
imageio.imwrite("output_simple.jpg", draw_edge(simple_image, ridgeLength, (0, 0, 255), 5))

(initial_row) = argmax(norm_edge_strength[:,0],axis=0)
initial_col = 0
ridgeV = viterbi(norm_edge_strength)
imageio.imwrite("output_map.jpg", draw_edge(map_image, ridgeV, (255, 0, 0), 5))

initial_row, initial_col = int(gt_row), int(gt_col)
ridgeH = viterbi(norm_edge_strength)
imageio.imwrite("output_human.jpg", draw_edge(human_image, ridgeH, (0, 255, 0), 5))
