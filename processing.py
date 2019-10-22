from config import *
from skimage import data, io, color, exposure, img_as_ubyte
import numpy as np
import pylab


def load_img(filepath):
    return img_as_ubyte(color.rgb2gray(data.load(filepath)))


def c(z_1i, z_2j):
    return (0.5*z_1i - 0.5*z_2j)*(0.5*z_1i - 0.5*z_2j) / (2*VAR)


def make_forward_pass(l_strip, r_strip):
    height, width = len(l_strip) + 1, len(r_strip) + 1
    C, memo = np.zeros((height, width)), np.zeros((height, width))
    for i in range(1, height):
        C[i, 0] = i*OCCLUSION
        memo[i, 0] = 2
    for i in range(1, width):
        C[0, i] = i*OCCLUSION
        memo[0, i] = 3
    for i in range(1, height):
        for j in range(1, width):
            costs = [C[i - 1, j - 1] + c(l_strip[i - 1], r_strip[j - 1]), C[i - 1, j] + OCCLUSION, C[i, j - 1] + OCCLUSION]
            mimimum = min(costs)
            C[i, j] = mimimum
            memo[i, j] = costs.index(mimimum) + 1
    return memo


def reconstruct_optimal_match(i, j, memo, disparity_vector):
    if i == 0 and j == 0:
        return disparity_vector
    elif memo[i][j] == 2:
        disparity_vector.append(0)
        return reconstruct_optimal_match(i - 1, j, memo, disparity_vector)
    elif memo[i][j] == 3:
        return reconstruct_optimal_match(i, j - 1, memo, disparity_vector)
    else:
        disparity_vector.append(abs(i - j))
        return reconstruct_optimal_match(i - 1, j - 1, memo, disparity_vector)


def match_strips(l_strip, r_strip):
    disparity_vector = list()
    memo = make_forward_pass(l_strip, r_strip)
    i, j = len(l_strip), len(r_strip)
    return reconstruct_optimal_match(i, j, memo, disparity_vector)


def disparity(left_img, right_img):
    disparity_map = np.fliplr(np.array([match_strips(left_img[i, ...], right_img[i, ...]) for i in range(left_img.shape[0])]))
    return exposure.rescale_intensity(disparity_map, out_range=("uint8"))


def random_image(shape):
    return np.random.randint(0, 255, shape)


def insert_pattern(background, pattern, location):
    img = background.copy()
    r0, c0 = location
    r1, c1 = r0 + pattern.shape[0], c0 + pattern.shape[1]
    if r1 < background.shape[0] and c1 < background.shape[1]:
        img[r0:r1, c0:c1] = pattern
    return img


if __name__ == "__main__":
    A, B = random_image((512, 512)), random_image((256, 256))
    L, R = insert_pattern(A, B, (128, 132)), insert_pattern(A, B, (128, 124))
    pylab.figure(1)
    io.imshow(L, cmap="gray")
    pylab.figure(2)
    io.imshow(R, cmap="gray")
    pylab.figure(3)
    io.imshow(disparity(L, R), cmap="gray")
    io.show()
