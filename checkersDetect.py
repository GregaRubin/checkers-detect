import math
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib.pyplot import imread
import numpy as np
import cv2 as cv
from skimage.transform import resize
import sys
import json

keras = tf.keras
models = keras.models

def getVirtualBoard(pieces, model, classes):
    #return 8 x 8 array representing current board

    board = np.zeros((8, 8))
    board[:] = -1

    for i in range(0, 8):
        start_pos = 0
        if i % 2 == 0:
            start_pos = 0
        else:
            start_pos = 1

        for j in range(0, 4):
            img = pieces[i*4 + j]
            img = img[np.newaxis, :, :]

            #prediction
            prediction = model.predict(img)
            max = np.amax(prediction)
            """
            plt.title(classes[np.argmax(prediction)] + " " + str(max))
            plt.imshow(img[0, :, :])
            plt.show()
            """
            if max > 0.7:
                board[7 - i, start_pos + j * 2] = np.argmax(prediction)

    return board

def getPieces(img, centers, widths, return_dim):
    #returns list of (piece_dim X piece_dim) images representing possible piece positions on board
    #the pieces are in order from A1 to H8
    img_src = np.copy(img)
    print(img_src.shape)
    print(centers[:].size)

    pieces = []
    for i in range(0, 32):
        center = centers[i]
        width = widths[i]
        dist = math.dist(center, width)
        x_start = round(center[0] - dist)
        x_end = round(center[0] + dist)
        y_start = round(center[1] - dist)
        y_end = round(center[1] + dist)
        crop = img_src[y_start: y_end, x_start:x_end, :]
        crop = resize(crop, (return_dim, return_dim))
        pieces.append(crop)
    return pieces

def getCentersAndWidths(img, width_factor=1.1):
    # destination image size: (dest_dim X dest_dim)
    # 4 keypoints from A1 to A8 counterclockwise are selected
    # returns centers of possible pieces on board and their widths
    # width = cell_size / 2 * width_factor
    # cell_size = dest_dim / 8
    dest_dim = 1000
    img_src = np.copy(img)
    plt.imshow(img_src)
    plt.xlabel("Select board corners (A1 -> H1 -> H8 -> A8)")
    #print(img_src.shape[0], img_src.shape[1])
    corners = plt.ginput(4)

    #for corner in corners:
        #plt.plot(corner[0], corner[1], 'o')
    #plt.show()

    key_points_src = np.array(corners, dtype="float")
    key_points_dest = np.array([(0, 0), (dest_dim - 1, 0), (dest_dim - 1, dest_dim - 1), (0, dest_dim - 1)],
                               dtype="float")
    M, mask = cv.findHomography(key_points_src, key_points_dest)
    dest_img = cv.warpPerspective(src=img_src, M=M, dsize=(dest_dim, dest_dim), flags=cv.INTER_LINEAR)
    #plt.imshow(dest_img)

    half_cell_size = (dest_dim) / 16
    dest_centers = []
    w = - half_cell_size
    h = - half_cell_size

    for i in range(0, 8):
        h += half_cell_size * 2
        for j in range(0, 8):
            w += half_cell_size * 2
            dest_centers.append((w, h))
        w = -half_cell_size

    dest_centers = np.array(dest_centers, dtype="float")
    dest_centers = np.reshape(dest_centers, (-1, 8, 2))
    dest_centers_flat = np.zeros(dtype="float", shape=(8, 4, 2))

    for i in range(0, 8):
        if i % 2 == 0:
            dest_centers_flat[i] = dest_centers[i, 0::2, :]
        else:
            dest_centers_flat[i] = dest_centers[i, 1::2, :]

    dest_centers_flat = np.reshape(dest_centers_flat, newshape=(32, 2))
    dest_widths = np.array([[i[0] + half_cell_size * width_factor, i[1]] for i in dest_centers_flat])

    """
    for c in dest_widths:
        plt.plot(c[0], c[1], "o")
    
    for c in dest_centers_flat:
        plt.plot(c[0], c[1], "o")
    plt.show()
    """

    src_centers = cv.perspectiveTransform(dest_centers_flat[np.newaxis, ...], m=np.linalg.inv(M))
    src_widths = cv.perspectiveTransform(dest_widths[np.newaxis, ...], m=np.linalg.inv(M))
    src_centers = src_centers[0, :, :]
    src_widths = src_widths[0, :, :]

    return (src_centers, src_widths)


def printBoard(board, classes):
    for i in range(0, 8):
        for j in range(0, 8):
            num = board[i, j]
            if num == -1:
                print("X", end=" ")
            else:
                print(int(num), end=" ")
        print()

def createArray(board, classes):
    res = []
    for i in range(0, 8):
        tmp = []
        for j in range(0, 8):
            num = board[i, j]
            if num == -1:
                tmp.append("X")
            else:
                tmp.append(classes[int(num)])
        res.append(tmp)
    
    return res


def main():
    n = len(sys.argv)
    if n != 2:
        sys.stdout.write("error")
        sys.exit(0)

    filePath = sys.argv[1]
    img = imread(filePath)
    model = models.load_model("model_7x7")
    model.summary()
    classes = ["man_white", "man_black", "king_white", "king_black"]

    centers, widths = getCentersAndWidths(img, 1.04)

    """
    plt.imshow(img)
    for w in widths[:, :]:
        plt.plot(w[0], w[1], "gx")
    for c in centers[:, :]:
        plt.plot(c[0], c[1], "bo")
    plt.show()
    """

    pieces = getPieces(img, centers, widths, 64)
    board  = getVirtualBoard(pieces, model, classes)
    printBoard(board, classes)
    jsonStr = createArray(board, classes)
    
    
    with open('board.json', 'w') as f:
        json.dump(jsonStr, f)
    sys.exit(0)




if __name__ == '__main__':
    main()

