#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:00:17 2017

@author: ts0
"""

import numpy as np
 
 
def trace_boundary(image, XStart=None, YStart=None,
                MaxLength=np.inf):
    
 
    # check type of input image
    if image.dtype != np.dtype('bool'):
        raise TypeError("Input 'Image' must be a bool")
 
    # scan for starting pixel if none provided
    if XStart is None and YStart is None:
        Indices = np.nonzero(image)
        if Indices[0].size > 0:
            YStart = Indices[0][0]
            XStart = Indices[1][0]
        else:
            X = []
            Y = []
            return X, Y
 
        
    X, Y = moore_neighbor(image, XStart, YStart, MaxLength)
    
 
 
def moore_neighbor(image, XStart, YStart, MaxLength):
    """Boundary tracing of a single object in a binary image
    using the Moore-neighbor algorithm.
 
    Parameters:
    -----------
    Mask : array_like
        A boolean type image where foreground pixels have value 'True', and
        background pixels have value 'False'.
    XStart : int
        Starting horizontal coordinate to begin tracing.
    YStart : int
        Starting vertical coordinate to begin tracing.
    MaxLength : int
        Maximum boundary length to trace before terminating.
 
    Returns:
    --------
    X : 
        Vector of horizontal coordinates of contour pixels.
    Y : 
        Vector of the vertical coordinates of contour pixels.
    """
 
    # initialize outputs
    X = []
    Y = []
 
    # add starting pixel and direction to outputs
    X.append(XStart)
    Y.append(YStart)
 
    # initialize direction
    DX = 1
    DY = 0
 
    # define clockwise ordered indices
    row = [2, 1, 0, 0, 0, 1, 2, 2]
    col = [0, 0, 0, 1, 2, 2, 2, 1]
    dX = [-1, 0, 0, 1, 1, 0, 0, -1]
    dY = [0, -1, -1, 0, 0, 1, 1, 0]
    oX = [-1, -1, -1, 0, 1, 1, 1, 0]
    oY = [1, 0, -1, -1, -1, 0, 1, 1]
   #k = 0
 
    while True:
        #k += 1
        #print(k)
        # rotate template surrounding current location to fit relative frame
        if (DX == 1) & (DY == 0):
            T = np.rot90(image[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 1)
            Angle = np.pi/2
        elif (DX == 0) & (DY == -1):
            T = image[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2]
            Angle = 0
        elif (DX == -1) & (DY == 0):
            T = np.rot90(image[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 3)
            Angle = 3 * np.pi / 2
        else:  # (Direction[0] == 0) & (DY[-1] == 1):
            T = np.rot90(image[Y[-1]-1:Y[-1]+2, X[-1]-1:X[-1]+2], 2)
            Angle = np.pi
 
        # get first template entry that is 1
        path = np.argmax(T[row, col])
 
        # transform points by incoming directions and add to contours
        R = np.array([[np.cos(Angle), -np.sin(Angle)],
                      [np.sin(Angle), np.cos(Angle)]])
        get_coords = R.dot(np.vstack((np.array(oX[path]),
                                  np.array(oY[path])))).round()
        Direction = R.dot(np.vstack((dX[path], dY[path]))).round()
        DX = Direction[0]
        DY = Direction[1]
 
        # capture next location
        #convert to int (cannot have float indices)
        X.append(int(X[-1] + get_coords[0][0]))
        Y.append(int(Y[-1] + get_coords[1][0]))
        #X = [int(x) for x in X]
        #Y = [int(y) for y in Y]
 
        # check of last two points on contour are first two points on contour
        if(len(X) > 3):
            if(len(X) >= MaxLength) or \
                (X[-1] == X[1] and X[-2] == X[0] and
                 Y[-1] == Y[1] and Y[-2] == Y[0]):
                    X = X[0:-1]
                    Y = Y[0:-1]
                    break
 
    return X, Y
 
 
