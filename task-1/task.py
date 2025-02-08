import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def add_kernel(X, Y):
    return cp.add(X, Y)

def sum_kernel(X):
    return cp.sum(X)

def subtract_kernel(X, Y):
    return cp.subtract(X, Y)

def mult_kernel(X, Y):
    return cp.multiply(X, Y)

def divide_kernel(X, Y):
    return cp.divide(X, Y)

def sqrt_kernel(X):
    return cp.sqrt(X)

def sqr_kernel(X):
    return cp.square(X)

def abs_kernel(X):
    return cp.abs(X)

def distance_cosine(X, Y):
    # Add streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()
    with stream1:
        dot = distance_dot(X, Y)
    with stream2:
        sum_X = sqrt_kernel(sum_kernel(sqr_kernel(X)))
    with stream3:
        sum_Y = sqrt_kernel(sum_kernel(sqr_kernel(Y)))
    cp.cuda.Stream.null.synchronize()
    Z = mult_kernel(sum_X, sum_Y)
    W = divide_kernel(dot, Z)
    U = subtract_kernel(1, W)
    return U

def distance_l2(X, Y):
    Z = add_kernel(X, Y)
    W = sqr_kernel(Z)
    U = sum_kernel(W)
    V = sqrt_kernel(U)
    return V

def distance_dot(X, Y):
    Z = mult_kernel(X, Y)
    W = sum_kernel(Z)
    return W

def distance_manhattan(X, Y):
    Z = subtract_kernel(X, Y)
    W = abs_kernel(Z)
    U = sum_kernel(W)
    return U

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, X, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

if __name__ == "__main__":
    test_kmeans()
