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

def distance_cosine(X, Y):
    # Add streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()
    with stream1:
        dot = distance_dot(X, Y)
    with stream2:
        sum_X = cp.sqrt(cp.sum(cp.sqr(X)))
    with stream3:
        sum_Y = cp.sqrt(cp.sum(cp.sqr(Y)))

    Z = cp.mult(sum_X, sum_Y)
    W = cp.divide(dot, Z)
    U = cp.subtract(1, W)
    return U

def distance_l2(X, Y):
    Z = cp.subtract(X, Y)
    W = cp.sqr(Z)
    U = cp.sum(W)
    V = cp.sqrt(U)
    return V

def distance_dot(X, Y):
    Z = cp.multiply(X, Y)
    W = cp.sum(Z)
    return W

def distance_manhattan(X, Y):
    Z = cp.subtract(X, Y)
    W = cp.abs(Z)
    U = cp.sum(W)
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
    N, D, A, K = testdata_kmeans("test_file.json")
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
