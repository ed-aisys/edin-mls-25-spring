import torch
import cupy as cp
# import triton
import numpy as np
import time
import json
import scipy
from test import testdata_kmeans, testdata_knn, testdata_ann
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
subtract_square = cp.ElementwiseKernel('float64 x, float64 y', 
                                       'float64 z', 
                                       '''
                                       z = (x - y);
                                       z = z * z;                                   
                                       '''
                                       )

subtract_abs = cp.ElementwiseKernel('float64 x, float64 y',
                                    'float64 z',
                                    '''
                                    z = abs(x - y)
                                    ''')

sum_sqrt = cp.ReductionKernel('float64 x', 'float64 y', 'x', 'a + b', 'y = sqrt(a)', '0')

multiply = cp.ElementwiseKernel('float64 x float64 y', 
                                'float64 z', 
                                '''z = x * y''')

_sum = cp.ReductionKernel('float64 x', 'float64 y', 'x', 'a + b', 'y = a', '0')

def distance_cosine(X, Y, use_kernel=True):
    # Add streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()
    with stream2:
        sum_X = cp.sqrt(cp.sum(cp.square(X)))
    with stream3:
        sum_Y = cp.sqrt(cp.sum(cp.square(Y)))
    Z = cp.multiply(sum_X, sum_Y)
    with stream1:
        dot = distance_dot(X, Y)
    W = cp.divide(dot, Z)
    U = cp.subtract(1, W)
    return U

def distance_l2(X, Y, use_kernel=True):
    if use_kernel:
        W = subtract_square(X, Y)
        V = sum_sqrt(W)
    else:
        Z = cp.subtract(X, Y)
        W = cp.square(Z)
        U = cp.sum(W)
        V = cp.sqrt(U)
    return V

def distance_dot(X, Y, use_kernel=True):
    if use_kernel:
        Z = multiply(X, Y)
        W = _sum(Z)
    else:
        Z = cp.multiply(X, Y)
        W = cp.sum(Z)
    return W

def distance_manhattan(X, Y, use_kernel=True):
    if use_kernel:
        Z = subtract_abs(X, Y)
        U = _sum(Z)
    else:
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

def test_cosine(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_cosine(X, Y)
    end = time.time()
    gold = 1 - torch.cosine_similarity(torch.tensor(X, dtype=float), torch.tensor(Y, dtype=float), dim=0).item()
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))

def test_l2(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_l2(X, Y, use_kernel=True)
    end = time.time()
    gold = cp.linalg.norm(X - Y)
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))

def test_dot(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_dot(X, Y)
    end = time.time()
    gold = cp.dot(X, Y)
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))

def test_manhattan(D=2, use_kernel=True):
    X, Y = cp.random.randn(D), cp.random.randn(D)
    start = time.time()
    ours = distance_manhattan(X, Y)
    end = time.time()
    gold = scipy.spatial.distance.cityblock(X.get(), Y.get())
    assert cp.isclose([ours], [gold])
    print("Execution Time: {}".format(end - start))

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
    print("Dimension: 2")
    D = 2
    print("Cosine Distance Test")
    test_cosine(D)
    print("L2 Distance Test")
    test_l2(D)
    print("Dot Distance Test")
    test_dot(D)
    print("Manhattan Distance Test")
    test_manhattan(D)
