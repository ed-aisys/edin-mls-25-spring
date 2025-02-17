import numpy as np
import json
from task import our_ann
import cupy as cp

def read_data(file_path=""):
    """
    Read data from a file
    """
    if file_path == "":
        return None
    if file_path.endswith(".npy"):
        return np.load(file_path)
    else:
        return np.loadtxt(file_path)

def testdata_kmeans(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        K = 10
        return N, D, A, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
        return N, D, A, K

def testdata_knn(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
            X = np.loadtxt(X_file)
        return N, D, A, X, K
    
def testdata_ann(test_file):
    if test_file == "":
        # use random data
        N = 1000
        D = 100
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        K = 10
        return N, D, A, X, K
    else:
        # read n, d, a_file, x_file, k from test_file.json
        with open(test_file, "r") as f:
            data = json.load(f)
            N = data["n"]
            D = data["d"]
            A_file = data["a_file"]
            X_file = data["x_file"]
            K = data["k"]
            A = np.loadtxt(A_file)
            X = np.loadtxt(X_file)
        return N, D, A, X, K
    
def test_our_ann():
    # Generate test data
    N, D, A, X, K = testdata_ann()
    
    # Convert data to CuPy arrays
    A_cp = cp.asarray(A)
    X_cp = cp.asarray(X)
    
    # Run the our_ann function
    top_k_indices = our_ann(N, D, A_cp, X_cp, K)
    
    # Convert the result back to NumPy for assertion
    top_k_indices_np = cp.asnumpy(top_k_indices)
    
    # Check the length of the result
    assert len(top_k_indices_np) == K, f"Expected {K} indices, but got {len(top_k_indices_np)}"
    
    # Check if the indices are within the valid range
    assert np.all(top_k_indices_np < N), "Some indices are out of range"
    
    print("top_k_indices_np.shape:", top_k_indices_np.shape)
    print("top_k_indices_np:", top_k_indices_np)
    
    print("Test passed!")

# Run the test
if __name__ == "__main__":
    test_our_ann()
