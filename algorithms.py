import numpy as np
import scipy
import math

def Naive(matrix, threshold):
    copy_matrix = matrix.copy()
    copy_matrix[np.abs(matrix) <= threshold] = 0
    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    print("number of non-zeros of the sparsed matrix: " + str(len(copy_matrix.nonzero()[0])))
    return copy_matrix

def AHK06(matrix, threshold):
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    probs = np.random.random((n, d))
    copy_matrix[np.abs(matrix) < threshold] = 0
    indices = probs < (np.abs(matrix) / threshold) * (np.abs(matrix) < threshold)
    copy_matrix[indices] = threshold * np.sign(matrix[indices])
    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    print("number of non-zeros of the sparsed matrix: " + str(len(copy_matrix.nonzero()[0])))
    return copy_matrix

def AKL13(matrix, s):
    matrix = matrix.T
    n, d = matrix.shape
    row_norms = np.linalg.norm(matrix, axis=1, ord=1)
    rou = compute_row_distribution(matrix, s, 0.1, row_norms)
    nonzero_indices = matrix.nonzero()
    data = matrix[nonzero_indices]
    row_norms[row_norms == 0] = 1
    probs_matrix = rou.reshape((n, 1)) * matrix / row_norms.reshape((n, 1))
    probs = probs_matrix[nonzero_indices]
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), data[selected] / (probs[selected] * s))
    result = result.T
    matrix = matrix.T
    result = scipy.sparse.csr_matrix(result)
    print("number of non-zeros of the sparsed matrix: " + str(len(result.nonzero()[0])))
    return result

def compute_row_distribution(matrix, s, delta, row_norms):
    m, n = matrix.shape
    z = row_norms / np.sum(row_norms)
    alpha, beta = math.sqrt(np.log((m + n) / delta) / s), np.log((m + n) / delta) / (3 * s)
    zeta = 1
    rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
    sum = np.sum(rou)
    while np.abs(sum - 1) > 1e-5:
        zeta *= sum
        rou = (alpha * z / (2 * zeta) + ((alpha * z / (2 * zeta)) ** 2 + beta * z / zeta) ** (1 / 2)) ** 2
        sum = np.sum(rou)
    return rou

def DZ11(matrix, threshold):
    copy_matrix = matrix.copy()
    n, d = matrix.shape
    norm_fro = np.linalg.norm(matrix, ord="fro")
    copy_matrix[np.abs(matrix) <= threshold / (n + d)] = 0
    s = int(14 * (n + d) * np.log(np.sqrt(2) / 2 * (n + d)) * (norm_fro / threshold) ** 2)
    nonzero_indices = copy_matrix.nonzero()
    data = copy_matrix[nonzero_indices]
    probs_matrix = copy_matrix * copy_matrix
    probs = probs_matrix[nonzero_indices]
    probs /= np.sum(probs)
    indices = np.arange(len(data))
    selected = np.random.choice(indices, s, p=probs, replace=True)
    result = np.zeros((n, d))
    np.add.at(result, (nonzero_indices[0][selected], nonzero_indices[1][selected]), data[selected] / (probs[selected] * s))
    result = scipy.sparse.csr_matrix(result)
    print("number of non-zeros of the sparsed matrix: " + str(len(result.nonzero()[0])))
    return result

def RMR(matrix, threshold):
    copy_matrix = matrix.copy()
    np.apply_along_axis(row_operation, 1, copy_matrix, threshold)
    copy_matrix = scipy.sparse.csr_matrix(copy_matrix)
    print("number of non-zeros of the sparsed matrix: " + str(len(copy_matrix.nonzero()[0])))
    return copy_matrix

def row_operation(row, threshold):
    argzero = np.argwhere((np.abs(row) <= threshold) * (row != 0))
    argzero = argzero.reshape(len(argzero),)
    copy_row = row.copy()
    row[argzero] = 0
    sum = np.sum(copy_row[argzero])
    if sum != 0:
        k = math.ceil(sum / threshold)
            
        indices = np.random.choice(argzero, k, p=copy_row[argzero]/sum, replace=True)
        np.add.at(row, indices, sum / k * np.sign(copy_row[indices]))
