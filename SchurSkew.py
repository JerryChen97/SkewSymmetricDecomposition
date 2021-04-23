from scipy.linalg import schur, block_diag
import numpy as np

def get_nonzero_modes(v_diag_k1):
    """
        Find the exact position of nonzero mode
    """
    L = len(v_diag_k1)
    n = (L + 1) // 2
    num_non_zero = 0
    non_zero_list = []
    non_zero_pos_list = []
    for i, v in enumerate(v_diag_k1):
        if not np.allclose(v, 0):
            non_zero_list.append(np.real(v))
            non_zero_pos_list.append([i, i+1])
            num_non_zero+=1

    
    return non_zero_list, non_zero_pos_list

def get_zero_modes(n, non_zero_pos_list):
    """
        Remove nonzero from the full space and then here are the zero modes
    """
    full_indices = [i for i in range(2*n)]
    zero_indices = full_indices.copy()
    for non_zero_pos in non_zero_pos_list:
        zero_indices = list(set(zero_indices) - set(non_zero_pos))
    return zero_indices

def Schur_Skew(mat):
    assert mat.shape[0]==mat.shape[1]
    assert mat.shape[0] % 2 == 0
    assert np.allclose(mat+mat.T, 0)

    n = mat.shape[0] // 2

    T, Z = schur(mat, output='real')
    T_offdiag = np.diag(T, k=1)
    non_zeros, non_zeros_pos = get_nonzero_modes(T_offdiag)
    non_zero_subspace = [Z[:, pos] for pos in non_zeros_pos]

    nullspace_indices = get_zero_modes(n, non_zeros_pos)
    nullspace = Z[:, nullspace_indices]
    n_zero = len(nullspace_indices) // 2

    for i, v in enumerate(non_zeros):
        subspace = non_zero_subspace[i]
        if v < 0: # swap the two bases to make the final block canonical...
            subspace[:, [0, 1]] = subspace[:, [1, 0]]
        v = np.abs(v)
    
    # return non_zeros, non_zero_subspace, nullspace
    non_zero_tuples = [(np.abs(non_zeros[i]), non_zero_subspace[i]) for i in range(len(non_zeros))]
    non_zero_tuples.sort(key=(lambda e:e[0]))
    
    # Create new blocks T
    T_zero = [np.zeros((2*n_zero, 2*n_zero))]
    T_nonzero = [t[0] * np.array([[0., 1.], [-1., 0.]]) for t in non_zero_tuples]
    T_new = block_diag(*(T_zero + T_nonzero))

    # Create new eigen-modes (the orthogonal matrix)
    Z_new = Z.copy()
    Z_new[:, :(2*n_zero)] = nullspace
    for i, t in enumerate(non_zero_tuples):
        subspace = t[1]
        start = (2*n_zero + 2*i)
        Z_new[:, start:start+2] = subspace
    # print(Z_new.shape)
    # print(np.allclose(mat - Z_new@T_new@Z_new.T, 0, atol=1e-6))
    return T_new, Z_new
