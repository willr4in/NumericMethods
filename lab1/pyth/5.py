import numpy as np

# Значения по умолчанию
DEFAULT_A = [
    [5, 8, -2],
    [7, -2, -4],
    [5, 8, -1]
]

# DEFAULT_A = [
#     [-3, 1, -1],
#     [6, 9, -4],
#     [5, -4, -8]
# ]


DEFAULT_EPS = 0.000001


def get_householder_matrix(A, col_num):
    n = A.shape[0]
    v = np.zeros(n)
    a = A[:, col_num]
    v[col_num] = a[col_num] + np.sign(a[col_num]) * np.sqrt(sum(t * t for t in a[col_num:]))
    for i in range(col_num + 1, n):
        v[i] = a[i]
    v = v[:, np.newaxis]
    H = np.eye(n) - (2 / (v.T @ v)) * (v @ v.T)
    return H


def QR_decomposition(A):
    n = A.shape[0]
    Q = np.eye(n)
    R = np.copy(A)

    for i in range(n - 1):
        H = get_householder_matrix(R, i)
        Q = Q @ H
        R = H @ R
    return Q, R


def get_roots(A, i):
    n = A.shape[0]
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def is_complex(A, i, eps):
    Q, R = QR_decomposition(A)
    A_next = R @ Q
    lambda1 = get_roots(A, i)
    lambda2 = get_roots(A_next, i)
    return abs(lambda1[0] - lambda2[0]) <= eps and abs(lambda1[1] - lambda2[1]) <= eps


def get_eigen_value(A, i, eps):
    A_i = np.copy(A)
    while True:
        Q, R = QR_decomposition(A_i)
        A_i = R @ Q
        if np.sqrt(sum(t * t for t in A_i[i + 1:, i])) <= eps:
            return A_i[i][i], A_i
        elif np.sqrt(sum(t * t for t in A_i[i + 2:, i])) <= eps and is_complex(A_i, i, eps):
            return get_roots(A_i, i), A_i


def get_eigen_values_QR(A, eps):
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []

    i = 0
    while i < n:
        cur_eigen_values, A_i_plus_1 = get_eigen_value(A_i, i, eps)
        if isinstance(cur_eigen_values, np.ndarray):
            eigen_values.extend(cur_eigen_values)
            i += 2
        else:
            eigen_values.append(cur_eigen_values)
            i += 1
        A_i = A_i_plus_1
    return eigen_values


def calculate_results(A, eps):
    A = np.array(A, dtype='float')
    eig_values = get_eigen_values_QR(A, eps)

    print("Собственные значения:")
    for i, value in enumerate(eig_values):
        if isinstance(value, np.complex128) or np.iscomplex(value):
            print(f"{i + 1}. Комплексное число: {value.real:.6f} + {value.imag:.6f}i")
        else:
            print(f"{i + 1}. Действительное число: {value:.6f}")


if __name__ == "__main__":
    print("Матрица A:")
    for row in DEFAULT_A:
        print(row)
    print(f"\nТочность: {DEFAULT_EPS}\n")
    calculate_results(DEFAULT_A, DEFAULT_EPS)
