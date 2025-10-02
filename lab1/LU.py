import numpy as np
import copy

# Дефолтные значения для матрицы A и вектора b
A = [
    [-7, -2, -1, -4],
    [-4, 6, 0, -4],
    [-8, 2, -9, -3],
    [0, 0, -7, 1]
]

b = [-12, 22, 51, 49]

def LU_decompose(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = copy.deepcopy(A)

    for k in range(n):
        if abs(U[k][k]) < 1e-12:
            raise ValueError("Матрица вырождена или требует перестановки строк")
            
        for i in range(k, n):
            L[i][k] = U[i][k] / U[k][k]
        for i in range(k + 1, n):
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U

def solve_system(L, U, b):
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        sum_val = sum(y[j] * L[i][j] for j in range(i))
        y[i] = b[i] - sum_val

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-12:
            raise ValueError("Матрица вырождена")
            
        sum_val = sum(x[j] * U[i][j] for j in range(i + 1, n))
        x[i] = (y[i] - sum_val) / U[i][i]
    return x

def determinant(A):
    _, U = LU_decompose(A)
    det = 1
    for i in range(len(U)):
        det *= U[i][i]
    return det

def inverse_matrix(A):
    n = len(A)
    E = [[float(i == j) for i in range(n)] for j in range(n)]
    L, U = LU_decompose(A)
    A_transposed = []
    for e in E:
        transposed_row = solve_system(L, U, e)
        A_transposed.append(transposed_row)
    return transpose(A_transposed)

def transpose(X):
    return [list(row) for row in zip(*X)]

def matrix_multiply(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

def matrix_multiply_col(A, B):
    res = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B)):
            res[i] += A[i][j] * B[j]
    return res

def matrix_to_string(matrix):
    return '\n'.join([' '.join([f'{value:6.2f}' for value in row]) for row in matrix])

def calculate_results(A, b):

    L, U = LU_decompose(A)
    x = solve_system(L, U, b)
    det = determinant(A)
    inv_A = inverse_matrix(A)

    print("LU-разложение:")
    print("L =")
    print(matrix_to_string(L))
    print("\nU =")
    print(matrix_to_string(U))

    print("\nLU =")
    print(matrix_to_string(matrix_multiply(L, U)))

    print("\nРешение системы (x):")
    print(x)

    print("\nПроверка (Ax):")
    print("b =", b)
    print("Ax =", matrix_multiply_col(A, x))

    print("\nОпределитель:")
    print("det =", det)
    print("Проверка с помощью numpy:", np.linalg.det(A))

    print("\nОбратная матрица:")
    print(matrix_to_string(inv_A))

    print("\nПроверка AA^(-1):")
    product = matrix_multiply(A, inv_A)
    print(matrix_to_string(product))

# Запуск расчётов
if __name__ == "__main__":
    calculate_results(A, b)