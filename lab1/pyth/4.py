import numpy as np

# Значения по умолчанию
DEFAULT_A = [
    [-8, -4, 8],
    [-4, -3, 9],
    [8, 9, -5]
]
DEFAULT_EPS = 0.000001


# Позиция максимального элемента выше главной диагонали
def find_max_upper_element(X):
    n = X.shape[0]
    i_max, j_max = 0, 1
    max_elem = abs(X[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(X[i][j]) > max_elem:
                max_elem = abs(X[i][j])
                i_max = i
                j_max = j

    return i_max, j_max


# Норма вне диагональных элементов
def matrix_norm(X):
    norm = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            norm += X[i][j] ** 2
    return np.sqrt(norm)


# Метод вращений Якоби
def rotation_method(A_, eps):
    n = A_.shape[0]
    A = np.copy(A_)
    eigen_vectors = np.eye(n)
    iterations = 0

    while matrix_norm(A) > eps:
        iterations += 1
        i, j = find_max_upper_element(A)
        if A[i][i] == A[j][j]:
            phi = np.pi / 4
        else:
            phi = 0.5 * np.arctan(2 * A[i][j] / (A[i][i] - A[j][j]))

        U = np.eye(n)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        U[i][i] = cos_phi
        U[j][j] = cos_phi
        U[i][j] = -sin_phi
        U[j][i] = sin_phi

        A = U.T @ A @ U
        eigen_vectors = eigen_vectors @ U

    eigen_values = [A[i][i] for i in range(n)]
    return eigen_values, eigen_vectors, iterations


def calculate_results(A, eps):
    A = np.array(A, dtype='float')

    values, vectors, iters = rotation_method(A, eps)

    print("Собственные значения:")
    for i, v in enumerate(values):
        print(f"λ{i + 1} = {v:.6f}")

    print("\nСобственные вектора (столбцы матрицы):")
    for i in range(vectors.shape[1]):
        print(f"v{i + 1} = {vectors[:, i]}")

    print(f"\nИтераций: {iters}")

    print("\nПроверка (A * v ≈ λ * v):")
    for i in range(len(values)):
        Av = A @ vectors[:, i]
        lv = values[i] * vectors[:, i]
        print(f"A * v{i + 1} = {Av}")
        print(f"λ{i + 1} * v{i + 1} = {lv}\n")


if __name__ == "__main__":
    calculate_results(DEFAULT_A, DEFAULT_EPS)
