# Значения по умолчанию
DEFAULT_A = [
    [-11, 9, 0, 0, 0],
    [-9, 17, 6, 0, 0],
    [0, 5, 20, 8, 0],
    [0, 0, -6, -20, 7],
    [0, 0, 0, 2, 8]
]

DEFAULT_b = [-117, -97, -6, 59, -86]


def tridiagonal_solve(A, d):
    n = len(A)

    P = [0] * n
    Q = [0] * n
    x = [0] * n

    # Прямой ход прогонки
    P[0] = -A[0][1] / A[0][0]
    Q[0] = d[0] / A[0][0]

    for i in range(1, n - 1):
        denom = A[i][i] + A[i][i - 1] * P[i - 1]
        P[i] = -A[i][i + 1] / denom
        Q[i] = (d[i] - A[i][i - 1] * Q[i - 1]) / denom

    # Последний шаг прямого хода
    denom = A[n - 1][n - 1] + A[n - 1][n - 2] * P[n - 2]
    Q[n - 1] = (d[n - 1] - A[n - 1][n - 2] * Q[n - 2]) / denom

    # Обратный ход
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def matrix_multiply_col(A, b):
    res = [0 for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(b)):
            res[i] += A[i][j] * b[j]
    return res


def print_matrix(matrix):
    for row in matrix:
        print(" ".join(f"{elem:6.2f}" for elem in row))


def calculate_results(A, b):
    print("Решаем систему методом прогонки...\n")

    x = tridiagonal_solve(A, b)
    Ax = matrix_multiply_col(A, x)

    print("Решение системы (x):")
    print(x)

    print("\nПроверка:")
    print("Исходный вектор b: ", b)
    print("Ax (вычисленный):  ", Ax)


if __name__ == "__main__":
    calculate_results(DEFAULT_A, DEFAULT_b)
