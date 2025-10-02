# Значения по умолчанию
DEFAULT_A = [
    [-25, 4, -4, 9],
    [-9, 21, 5, -6],
    [9, 2, 19, -7],
    [-7, 4, -7, 25]
]

DEFAULT_b = [86, 29, 28, 68]

DEFAULT_eps = 0.001


# Метод простых итераций
def solve_iterative(A, b, eps):
    n = len(b)
    alpha = [[(-A[i][j] / A[i][i] if i != j else 0) for j in range(n)] for i in range(n)]
    beta = [b[i] / A[i][i] for i in range(n)]
    x = beta[:]
    iterations = 0

    while True:
        iterations += 1
        mult = matrix_multiply_col(alpha, x)
        x_new = [mult[i] + beta[i] for i in range(n)]

        norm_alpha = max(sum(abs(t) for t in alpha[i]) for i in range(n))
        if norm_alpha < 1:
            eps_i = max(abs(x_new[i] - x[i]) for i in range(n)) * norm_alpha / (1 - norm_alpha)
        else:
            eps_i = max(abs(x_new[i] - x[i]) for i in range(n))

        if eps_i < eps:
            break
        x = x_new[:]

    return x_new, iterations


# Метод Зейделя
def solve_seidel(A, b, eps):
    n = len(b)
    alpha = [[(-A[i][j] / A[i][i] if i != j else 0) for j in range(n)] for i in range(n)]
    beta = [b[i] / A[i][i] for i in range(n)]
    x = beta[:]
    iterations = 0

    while True:
        iterations += 1
        x_new = [0] * n

        for i in range(n):
            sum1 = sum(alpha[i][j] * x_new[j] for j in range(i))
            sum2 = sum(alpha[i][j] * x[j] for j in range(i, n))
            x_new[i] = beta[i] + sum1 + sum2

        norm_alpha = max(sum(abs(t) for t in alpha[i]) for i in range(n))
        norm_c = max(sum(abs(alpha[i][j]) for j in range(i, n)) for i in range(n))

        if norm_alpha < 1:
            eps_i = max(abs(x_new[i] - x[i]) for i in range(n)) * norm_c / (1 - norm_alpha)
        else:
            eps_i = max(abs(x_new[i] - x[i]) for i in range(n))

        if eps_i < eps:
            break

        x = x_new[:]

    return x_new, iterations


def matrix_multiply_col(A, b):
    return [sum(A[i][j] * b[j] for j in range(len(b))) for i in range(len(A))]


def calculate_results(A, b, eps):
    print("Решение системы методом итераций и методом Зейделя\n")

    x_iter, i_iter = solve_iterative(A, b, eps)
    x_seidel, i_seidel = solve_seidel(A, b, eps)

    print("Метод простых итераций:")
    print(f"x = {x_iter}")
    print(f"Количество итераций: {i_iter}")
    print(f"Проверка: Ax = {matrix_multiply_col(A, x_iter)}\n")

    print("Метод Зейделя:")
    print(f"x = {x_seidel}")
    print(f"Количество итераций: {i_seidel}")
    print(f"Проверка: Ax = {matrix_multiply_col(A, x_seidel)}")


# Запуск программы
if __name__ == "__main__":
    calculate_results(DEFAULT_A, DEFAULT_b, DEFAULT_eps)
