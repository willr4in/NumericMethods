import numpy as np
import sys
from math import atan, pi


def splitting(x0, xk, h):
    """
    Разбивает отрезок [x0, xk] на узлы с шагом h.
    Проверяет корректность: h>0, xk>x0.
    """
    if h <= 0:
        raise ValueError("Шаг h должен быть положительным.")
    if xk <= x0:
        raise ValueError("Правая граница xk должна быть больше левой границы x0.")
    xs = []
    x = x0
    while x <= xk + 1e-10:  # Изменил условие для включения конечной точки
        xs.append(x)
        x += h
    # Убираем возможные дубликаты из-за погрешности вычислений
    xs = list(dict.fromkeys([round(x, 10) for x in xs]))
    return xs


def tridiagonalMatrixAlgorithm(A, b):
    flag = False
    """
    Решение трехдиагональной системы Ax = b (размер n×n, но A хранится в виде
    массива Nx3: A[i][0]=нижний, A[i][1]=диагональный, A[i][2]=верхний).
    """
    n = len(b)
    if A.shape != (n, 3):
        raise ValueError("Матрица A должна иметь форму (n,3), где n = len(b).")

    # Диагональное преобладание (для внутренних строк)
    for i in range(n):
        diag = abs(A[i][1])
        off_sum = abs(A[i][0]) + abs(A[i][2])
        if i == 0:
            if abs(A[i][1]) < 1e-14:
                raise ValueError(f"Нулевой коэффициент на диагонали в строке {i}.")
        elif i == n-1:
            if abs(A[i][1]) < 1e-14:
                raise ValueError(f"Нулевой коэффициент на диагонали в строке {i}.")
        else:
            if diag < off_sum and not flag:
                print(
                    f"Строка {i}: нарушено условие диагонального преобладания: "
                    f"|b[{i}]|={diag:.3e} < |a[{i}]|+|c[{i}]|={off_sum:.3e}."
                )
                flag = True

    # Прямой ход (метод прогонки)
    P = np.zeros(n)
    Q = np.zeros(n)
    
    P[0] = -A[0][2] / A[0][1]
    Q[0] = b[0] / A[0][1]

    for i in range(1, n):
        denom = A[i][1] + A[i][0] * P[i - 1]
        if abs(denom) < 1e-14:
            raise ZeroDivisionError(f"Нулевой знаменатель в прогонке на строке {i}.")
        P[i] = -A[i][2] / denom
        Q[i] = (b[i] - A[i][0] * Q[i - 1]) / denom

    # Обратный ход
    x = np.zeros(n)
    x[n - 1] = Q[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def FiniteDifference(n, xs, h):
    """
    Решение краевой задачи:
      (x^2 + 1)*y'' - 2*y = 0,  x ∈ [0,1]
      y'(0) = 2,   y(1) = 3 + pi/2
    методом конечных разностей.
    """
    if n != len(xs):
        raise ValueError("n должно совпадать с числом узлов в xs.")

    A = np.zeros((n, 3))
    b = np.zeros(n)

    # Левое граничное условие: y'(0) = 2
    # Аппроксимируем: (y_1 - y_0)/h = 2 → -y_0 + y_1 = 2h
    A[0, 1] = -1.0
    A[0, 2] = 1.0
    b[0] = 2.0 * h

    # Правое граничное условие: y(1) = 3 + pi/2
    # y_n = 3 + pi/2
    A[n - 1, 1] = 1.0
    b[n - 1] = 3.0 + pi/2

    # Заполняем внутренние узлы k = 1, 2, ..., n-2
    for k in range(1, n - 1):
        xk = xs[k]
        # Уравнение: (x^2 + 1)*y'' - 2*y = 0
        # Аппроксимируем: (x^2 + 1) * (y_{k-1} - 2y_k + y_{k+1})/h^2 - 2*y_k = 0
        # Умножаем на h^2: (x^2 + 1) * (y_{k-1} - 2y_k + y_{k+1}) - 2*h^2*y_k = 0
        # Перегруппируем: (x^2 + 1)*y_{k-1} + [-2(x^2 + 1) - 2*h^2]*y_k + (x^2 + 1)*y_{k+1} = 0
        coeff = xk**2 + 1.0
        A[k, 0] = coeff                    # коэффициент при y_{k-1}
        A[k, 1] = -2.0 * coeff - 2.0 * h**2  # коэффициент при y_k
        A[k, 2] = coeff                    # коэффициент при y_{k+1}
        b[k] = 0.0

    # Решаем трехдиагональную систему
    ys = tridiagonalMatrixAlgorithm(A, b)
    return ys


def RungeError(ys: np.ndarray, ys2: np.ndarray, order: int):
    """
    Апостериорная оценка погрешности по Рунге:
    сравниваем сетку с шагом h (ys) и h/2 (ys2).
    Используем только общие узлы.
    """
    factor = 2**order - 1
    max_err = 0.0
    # Используем минимальную длину для сравнения
    n = min(len(ys), len(ys2) // 2 + 1)
    for i in range(n):
        if i * 2 < len(ys2):  # Проверяем, чтобы индекс не выходил за границы
            err_i = abs(ys2[i * 2] - ys[i]) / factor
            if err_i > max_err:
                max_err = err_i
    return max_err


def getTrueY(x: float):
    """
    Точное решение: y(x) = x^2 + x + 1 + (x^2 + 1)*arctg(x)
    """
    return x**2 + x + 1 + (x**2 + 1) * atan(x)


if __name__ == "__main__":
    # Параметры задачи
    a = 0.0
    b = 1.0
    h = 0.1
    order = 2    # порядок метода конечных разностей

    print(f"Краевая задача: (x² + 1)y'' - 2y = 0")
    print(f"Граничные условия: y'(0) = 2, y(1) = {3 + pi/2:.6f}")
    print(f"Отрезок: [{a}, {b}], Шаг: {h}")
    print(f"Точное решение: y(x) = x² + x + 1 + (x² + 1)*arctg(x)\n")

    # Строим узлы на [0, 1] с шагом h
    try:
        xs = splitting(a, b, h)
    except ValueError as e:
        print("Ошибка при разбиении отрезка:", e)
        sys.exit(1)

    n = len(xs)

    # Решаем краевую задачу методом конечных разностей
    try:
        ys = FiniteDifference(n, xs, h)
    except Exception as e:
        print("Ошибка при решении методом конечных разностей:", e)
        sys.exit(1)

    # Выводим сравнение с точным решением
    print("x     | y_exact       | y_num (конечно-разностный) |    |e|")
    print("-------------------------------------------------------------")
    for i, xi in enumerate(xs):
        y_ex = getTrueY(xi)
        y_nm = ys[i]
        err = abs(y_nm - y_ex)
        print(f"{xi:.3f}  | {y_ex:12.8f} | {y_nm:24.8f} | {err:.2e}")
    print("-------------------------------------------------------------\n")

    # Апостериорная оценка погрешности (метод Рунге) на шаге h/2
    h2 = h / 2.0
    try:
        xs2 = splitting(a, b, h2)
    except ValueError as e:
        print("Ошибка при разбиении отрезка для h/2:", e)
        sys.exit(1)
    n2 = len(xs2)

    try:
        ys2 = FiniteDifference(n2, xs2, h2)
    except Exception as e:
        print("Ошибка при решении методом конечных разностей (h/2):", e)
        sys.exit(1)
    
    err_est = RungeError(ys, ys2, order)
    print(f"Апостериорная оценка погрешности по Рунге (порядок {order}): {err_est:.2e}")