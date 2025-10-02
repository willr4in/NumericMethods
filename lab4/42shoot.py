import numpy as np
from random import randint
import matplotlib.pyplot as plt


def splitting(x0, xk, h):
    """
    Разбивает отрезок [x0, xk] на узлы с шагом h.
    """
    if h <= 0:
        raise ValueError("Шаг h должен быть положительным.")
    if xk <= x0:
        raise ValueError("Правая граница xk должна быть больше левой границы x0.")
    xs = []
    x = x0
    while x <= xk + 1e-10:
        xs.append(x)
        x += h
    xs = list(dict.fromkeys([round(x, 10) for x in xs]))
    return xs


# Параметры метода Рунге–Кутты 4-го порядка
p = 4
as_ = [0, 0.5, 0.5, 1]
bs = [[0.5], [0, 0.5], [0, 0, 1]]
cs = [1/6, 1/3, 1/3, 1/6]


def getKs(x: float, y: np.ndarray, h: float):
    """
    Вычисление приращений Ki для одного шага RK4.
    y — вектор [y, y'].
    """
    dim = y.shape[0]
    Ks = np.empty((p, dim))
    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += bs[i-1][j] * Ks[j]
        K = h * f(newX, newY)
        Ks[i] = K
    return Ks


def getDeltaY(x: float, y: np.ndarray, h: float):
    """
    Суммирует вклад Ki с коэффициентами ci для RK4.
    """
    Ks = getKs(x, y, h)
    dim = Ks.shape[1]
    total = np.zeros(dim)
    for i in range(p):
        total += cs[i] * Ks[i]
    return total


def RungeKutta(xs: list, y0: np.ndarray, h: float):
    """
    Применение метода Рунге–Кутты 4-го порядка
    к системе y' = f(x, y) на сетке xs с начальным условием y0.
    """
    N = len(xs) - 1
    dim = y0.shape[0]
    ys = np.empty((N + 1, dim))
    ys[0] = y0
    for k in range(1, N + 1):
        ys[k] = ys[k-1] + getDeltaY(xs[k-1], ys[k-1], h)
    return ys


def Shooting(xs: list, y0_val: float, h: float, eps: float):
    """
    Метод стрельбы для краевой задачи:
      (x² + 1)y'' - 2y = 0,
      y'(0) = 2,   y(1) = 3 + π/2.
    """
    # Используем более осмысленные начальные приближения
    eta0 = 0.5
    eta1 = 1.5

    # Интегрируем дважды: сначала с η0, потом с η1
    ys0 = RungeKutta(xs, np.array([eta0, y0_val]), h)
    F0 = ys0[-1][0] - (3 + np.pi/2)

    ys1 = RungeKutta(xs, np.array([eta1, y0_val]), h)
    F1 = ys1[-1][0] - (3 + np.pi/2)

    iteration = 1
    while True:
        if abs(F1 - F0) < 1e-16:
            raise ZeroDivisionError("Деление на ноль при вычислении нового η (F1≈F0).")
        eta = eta1 - (eta1 - eta0) * F1 / (F1 - F0)

        ys = RungeKutta(xs, np.array([eta, y0_val]), h)
        F_new = ys[-1][0] - (3 + np.pi/2)

        if abs(F_new) < eps:
            return ys, iteration, eta

        iteration += 1
        eta0, F0 = eta1, F1
        eta1, F1 = eta, F_new


def RungeError(ys_h: np.ndarray, ys_h2: np.ndarray, order: int):
    """
    Апостериорная оценка погрешности по Рунге (сравнение сеток h и h/2).
    """
    factor = 2**order - 1
    max_err = 0.0
    n = min(len(ys_h), len(ys_h2) // 2 + 1)
    
    for i in range(n):
        if i * 2 < len(ys_h2):
            err_i = abs(ys_h2[i*2][0] - ys_h[i][0]) / factor
            if err_i > max_err:
                max_err = err_i
    return max_err


def f(x: float, y: np.ndarray):
    """
    Правая часть системы для ОДУ:
      (x² + 1)y'' - 2y = 0
    """
    return np.array([
        y[1],  # y1' = y2
        2 * y[0] / (x**2 + 1)  # y2' = 2y1 / (x² + 1)
    ])


def getTrueY(x: float):
    """
    Точное решение задачи: y(x) = x² + x + 1 + (x² + 1)*arctg(x)
    """
    return x**2 + x + 1 + (x**2 + 1) * np.arctan(x)


if __name__ == "__main__":
    # Параметры задачи
    a = 0.0       # левая граница
    b = 1.0       # правая граница
    y0 = 2.0      # y'(0) = 2
    h = 0.1       # шаг
    eps = 1e-9    # точность для метода стрельбы

    print(f"Краевая задача: (x² + 1)y'' - 2y = 0")
    print(f"Граничные условия: y'(0) = {y0}, y(1) = {3 + np.pi/2:.6f}")
    print(f"Отрезок: [{a}, {b}], Шаг: {h}, Точность (eps): {eps}")
    print(f"Точное решение: y(x) = x² + x + 1 + (x² + 1)*arctg(x)\n")

    xs = splitting(a, b, h)

    # Запускаем метод стрельбы
    try:
        ysShooting, iterShooting, eta_found = Shooting(xs, y0, h, eps)
    except ZeroDivisionError as e:
        print("Ошибка в методе стрельбы:", e)
        raise

    print(f"Метод стрельбы завершился за {iterShooting} итераций.")

    # Сравниваем с точным решением в узлах xs
    print("x     | y_exact       | y_num (стрельбой) |    |e|")
    print("-----------------------------------------------------")
    for i in range(len(xs)):
        y_ex = getTrueY(xs[i])
        y_nm = ysShooting[i][0]
        err = abs(y_nm - y_ex)
        print(f"{xs[i]:.3f}  | {y_ex:12.8f} | {y_nm:16.8f} | {err:.2e}")
    print("-----------------------------------------------------")

    # Апостериорная оценка погрешности
    h2 = h / 2
    xs2 = splitting(a, b, h2)
    ysShooting2 = RungeKutta(xs2, np.array([eta_found, y0]), h2)
    err_est = RungeError(ysShooting, ysShooting2, p)
    print(f"Апостериорная оценка погрешности (порядок {p}): {err_est:.2e}")

    # Построение графика сравнения
    plt.figure(figsize=(10, 6))
    
    # Точное решение (плавная линия)
    x_dense = np.linspace(a, b, 1000)
    y_exact_dense = [getTrueY(x) for x in x_dense]
    plt.plot(x_dense, y_exact_dense, 'b-', linewidth=2, label='Точное решение')
    
    # Численное решение методом стрельбы
    y_shooting = ysShooting[:, 0]
    plt.plot(xs, y_shooting, 'ro-', markersize=4, linewidth=1, label='Метод стрельбы')
    
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение точного решения и метода стрельбы')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()