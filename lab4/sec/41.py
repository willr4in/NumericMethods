import numpy as np
import matplotlib.pyplot as plt

debug = False


def splitting(x0, xk, h):
    """
    Разбиение отрезка [x0, xk] с шагом h.
    Проверка корректности шагa: (xk - x0) должно быть кратно h с учётом погрешности.
    """
    if h <= 0:
        raise ValueError("Шаг сетки h должен быть положительным числом.")
    if xk <= x0:
        raise ValueError("Правая граница xk должна быть больше левой границы x0.")
    # Проверим, что (xk - x0) / h близко к целому
    n_steps = (xk - x0) / h
    if abs(round(n_steps) - n_steps) > 1e-8:
        raise ValueError(f"На отрезке [{x0}, {xk}] шаг h={h} не даёт целого числа шагов.")
    xs = []
    x = x0
    # Вдобавок проверим, что x0 != 0, чтобы избежать деления на ноль в правой части ОДУ (x^2 в знаменателе)
    if x0 == 0:
        raise ValueError("Левая граница x0 не должна быть равна нулю, чтобы избежать деления на ноль.")
    while x < xk:
        xs.append(x)
        x += h
    xs.append(xk)
    return xs


# Параметры метода Рунге–Кутты 4 порядка (не изменялись)
p = 4
as_ = [0, 0.5, 0.5, 1]
bs = [[0.5], [0, 0.5], [0, 0, 1]]
cs = [1 / 6, 1 / 3, 1 / 3, 1 / 6]


def getKs(x: float, y: np.ndarray, h):
    """
    Вычисление промежуточных приращений K для метода Рунге–Кутты 4 порядка.
    """
    dim = y.shape[0]
    Ks = np.empty((p, dim))

    for i in range(p):
        newX = x + as_[i] * h
        newY = np.copy(y)
        for j in range(i):
            newY += bs[i - 1][j] * Ks[j]

        K = h * f(newX, newY)
        if debug:
            print(f"\tK{i + 1} = {K}")
        Ks[i] = K

    return Ks


def getDeltaY(x: float, y: np.ndarray, h):
    """
    Суммирование K с коэффициентами cs, чтобы получить приращение y.
    """
    Ks = getKs(x, y, h)
    dim = Ks.shape[1]
    sum_ = np.zeros(dim)
    for i in range(p):
        sum_ += cs[i] * Ks[i]
    if debug:
        print(f"\tdeltaY = {sum_}")
    return sum_


def RungeKutta(xs: list, y0: np.ndarray, h):
    """
    Метод Рунге–Кутты 4-го порядка.
    xs — список узлов,
    y0 — вектор начальных условий [y(1), y'(1)],
    h — шаг.
    """
    N = len(xs) - 1
    dim = y0.shape[0]
    if dim != 2:
        raise ValueError("Вектор y0 должен быть размерности 2: [y(начало), y'(начало)].")
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    if debug:
        print(f"N = {N}, dim = {dim}")

    for k in range(1, N + 1):
        if debug:
            print(f"Шаг {k}")
        ys[k] = ys[k - 1] + getDeltaY(xs[k - 1], ys[k - 1], h)
        if debug:
            print(f"\ty = {ys[k]}")

    return ys


def Euler(xs: list, y0: np.ndarray, h):
    """
    Метод Эйлера.
    xs — список узлов, y0 — вектор начальных условий, h — шаг.
    """
    N = len(xs) - 1
    dim = y0.shape[0]
    if dim != 2:
        raise ValueError("Вектор y0 должен быть размерности 2: [y(начало), y'(начало)].")
    ys = np.empty((N + 1, dim))
    ys[0] = y0

    for k in range(N):
        ys[k + 1] = ys[k] + h * f(xs[k], ys[k])

    return ys


def Adams(xs: list, y0s: np.ndarray, h):
    """
    Метод Адамса 4-го порядка.
    xs — список узлов, y0s — массив из первых четырёх значений ys (запуск с помощью Runge–Kutta),
    h — шаг.
    """
    N = len(xs) - 1
    dim = y0s.shape[1]
    if dim != 2:
        raise ValueError("y0s должен быть размером (4, 2): четыре вектора начальных значений [y, y'] для запуска.")
    if N < 4:
        raise ValueError("Недостаточно узлов для применения метода Адамса 4-го порядка (требуется как минимум 5 точек).")
    ys = np.empty((N + 1, dim))

    fs = np.empty((N + 1, dim))
    # Копируем первые четыре значения
    for i in range(4):
        ys[i] = np.copy(y0s[i])
        fs[i] = f(xs[i], ys[i])

    for k in range(4, N + 1):
        ys[k] = ys[k - 1] + h / 24 * (
            55 * fs[k - 1]
            - 59 * fs[k - 2]
            + 37 * fs[k - 3]
            - 9 * fs[k - 4]
        )
        fs[k] = f(xs[k], ys[k])

    return ys


# Новая правая часть системы для ОДУ второго порядка:
def f(x: float, y: np.ndarray):
    """
    Преобразуем второе уравнение x^2 y'' + (x^2 - 2) y = 0 в систему первого порядка:
    y1 = y, y2 = y',
    тогда y1' = y2,
         y2' = ((2 - x^2) / x^2) * y1.
    """
    if x == 0:
        raise ZeroDivisionError("Вычисление f(x, y) невозможно при x=0 (деление на ноль).")
    return np.array([
        y[1],
        (2 - x ** 2) / (x ** 2) * y[0]
    ])


def getTrueY(x: float):
    """
    Точное решение: y = sin(x - 1) + (1 / x) * cos(x - 1).
    """
    return np.sin(x - 1) + (1 / x) * np.cos(x - 1)


def plot_comparisons(xs, ys_true, ys_euler, ys_runge, ys_adams):
    """
    Строит 3 отдельных графика сравнения точного решения с каждым методом
    """
    # Создаем гладкое точное решение для красивого отображения
    x_smooth = np.linspace(min(xs), max(xs), 100)
    y_smooth = [getTrueY(x) for x in x_smooth]
    
    # График 1: Метод Эйлера
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Точное решение')
    plt.plot(xs, ys_euler[:, 0], 'ro--', markersize=4, linewidth=1, label='Метод Эйлера')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение: Метод Эйлера')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 2: Метод Рунге-Кутты
    plt.subplot(1, 3, 2)
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Точное решение')
    plt.plot(xs, ys_runge[:, 0], 'gs--', markersize=4, linewidth=1, label='Метод Рунге-Кутты')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение: Метод Рунге-Кутты')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График 3: Метод Адамса
    plt.subplot(1, 3, 3)
    plt.plot(x_smooth, y_smooth, 'b-', linewidth=2, label='Точное решение')
    plt.plot(xs, ys_adams[:, 0], 'm^--', markersize=4, linewidth=1, label='Метод Адамса')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title('Сравнение: Метод Адамса')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Интервал
    a = 1.0
    b = 2.0

    # Шаг
    h = 0.1

    # Проверяем корректность шага и границ
    try:
        xs = splitting(a, b, h)
    except ValueError as ve:
        print(f"Ошибка при разбиении отрезка: {ve}")
        exit(1)

    # Начальные условия для новой задачи: y(1) = 1, y'(1) = 0
    y0 = np.array([1.0, 0.0])

    # Решения с шагом h
    try:
        ysEuler = Euler(xs, y0, h)
        ysRungeKutta = RungeKutta(xs, y0, h)
        ysAdams = Adams(xs, ysRungeKutta, h)
    except Exception as e:
        print(f"Ошибка при численном решении: {e}")
        exit(1)

    # Вычисляем точные значения для всех узлов
    ys_true = [getTrueY(x) for x in xs]

    print(f"Шаг: {h}")
    for i in range(len(xs)):
        x_val = xs[i]
        y_true = ys_true[i]

        print(f"x = {np.round(x_val, 5)}, y(точно) = {np.round(y_true, 5)}")

        errorEuler = abs(ysEuler[i][0] - y_true)
        errorRungeKutta = abs(ysRungeKutta[i][0] - y_true)
        errorAdams = abs(ysAdams[i][0] - y_true)

        print(f"\tЭйлер:      yк = {np.round(ysEuler[i][0], 5)}, e = {np.round(errorEuler, 8)}")
        print(f"\tРунге-Кутт: yк = {np.round(ysRungeKutta[i][0], 5)}, e = {np.round(errorRungeKutta, 8)}")
        print(f"\tАдамс:      yк = {np.round(ysAdams[i][0], 5)}, e = {np.round(errorAdams, 8)}")

    # Построение графиков
    plot_comparisons(xs, ys_true, ysEuler, ysRungeKutta, ysAdams)