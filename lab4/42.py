import numpy as np
from math import atan, pi
import matplotlib.pyplot as plt
from TDMA import tridiagonal_solve

def splitting(x0, xk, h):
    xs = []
    x = x0
    while x <= xk + 1e-10:
        xs.append(x)
        x += h
    xs = list(dict.fromkeys([round(x, 10) for x in xs]))
    return xs

def FiniteDifference(n, xs, h, A_b1, A_c1, A_an, A_bn, b1, bn):
    # Создаем полноразмерную матрицу n×n
    A = np.zeros((n, n))
    b = np.empty(n)
    
    # Первое уравнение: левое граничное условие
    A[0][0] = A_b1
    A[0][1] = A_c1
    b[0] = b1
    
    # Внутренние точки
    for k in range(1, n-1):
        x_val = xs[k]
        # Коэффициенты для уравнения: (x^2 + 1)*y'' - 2*y = 0
        A[k][k-1] = (x_val**2 + 1) / h**2                    # коэффициент перед y_{k-1}
        A[k][k] = -2 * (x_val**2 + 1) / h**2 - 2            # коэффициент перед y_k
        A[k][k+1] = (x_val**2 + 1) / h**2                   # коэффициент перед y_{k+1}
        b[k] = 0
    
    # Последнее уравнение: правое граничное условие
    A[n-1][n-2] = A_an
    A[n-1][n-1] = A_bn
    b[n-1] = bn

    ys = tridiagonal_solve(A, b)
    return ys


def getTrueY(x):
    """Точное аналитическое решение"""
    return x**2 + x + 1 + (x**2 + 1) * atan(x)


def RungeError(ys: np.ndarray, ys2: np.ndarray, p):
    k = 2
    error = 0
    n = min(len(ys), len(ys2) // 2 + 1)
    for i in range(n):
        if i * 2 < len(ys2):
            error = max(error, abs(ys2[i * 2] - ys[i]) / (k ** p - 1))
    return error


# Параметры задачи
a = 0
b = 1
h = 0.1

print(f"Краевая задача: (x² + 1)y'' - 2y = 0")
print(f"Граничные условия: y'(0) = 2, y(1) = {3 + pi/2:.6f}")
print(f"Отрезок: [{a}, {b}], Шаг: {h}")
print(f"Точное решение: y(x) = x² + x + 1 + (x² + 1)*arctg(x)\n")

xs = splitting(a, b, h)
n = len(xs)

# Конечно-разностная схема:
# Левое граничное условие: y'(0) = 2
# Аппроксимируем: (y1 - y0)/h = 2 => -y0/h + y1/h = 2
A_b1 = -1/h      # коэффициент перед y0
A_c1 = 1/h       # коэффициент перед y1  
b1 = 2           # правая часть

# Правое граничное условие: y(1) = 3 + pi/2
# y_n = 3 + pi/2
A_an = 0         # коэффициент перед y_{n-2}
A_bn = 1         # коэффициент перед y_{n-1}
bn = 3 + pi/2    # правая часть

ys = FiniteDifference(n, xs, h, A_b1, A_c1, A_an, A_bn, b1, bn)

print("x     | y_exact       | y_num (конечно-разностный) |    |e|")
print("-------------------------------------------------------------")
for i in range(len(xs)):
    y_ex = getTrueY(xs[i])
    y_nm = ys[i]
    err = abs(y_nm - y_ex)
    print(f"{xs[i]:.3f}  | {y_ex:12.8f} | {y_nm:24.8f} | {err:.2e}")
print("-------------------------------------------------------------")

# Апостериорная оценка погрешности
h2 = h / 2
xs2 = splitting(a, b, h2)
n2 = len(xs2)
ys2 = FiniteDifference(n2, xs2, h2, -1/h2, 1/h2, 0, 1, 2, 3 + pi/2)
err_est = RungeError(ys, ys2, 2)
print(f"Апостериорная оценка погрешности (порядок 2): {err_est:.2e}")

# Построение графика сравнения
plt.figure(figsize=(10, 6))

# Точное решение (плавная линия)
x_dense = np.linspace(a, b, 1000)
y_exact_dense = [getTrueY(x) for x in x_dense]
plt.plot(x_dense, y_exact_dense, 'b-', linewidth=2, label='Точное решение')

# Численное решение конечно-разностным методом
plt.plot(xs, ys, 'ro-', markersize=4, linewidth=1, label='Конечно-разностный метод')

plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Сравнение точного решения и конечно-разностного метода')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()