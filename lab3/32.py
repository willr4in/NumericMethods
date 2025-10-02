import numpy as np
import matplotlib.pyplot as plt
from TDMA import tridiagonal_solve
import sympy as sp


def s(a, b, c, d, x):
    # Вычисление i-ого сплайна
    return a + b * x + c * x ** 2 + d * x ** 3


def spline_interpolation(x, y, x_test):
    n = len(x)

    # Вычисление коэффициентов c
    h = [x[i] - x[i - 1] for i in range(1, len(x))]
    # Решение СЛАУ с трехдиагональной матрицей
    A = [[0 for _ in range(len(h) - 1)] for _ in range(len(h) - 1)]
    A[0][0] = 2 * (h[0] + h[1])
    A[0][1] = h[1]
    for i in range(1, len(A) - 1):
        A[i][i - 1] = h[i - 1]
        A[i][i] = 2 * (h[i - 1] + h[i])
        A[i][i + 1] = h[i]
    A[-1][-2] = h[-2]
    A[-1][-1] = 2 * (h[-2] + h[-1])

    m = [3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]) for i in range(1, len(h))]

    c = [0] + tridiagonal_solve(A, m)

    # Вычисление коэффициентов a
    a = [y[i - 1] for i in range(1, n)]

    # Вычисление коэффициентов b
    b = [(y[i] - y[i - 1]) / h[i - 1] - (h[i - 1] / 3.0) * (2.0 * c[i - 1] + c[i]) for i in range(1, len(h))]
    b.append((y[-1] - y[-2]) / h[-1] - (2.0 * h[-1] * c[-1]) / 3.0)

    # Вычисление коэффициентов d
    d = [(c[i] - c[i - 1]) / (3.0 * h[i - 1]) for i in range(1, len(h))]
    d.append(-c[-1] / (3.0 * h[-1]))

    # Вычисление значения функции f(x*)
    for interval in range(len(x) - 1):
        if x[interval] <= x_test < x[interval + 1]:
            i = interval
            break
    else:
        i = len(x) - 2  # если x_test == x[-1]

    y_test = s(a[i], b[i], c[i], d[i], x_test - x[i])

    return a, b, c, d, y_test


def format_polynomial_sympy(a, b, c, d, x_i):
    x = sp.symbols('x')
    poly = a + b*(x - x_i) + c*(x - x_i)**2 + d*(x - x_i)**3
    
    # Раскрываем скобки и упрощаем
    poly_expanded = sp.expand(poly)
    
    # Форматируем в красивый вид
    poly_str = str(poly_expanded)
    
    # Заменяем ** на ^
    poly_str = poly_str.replace('**', '^')
    
    # Убираем лишние пробелы вокруг знаков
    poly_str = poly_str.replace('*', '')
    
    # Добавляем пробелы вокруг знаков + и -
    poly_str = poly_str.replace('+', ' + ')
    poly_str = poly_str.replace('-', ' - ')
    
    # Убираем лишние пробелы в начале
    if poly_str.startswith(' + '):
        poly_str = poly_str[3:]
    elif poly_str.startswith(' - '):
        poly_str = '-' + poly_str[3:]
    
    return f"s(x) = {poly_str}"


def draw_plot(x_original, y_original, a, b, c, d):
    plt.figure(figsize=(12, 8))
    
    # Исходные точки
    plt.scatter(x_original, y_original, color='black', s=100, zorder=5, label='Узлы интерполяции')
    
    # Сплайны
    x_plot_all = []
    y_plot_all = []
    
    for i in range(len(x_original) - 1):
        x_plot = np.linspace(x_original[i], x_original[i + 1], 100)
        y_plot = [s(a[i], b[i], c[i], d[i], x - x_original[i]) for x in x_plot]
        
        plt.plot(x_plot, y_plot, 'gray', linewidth=2)
        
        x_plot_all.extend(x_plot)
        y_plot_all.extend(y_plot)
    
    # Тестовая точка
    x_test = 0.8
    test_index = None
    for i in range(len(x_original) - 1):
        if x_original[i] <= x_test <= x_original[i + 1]:
            test_index = i
            break
    
    if test_index is not None:
        y_test = s(a[test_index], b[test_index], c[test_index], d[test_index], x_test - x_original[test_index])
        plt.scatter([x_test], [y_test], color='magenta', s=100, marker='x', linewidth=3, 
                   zorder=6, label=f'Тестовая точка x={x_test}')
    
    # Настройки графика
    plt.title('Интерполяция кубическими сплайнами', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    x = [0.1, 0.5, 0.9, 1.3, 1.7]
    y = [100.00, 4.0, 1.2346, 0.59172, 0.34602]
    x_test = 0.8

    a, b, c, d, y_test = spline_interpolation(x, y, x_test)

    print('=' * 70)
    print('ИНТЕРПОЛЯЦИЯ КУБИЧЕСКИМИ СПЛАЙНАМИ')
    print('=' * 70)
    
    for i in range(len(x) - 1):
        print(f'\nИнтервал [{x[i]}; {x[i+1]}):')
        print("-" * 50)
        
        print("Сплайн:")
        # Форматируем вывод с символами ^
        print(f"s(x) = {a[i]:.6f} + {b[i]:.6f}(x - {x[i]:.1f}) + {c[i]:.6f}(x - {x[i]:.1f})^2 + {d[i]:.6f}(x - {x[i]:.1f})^3")

    print(f'\n' + '=' * 70)
    print(f'РЕЗУЛЬТАТ ИНТЕРПОЛЯЦИИ:')
    print(f's(x*) = s({x_test}) = {y_test:.6f}')
    print('=' * 70)

    draw_plot(x, y, a, b, c, d)