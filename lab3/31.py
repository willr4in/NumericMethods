import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 1 / (x**2)


def lagrange_basis_polynomial(x_points, i, x):
    """Вычисляет i-й базисный полином Лагранжа в точке x"""
    result = 1.0
    for j in range(len(x_points)):
        if i != j:
            result *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return result


def lagrange_interpolation(x_points, y_points, x):
    """Вычисляет значение интерполяционного полинома Лагранжа в точке x"""
    result = 0.0
    for i in range(len(x_points)):
        result += y_points[i] * lagrange_basis_polynomial(x_points, i, x)
    return result


def newton_interpolation(x_points, y_points, x):
    """Вычисляет значение интерполяционного полинома Ньютона в точке x"""
    n = len(x_points)
    
    # Таблица разделенных разностей
    coefs = y_points.copy()
    for i in range(1, n):
        for j in range(n-1, i-1, -1):
            coefs[j] = (coefs[j] - coefs[j-1]) / (x_points[j] - x_points[j-i])
    
    # Вычисление значения полинома
    result = coefs[0]
    product = 1.0
    for i in range(1, n):
        product *= (x - x_points[i-1])
        result += coefs[i] * product
    
    return result


def expand_polynomial_from_roots(roots):
    """Раскрывает полином из произведения (x-r1)(x-r2)... в стандартную форму"""
    # Начинаем с полинома 1
    coeffs = [1.0]
    
    for root in roots:
        # Умножаем текущий полином на (x - root)
        new_coeffs = [0.0] * (len(coeffs) + 1)
        for i in range(len(coeffs)):
            new_coeffs[i] += coeffs[i] * (-root)  # Умножение на -root
            new_coeffs[i+1] += coeffs[i]          # Умножение на x
        coeffs = new_coeffs
    
    return coeffs


def get_lagrange_polynomial_str(x_points, y_points):
    """Возвращает строковое представление полинома Лагранжа в стандартной форме"""
    n = len(x_points)
    
    # Вычисляем общие коэффициенты полинома
    total_coeffs = [0.0] * n
    
    for i in range(n):
        # Вычисляем коэффициент для i-го базисного полинома
        denominator = 1.0
        for j in range(n):
            if i != j:
                denominator *= (x_points[i] - x_points[j])
        
        coefficient = y_points[i] / denominator
        
        # Получаем коэффициенты для произведения (x-r1)(x-r2)...
        other_roots = [x_points[j] for j in range(n) if j != i]
        basis_coeffs = expand_polynomial_from_roots(other_roots)
        
        # Умножаем на коэффициент и добавляем к общему полиному
        for k in range(len(basis_coeffs)):
            total_coeffs[k] += coefficient * basis_coeffs[k]
    
    # Формируем строку полинома
    polynom_str = 'L(x) ='
    first = True
    
    for i in range(len(total_coeffs)-1, -1, -1):
        coeff = total_coeffs[i]
        if abs(coeff) < 1e-10:  # Пропускаем нулевые коэффициенты
            continue
            
        if first:
            if coeff < 0:
                polynom_str += ' -'
            first = False
        else:
            if coeff >= 0:
                polynom_str += ' +'
            else:
                polynom_str += ' -'
        
        abs_coeff = abs(coeff)
        if i == 0:
            polynom_str += f' {abs_coeff:.4f}'
        elif i == 1:
            if abs_coeff == 1:
                polynom_str += ' x'
            else:
                polynom_str += f' {abs_coeff:.4f}x'
        else:
            if abs_coeff == 1:
                polynom_str += f' x^{i}'
            else:
                polynom_str += f' {abs_coeff:.4f}x^{i}'
    
    return polynom_str


def get_newton_polynomial_str(x_points, y_points):
    """Возвращает строковое представление полинома Ньютона в стандартной форме"""
    n = len(x_points)
    coefs = y_points.copy()
    
    # Вычисление коэффициентов Ньютона
    for i in range(1, n):
        for j in range(n-1, i-1, -1):
            coefs[j] = (coefs[j] - coefs[j-1]) / (x_points[j] - x_points[j-i])
    
    # Начинаем с постоянного члена
    total_coeffs = [coefs[0]]
    
    # Последовательно добавляем члены полинома Ньютона
    for i in range(1, n):
        # Получаем коэффициенты для произведения (x-x0)(x-x1)...(x-x_{i-1})
        roots = x_points[:i]
        basis_coeffs = expand_polynomial_from_roots(roots)
        
        # Умножаем на коэффициент Ньютона и добавляем к общему полиному
        # Расширяем массив коэффициентов если нужно
        while len(total_coeffs) < len(basis_coeffs):
            total_coeffs.append(0.0)
        
        for k in range(len(basis_coeffs)):
            total_coeffs[k] += coefs[i] * basis_coeffs[k]
    
    # Формируем строку полинома
    polynom_str = 'P(x) ='
    first = True
    
    for i in range(len(total_coeffs)-1, -1, -1):
        coeff = total_coeffs[i]
        if abs(coeff) < 1e-10:  # Пропускаем нулевые коэффициенты
            continue
            
        if first:
            if coeff < 0:
                polynom_str += ' -'
            first = False
        else:
            if coeff >= 0:
                polynom_str += ' +'
            else:
                polynom_str += ' -'
        
        abs_coeff = abs(coeff)
        if i == 0:
            polynom_str += f' {abs_coeff:.4f}'
        elif i == 1:
            if abs_coeff == 1:
                polynom_str += ' x'
            else:
                polynom_str += f' {abs_coeff:.4f}x'
        else:
            if abs_coeff == 1:
                polynom_str += f' x^{i}'
            else:
                polynom_str += f' {abs_coeff:.4f}x^{i}'
    
    return polynom_str


def plot_interpolation(x_points, y_points, title):
    """Строит график интерполяции"""
    # Создание точек для построения графика
    x_plot = np.linspace(min(x_points), max(x_points), 300)
    y_true = [f(x) for x in x_plot]
    
    # Вычисление значений полиномов
    y_lagrange = [lagrange_interpolation(x_points, y_points, x) for x in x_plot]
    y_newton = [newton_interpolation(x_points, y_points, x) for x in x_plot]
    
    # Построение графика
    plt.figure(figsize=(12, 8))
    plt.plot(x_plot, y_true, color='gray', linestyle='-', linewidth=2, 
             label='Исходная функция: 1/x²')
    plt.plot(x_plot, y_lagrange, 'r--', linewidth=1.5, label='Полином Лагранжа')
    plt.plot(x_plot, y_newton, 'b:', linewidth=1.5, label='Полином Ньютона')
    plt.scatter(x_points, y_points, c='black', s=100, marker='o', 
                label='Узлы интерполяции')
    
    # Добавление тестовой точки
    x_test = 0.8
    y_test_true = f(x_test)
    y_test_lagrange = lagrange_interpolation(x_points, y_points, x_test)
    y_test_newton = newton_interpolation(x_points, y_points, x_test)
    
    plt.scatter([x_test], [y_test_true], c='magenta', s=100, marker='x',
                label=f'Тестовая точка x={x_test}')
    
    # Настройки графика
    plt.title(f'Интерполяция полиномами Лагранжа и Ньютона\n{title}', fontsize=14)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return x_test, y_test_true, y_test_lagrange, y_test_newton


if __name__ == '__main__':
    # Узлы интерполяции
    x_a = [0.1, 0.5, 0.9, 1.3]
    y_a = [f(x) for x in x_a]
    
    x_b = [0.1, 0.5, 1.1, 1.3]
    y_b = [f(x) for x in x_b]
    
    x_test = 0.8
    
    print('=' * 60)
    print('ИНТЕРПОЛЯЦИЯ ПОЛИНОМАМИ')
    print('=' * 60)
    
    # Точки A
    print('\nТОЧКИ A:', x_a)
    print('-' * 40)
    
    print('Полином Лагранжа:')
    lagrange_str_a = get_lagrange_polynomial_str(x_a, y_a)
    print(lagrange_str_a)
    
    print('\nПолином Ньютона:')
    newton_str_a = get_newton_polynomial_str(x_a, y_a)
    print(newton_str_a)
    
    # Тестовая точка для A
    y_test_true = f(x_test)
    y_lagrange_a = lagrange_interpolation(x_a, y_a, x_test)
    y_newton_a = newton_interpolation(x_a, y_a, x_test)
    
    print(f'\nТестовая точка x = {x_test}:')
    print(f'Истинное значение: y = {y_test_true:.6f}')
    print(f'Лагранж: y = {y_lagrange_a:.6f}, ошибка = {abs(y_lagrange_a - y_test_true):.6f}')
    print(f'Ньютон:  y = {y_newton_a:.6f}, ошибка = {abs(y_newton_a - y_test_true):.6f}')
    
    # Точки B
    print('\n' + '=' * 60)
    print('\nТОЧКИ B:', x_b)
    print('-' * 40)
    
    print('Полином Лагранжа:')
    lagrange_str_b = get_lagrange_polynomial_str(x_b, y_b)
    print(lagrange_str_b)
    
    print('\nПолином Ньютона:')
    newton_str_b = get_newton_polynomial_str(x_b, y_b)
    print(newton_str_b)
    
    # Тестовая точка для B
    y_lagrange_b = lagrange_interpolation(x_b, y_b, x_test)
    y_newton_b = newton_interpolation(x_b, y_b, x_test)
    
    print(f'\nТестовая точка x = {x_test}:')
    print(f'Истинное значение: y = {y_test_true:.6f}')
    print(f'Лагранж: y = {y_lagrange_b:.6f}, ошибка = {abs(y_lagrange_b - y_test_true):.6f}')
    print(f'Ньютон:  y = {y_newton_b:.6f}, ошибка = {abs(y_newton_b - y_test_true):.6f}')
    
    # Построение графиков
    print('\n' + '=' * 60)
    
    plot_interpolation(x_a, y_a, "Точки A")
    plot_interpolation(x_b, y_b, "Точки B")