import matplotlib.pyplot as plt
import math
import numpy as np

def f(x):
    return x ** 6 - 5 * x - 2

def df(x):
    return 6 * x ** 5 - 5

def ddf(x):
    return 30 * x ** 4

def phi(x):
    return (5 * x + 2) ** (1 / 6)

def dphi(x):
    return (5 / 6) * (5 * x + 2) ** (-5 / 6)

def check_conditions(l0, r0):
    """Проверка условий сходимости"""
    result = []
    x0 = (l0 + r0) / 2  # Начальное приближение
    
    # 1. Проверка существования корня
    f_l0, f_r0 = f(l0), f(r0)
    if f_l0 * f_r0 < 0:
        result.append("На интервале есть корень (теорема Больцано-Коши)")
        result.append(f"  f({l0}) = {f_l0:.3f}, f({r0}) = {f_r0:.3f}")
    else:
        result.append("Не гарантировано существование корня на интервале")
        result.append(f"  f({l0}) = {f_l0:.3f}, f({r0}) = {f_r0:.3f}")
    
    # 2. Проверка монотонности
    x_vals = np.linspace(l0, r0, 1000)
    df_vals = df(x_vals)
    if np.all(df_vals > 0) or np.all(df_vals < 0):
        result.append("Функция монотонна на интервале")
    else:
        result.append("Функция не монотонна на интервале")
    
    # 3. Проверка условия сходимости метода Ньютона (для начальной точки)
    f_x0, ddf_x0 = f(x0), ddf(x0)
    newton_condition = f_x0 * ddf_x0
    if newton_condition > 0:
        result.append(f"Выполнено условие сходимости метода Ньютона")
        result.append(f"  f(x₀)·f''(x₀) = {f_x0:.1f}·{ddf_x0:.1f} = {newton_condition:.1f} > 0")
    else:
        result.append(f"Не выполнено условие сходимости метода Ньютона")
        result.append(f"  f(x₀)·f''(x₀) = {f_x0:.1f}·{ddf_x0:.1f} = {newton_condition:.1f} ≤ 0")
    
    # 4. Проверка сходимости метода простой итерации
    # Проверка, что φ(x) отображает отрезок в себя
    phi_vals = []
    for x in x_vals:
        if (5 * x + 2) > 0:  # Проверка области определения
            phi_val = phi(x)
            phi_vals.append(phi_val)
    
    if phi_vals:
        phi_min, phi_max = min(phi_vals), max(phi_vals)
        if l0 <= phi_min and phi_max <= r0:
            result.append("φ(x) отображает отрезок в себя")
            result.append(f"  φ([{l0},{r0}]) ⊆ [{phi_min:.3f}, {phi_max:.3f}] ⊆ [{l0},{r0}]")
        else:
            result.append("φ(x) не отображает отрезок в себя")
            result.append(f"  φ([{l0},{r0}]) = [{phi_min:.3f}, {phi_max:.3f}] не содержится в [{l0},{r0}]")
    else:
        result.append("φ(x) не определена на интервале")
    
    # 5. Проверка условия Липшица для метода простой итерации
    dphi_vals = []
    for x in x_vals:
        if (5 * x + 2) > 0:  # Проверка области определения
            dphi_val = abs(dphi(x))
            dphi_vals.append(dphi_val)
    
    if dphi_vals:
        q = max(dphi_vals)
        if q < 1:
            result.append(f"Выполнено условие сходимости метода простой итерации")
            result.append(f"  q = max|φ'(x)| = {q:.3f} < 1")
        else:
            result.append(f"Не выполнено условие сходимости метода простой итерации")
            result.append(f"  q = max|φ'(x)| = {q:.3f} ≥ 1")
    
    return "\n".join(result)

def simple_iteration(eps, intervals, max_iter=100):
    """Метод простой итерации с улучшенной обработкой ошибок"""
    l0, r0 = intervals
    x0 = (l0 + r0) / 2
    x = x0
    errors = []
    
    print(f"\nМЕТОД ПРОСТОЙ ИТЕРАЦИИ:")
    print(f"Начальное приближение: x₀ = {x0}")
    
    for i in range(max_iter):
        try:
            # Проверка области определения
            if 5 * x + 2 <= 0:
                return None, None, None, "Область определения нарушена: 5x + 2 ≤ 0"
            
            x_next = phi(x)
            
            # Проверка выхода за границы интервала
            if not (l0 <= x_next <= r0):
                return None, None, None, f"Выход за границы интервала: x = {x_next:.6f}"
            
            error = abs(x_next - x)
            errors.append(error)
            
            print(f"Итерация {i+1}: x = {x:.8f}, φ(x) = {x_next:.8f}, Δx = {error:.2e}")
            
            if error < eps:
                return x_next, i + 1, errors, "Успешно"
            
            x = x_next
            
        except (ValueError, ZeroDivisionError) as e:
            return None, None, None, f"Ошибка вычислений: {e}"
    
    return None, None, None, "Превышено максимальное число итераций"

def newton_method(eps, intervals, max_iter=100):
    """Метод Ньютона с улучшенной обработкой ошибок"""
    l0, r0 = intervals
    x0 = (l0 + r0) / 2
    x = x0
    errors = []
    
    print(f"\nМЕТОД НЬЮТОНА:")
    print(f"Начальное приближение: x₀ = {x0}")
    print(f"f(x₀) = {f(x0):.3f}, f'(x₀) = {df(x0):.3f}, f''(x₀) = {ddf(x0):.3f}")
    print(f"f(x₀)·f''(x₀) = {f(x0)*ddf(x0):.3f}")
    
    for i in range(max_iter):
        try:
            fx = f(x)
            dfx = df(x)
            
            # Проверка производной
            if abs(dfx) < 1e-12:
                return None, None, None, "Производная близка к нулю"
            
            x_next = x - fx / dfx
            error = abs(x_next - x)
            errors.append(error)
            
            print(f"Итерация {i+1}: x = {x:.8f}, f(x) = {fx:.2e}, f'(x) = {dfx:.3f}, Δx = {error:.2e}")
            
            if error < eps:
                return x_next, i + 1, errors, "Успешно"
            
            x = x_next
            
        except (ValueError, ZeroDivisionError) as e:
            return None, None, None, f"Ошибка вычислений: {e}"
    
    return None, None, None, "Превышено максимальное число итераций"

def plot_function_and_convergence(l0, r0, errors1=None, errors2=None):
    """Построение графиков функции и сходимости"""
    plt.figure(figsize=(15, 5))
    
    # График функции
    plt.subplot(1, 3, 1)
    x_vals = np.linspace(l0, r0, 400)
    y_vals = f(x_vals)
    
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x) = x⁶ - 5x - 2')
    plt.axhline(0, color='red', linestyle='--', alpha=0.7, label='y = 0')
    plt.axvline(1.448678, color='green', linestyle=':', alpha=0.7, label='Найденный корень')
    
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # График производной
    plt.subplot(1, 3, 2)
    df_vals = df(x_vals)
    
    plt.plot(x_vals, df_vals, 'g-', linewidth=2, label="f'(x) = 6x⁵ - 5")
    plt.axhline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('x')
    plt.ylabel("f'(x)")
    plt.title('График производной')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Графики сходимости
    plt.subplot(1, 3, 3)
    if errors1:
        plt.plot(range(1, len(errors1) + 1), errors1, 'o-', 
                label='Метод простой итерации', linewidth=2, markersize=6)
    if errors2:
        plt.plot(range(1, len(errors2) + 1), errors2, 's-', 
                label='Метод Ньютона', linewidth=2, markersize=6)
    
    plt.yscale('log')
    plt.xlabel("Номер итерации")
    plt.ylabel("Погрешность (Δx)")
    plt.title("Сравнение сходимости методов")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    eps = 1e-6
    l0, r0 = 1.0, 2.0
    
    print("=" * 70)
    print("РЕШЕНИЕ УРАВНЕНИЯ: x⁶ - 5x - 2 = 0")
    print("=" * 70)
    print(f"Интервал: [{l0}, {r0}]")
    print(f"Точность: {eps}")
    
    print("\n" + "=" * 70)
    print("ПРОВЕРКА УСЛОВИЙ СХОДИМОСТИ:")
    print("=" * 70)
    conditions = check_conditions(l0, r0)
    print(conditions)
    
    # Вычисление методов
    print("\n" + "=" * 70)
    print("ВЫЧИСЛЕНИЕ КОРНЯ:")
    print("=" * 70)
    
    x1, it1, errors1, status1 = simple_iteration(eps, (l0, r0))
    x2, it2, errors2, status2 = newton_method(eps, (l0, r0))
    
    
    # Построение графиков
    plot_function_and_convergence(l0, r0, errors1, errors2)

if __name__ == "__main__":
    main()