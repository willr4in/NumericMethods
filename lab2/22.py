import numpy as np
import matplotlib.pyplot as plt
import sys
import copy  # Добавляем import copy
from LU import LU_decompose, solve_system

def f(x):
    return np.array([
        x[0] ** 2 - x[0] + x[1] ** 2 - 1,
        x[1] - np.tan(x[0])
    ])


def f_der(x):
    return np.array([
        [2 * x[0] - 1, 2 * x[1]],
        [-(1.0 / (np.cos(x[0]) ** 2)), 1.0]
    ])


def phi(x):
    x1_next = np.arctan(x[1])
    expr = 1 - x[0] ** 2 + x[0]
    x2_next = np.sqrt(expr) if expr >= 0 else np.nan
    return np.array([x1_next, x2_next])

def phi_der(x):
    x1, x2 = x
    # φ₁(x) = arctan(x₂)
    dphi1_dx1 = 0.0
    dphi1_dx2 = 1 / (1 + x2 ** 2)
    
    # φ₂(x) = sqrt(1 - x₁² + x₁)
    expr = 1 - x1**2 + x1
    if expr > 1e-12:
        dphi2_dx1 = (1 - 2*x1) / (2 * np.sqrt(expr))
    else:
        dphi2_dx1 = np.nan
    dphi2_dx2 = 0.0
    
    return np.array([
        [dphi1_dx1, dphi1_dx2],
        [dphi2_dx1, dphi2_dx2]
    ])


def is_within_interval(x, interval):
    x1_min, x1_max, x2_min, x2_max = interval
    return x1_min <= x[0] <= x1_max and x2_min <= x[1] <= x2_max


# Метод Ньютона
def newton(x0, eps, interval):
    xPrev = np.copy(x0)
    iter = 0
    print("\n--- Метод Ньютона ---")
    print(f"Начальное приближение: {xPrev}")

    if not is_within_interval(xPrev, interval):
        print("ОШИБКА: Начальное приближение находится вне указанного интервала.")
        return None, iter

    jac_at_x0 = f_der(xPrev)
    det_jac_at_x0 = np.linalg.det(jac_at_x0)
    if np.isclose(det_jac_at_x0, 0, atol=1e-9):
        print("ПРЕДУПРЕЖДЕНИЕ: Якобиан в начальной точке близок к сингулярному. Метод может быть неустойчивым.")

    while (True):
        iter += 1
        jac_f = f_der(xPrev)
        
        try:
            L, U = LU_decompose(jac_f.tolist())  # Преобразуем numpy array в list
            delta_x_list = solve_system(L, U, (-f(xPrev)).tolist())
            delta_x = np.array(delta_x_list)
        except ValueError as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Проблема с решением системы на итерации {iter}: {e}")
            return xPrev, iter

        xCur = xPrev + delta_x

        if not is_within_interval(xCur, interval):
            print(f"ИНФОРМАЦИЯ: Итерация {iter}: [{xCur[0]:.6f}, {xCur[1]:.6f}] вышла за пределы интервала поиска.")
            return None, iter

        error = np.linalg.norm(xCur - xPrev, np.inf)

        if error < eps:
            print(f"Критерий останова (||x(k+1) - x(k)|| = {error:.6e} < {eps}) выполнен.")
            break
        xPrev = np.copy(xCur)

        if iter > 200:
            print("ПРЕДУПРЕЖДЕНИЕ: Достигнуто максимальное количество итераций.")
            break

    print(f"Корень найден в пределах интервала за {iter} итераций.")
    return xCur, iter


# Метод простых итераций с проверкой нахождения в интервале и условием сходимости
def simple_iterations(x0, interval, eps):
    xPrev = np.copy(x0)
    iter = 0
    print("\n--- Метод простых итераций ---")
    print(f"Начальное приближение: {xPrev}")

    if not is_within_interval(xPrev, interval):
        print("ОШИБКА: Начальное приближение находится вне указанного интервала.")
        return None, iter

    x1_min, x1_max, x2_min, x2_max = interval

    # Проверка допустимости интервала для x1
    x0_samples = np.linspace(x1_min, x1_max, 1000)
    expr = 1 - x0_samples ** 2 + x0_samples
    if np.any(expr < 0):
        print("ОШИБКА: В интервале x1 есть значения, где 1 -x1² +x1 < 0. Метод неприменим.")
        return None, iter

    # Вычисление max_df1_dx2
    max_df1_dx2 = 1.0 / (1 + x2_min ** 2)

    # Вычисление max_df2_dx0
    df2_dx0 = np.abs((-2 * x0_samples + 1) / (2 * np.sqrt(1 - x0_samples ** 2 + x0_samples + 1e-12)))
    max_df2_dx0 = np.max(df2_dx0)

    q_calculated = max(max_df1_dx2, max_df2_dx0)
    print(f"Расчетное значение q (max ||phi'(x)||inf): {q_calculated:.6f}")

    # if q_calculated >= 1:
    #     print("ПРЕДУПРЕЖДЕНИЕ: Условие сходимости ||phi'(x)|| <= q < 1 не выполнено в данной области.")

    while True:
        iter += 1
        try:
            xCur = phi(xPrev)
        except:
            print("Ошибка при вычислении phi(x). Проверьте допустимость значений.")
            return None, iter

        if not is_within_interval(xCur, interval):
            print(f"ИНФОРМАЦИЯ: Итерация {iter}: [{xCur[0]:.6f}, {xCur[1]:.6f}] вышла за пределы интервала поиска.")
            return None, iter

        if q_calculated < 1:
            error_apriori = (q_calculated ** iter) / (1 - q_calculated) * np.linalg.norm(xCur - x0, np.inf)
            error_check = error_apriori
            error_type = "априорная оценка ошибки"
        else:
            error_aposteriori = np.linalg.norm(xCur - xPrev, np.inf)
            error_check = error_aposteriori
            error_type = "апостериорная ошибка (норма разности)"

        if error_check < eps:
            print(f"Критерий останова ({error_type} {error_check:.6e} < {eps}) выполнен.")
            break

        xPrev = np.copy(xCur)

        if iter > 1000:
            print("ПРЕДУПРЕЖДЕНИЕ: Достигнуто максимальное количество итераций.")
            break

    print(f"Корень найден в пределах интервала за {iter} итераций.")
    return xCur, iter


# --- Основная часть скрипта ---

print("Введите границы интервала для поиска корня [x1_min, x1_max] x [x2_min, x2_max]:")
try:
    x1_min = float(input("x1_min: "))
    x1_max = float(input("x1_max: "))
    x2_min = float(input("x2_min: "))
    x2_max = float(input("x2_max: "))
except ValueError:
    print("Ошибка ввода: введите корректные числа.")
    sys.exit()

if x1_min >= x1_max or x2_min >= x2_max:
    print("Ошибка: Некорректные границы интервала.")
    sys.exit()

print("\nВведите начальное приближение (x0_1, x0_2):")
try:
    x0_1 = float(input("x0_1: "))
    x0_2 = float(input("x0_2: "))
    x0 = np.array([x0_1, x0_2])
except ValueError:
    print("Ошибка ввода: введите корректные числа.")
    sys.exit()

interval = (x1_min, x1_max, x2_min, x2_max)

if not is_within_interval(x0, interval):
    print("ОШИБКА: Начальное приближение", x0, "находится вне указанного интервала [", x1_min, ",", x1_max, "] x [",
          x2_min, ",", x2_max, "].")
    sys.exit()

try:
    eps = float(input("\nВведите требуемую точность (eps > 0): "))
    if eps <= 0:
        print("Ошибка: Точность должна быть положительным числом.")
        sys.exit()
except ValueError:
    print("Ошибка ввода: введите корректное число.")
    sys.exit()

# Метод Ньютона
newtonAns, newton_iter = newton(x0, eps, interval)
if newtonAns is not None:
    print("\n======================================")
    print("Результаты метода Ньютона (найден в области):")
    print(f"\tНайденный корень: [{newtonAns[0]:.6f}, {newtonAns[1]:.6f}]")
    print(f"\tКоличество итераций: {newton_iter}")
    print(f"\tЗначение f(корень): [{f(newtonAns)[0]:.6e}, {f(newtonAns)[1]:.6e}]")
    print("======================================")
else:
    print("\n======================================")
    print("Метод Ньютона: Корень не найден в указанной области или метод прерван.")
    print("======================================")

# Метод простых итераций
simpleIterationsAns, simple_iter = simple_iterations(x0, interval, eps)
if simpleIterationsAns is not None:
    print("\n======================================")
    print("Результаты метода простых итераций (найден в области):")
    print(f"\tНайденный корень: [{simpleIterationsAns[0]:.6f}, {simpleIterationsAns[1]:.6f}]")
    print(f"\tКоличество итераций: {simple_iter}")
    print(f"\tЗначение f(корень): [{f(simpleIterationsAns)[0]:.6e}, {f(simpleIterationsAns)[1]:.6e}]")
    print("======================================")
else:
    print("\n======================================")
    print("Метод простых итераций: Корень не найден в указанной области или метод прерван.")
    print("======================================")

if newtonAns is not None:
    f_at_newton_root = f(newtonAns)
    norm_f_newton = np.linalg.norm(f_at_newton_root, np.inf)
    check_passed_newton = norm_f_newton <= eps
    print("\n--- Проверка решения (Метод Ньютона) ---")
    print(f"Норма невязки (||f(x*)||inf): {norm_f_newton:.6e}")
    print(f"Проверка <= eps ({eps}): {check_passed_newton}")
    print("-----------------------------------------")
else:
    print("\n--- Проверка решения (Метод Ньютона) ---")
    print("Решение методом Ньютона не было найдено в заданной области.")
    print("-----------------------------------------")

if simpleIterationsAns is not None:
    f_at_simple_root = f(simpleIterationsAns)
    norm_f_simple = np.linalg.norm(f_at_simple_root, np.inf)
    check_passed_simple = norm_f_simple <= eps
    print("\n--- Проверка решения (Метод простых итераций) ---")
    print(f"Норма невязки (||f(x*)||inf): {norm_f_simple:.6e}")
    print(f"Проверка <= eps ({eps}): {check_passed_simple}")
    print("-----------------------------------------")
else:
    print("\n--- Проверка решения (Метод простых итераций) ---")
    print("Решение методом простых итераций не было найдено в заданной области.")
    print("-----------------------------------------")

# Расширяем диапазон для построения графиков, чтобы было видно кривые вокруг интервала
plot_x_min = min(x1_min, x0[0]) - 0.2
plot_x_max = max(x1_max, x0[0]) + 0.2
plot_y_min = min(x2_min, x0[1]) - 0.2
plot_y_max = max(x2_max, x0[1]) + 0.2

# Если корни найдены в пределах области, убедимся, что они попадают в диапазон графика
if newtonAns is not None:
    plot_x_min = min(plot_x_min, newtonAns[0] - 0.1)
    plot_x_max = max(plot_x_max, newtonAns[0] + 0.1)
    plot_y_min = min(plot_y_min, newtonAns[1] - 0.1)
    plot_y_max = max(plot_y_max, newtonAns[1] + 0.1)

if simpleIterationsAns is not None:
    plot_x_min = min(plot_x_min, simpleIterationsAns[0] - 0.1)
    plot_x_max = max(plot_x_max, simpleIterationsAns[0] + 0.1)
    plot_y_min = min(plot_y_min, simpleIterationsAns[1] - 0.1)
    plot_y_max = max(plot_y_max, simpleIterationsAns[1] + 0.1)

x_range = np.linspace(plot_x_min, plot_x_max, 400)
y_range = np.linspace(plot_y_min, plot_y_max, 400)

# Для первой кривой: x1^2 - x1 + x2^2 - 1 = 0
# x2 = ±sqrt(1 - x1^2 + x1)
x_vals_eq1 = np.linspace(plot_x_min, plot_x_max, 400)
y_vals_eq1_pos = np.sqrt(1 - x_vals_eq1 ** 2 + x_vals_eq1)
y_vals_eq1_neg = -np.sqrt(1 - x_vals_eq1 ** 2 + x_vals_eq1)

# Для второй кривой: x2 - tg(x1) = 0
# x2 = tg(x1)
x_vals_eq2 = np.linspace(plot_x_min, plot_x_max, 400)
y_vals_eq2 = np.tan(x_vals_eq2)
y_vals_eq2[:-1][np.diff(y_vals_eq2) < 0] = np.nan

plt.figure(figsize=(8, 6))

# Рисуем кривые
plt.plot(x_vals_eq1, y_vals_eq1_pos, label='$x_1^2 - x_1 + x_2^2 - 1 = 0$ (верхняя)', color='blue')
plt.plot(x_vals_eq1, y_vals_eq1_neg, label='$x_1^2 - x_1 + x_2^2 - 1 = 0$ (нижняя)', color='blue', linestyle='--')
plt.plot(x_vals_eq2, y_vals_eq2, label='$x_2 - \\tan(x_1) = 0$', color='red')

if newtonAns is not None:
    plt.scatter(newtonAns[0], newtonAns[1], color='green', zorder=5, label='Корень (Ньютон)', s=50)
if simpleIterationsAns is not None:
    plt.scatter(simpleIterationsAns[0], simpleIterationsAns[1], color='purple', zorder=5,
                label='Корень (Простые итерации)', s=50, marker='X')

plt.scatter(x0[0], x0[1], color='orange', zorder=5, label='Начальное приближение', s=50, marker='*')

plt.plot([x1_min, x1_max, x1_max, x1_min, x1_min], [x2_min, x2_min, x2_max, x2_max, x2_min], color='gray',
         linestyle='--', label='Область поиска')

plt.title('Графики функций системы и ее корни')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.grid(True)
plt.legend()
plt.axis('equal')
plt.xlim(plot_x_min, plot_x_max)
plt.ylim(plot_y_min, plot_y_max)

plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)

# Вместо сохранения в файл выводим график на экран
plt.show()