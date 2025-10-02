def f(x):
    """Вычисляет значение подынтегральной функции"""
    return 16 - x**2


def integrate_by_rectangle_method(func, a, b, h):
    """
    Вычисляет интеграл функции func на интервале [a, b]
    методом прямоугольников с шагом step_size.
    
    Параметры:
        func: подынтегральная функция
        a: левая граница интегрирования
        b: правая граница интегрирования
        h: размер шага интегрирования
    
    Возвращает:
        Численное значение интеграла
    """
    integral_value = 0.0
    current_x = a
    
    while current_x < b:
        # Вычисляем значение в средней точке прямоугольника
        midpoint = (current_x + current_x + h) * 0.5
        integral_value += h * func(midpoint)
        current_x += h
        
    return integral_value


def integrate_by_trapezoid_method(func, a, b, h):
    """
    Вычисляет интеграл функции func на интервале [a, b]
    методом трапеций с шагом h.
    """
    integral_value = 0.0
    current_x = a
    
    while current_x < b:
        # Вычисляем площадь трапеции между current_x и current_x + step_size
        integral_value += h * 0.5 * (func(current_x + h) + func(current_x))
        current_x += h
        
    return integral_value


def integrate_by_simpson_method(func, a, b, h):
    """
    Вычисляет интеграл функции func на интервале [a, b]
    методом Симпсона с шагом h.
    """
    n = int((b - a) / h)
    if n % 2 != 0:
        n += 1  # Делаем количество узлов четным
    
    integral_value = func(a) + func(b)
    
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            integral_value += 2 * func(x)
        else:
            integral_value += 4 * func(x)
            
    return integral_value * h / 3


def apply_runge_romberg_rule(step_size_big, step_size_small, integral_big, integral_small, convergence_order):
    """
    Уточняет значение интеграла по методу Рунге-Ромберга.
    
    Параметры:
        step_size_big: первый шаг интегрирования (больший)
        step_size_small: второй шаг интегрирования (меньший)
        integral_big: значение интеграла с шагом step_size_big
        integral_small: значение интеграла с шагом step_size_small
        convergence_order: порядок сходимости метода
    
    Возвращает:
        Уточненное значение интеграла
    """
    return integral_big + (integral_big - integral_small) / ((step_size_small / step_size_big) ** convergence_order - 1)


if __name__ == '__main__':
    # Параметры интегрирования
    INTEGRATION_START = -2
    INTEGRATION_END = 2
    STEP_SMALL = 1.0
    STEP_LARGE = 0.5
    
    # Метод прямоугольников
    print('Метод прямоугольников:')
    rect_large_step = integrate_by_rectangle_method(f, INTEGRATION_START, INTEGRATION_END, STEP_LARGE)
    rect_small_step = integrate_by_rectangle_method(f, INTEGRATION_START, INTEGRATION_END, STEP_SMALL)
    print(f'Шаг = {STEP_LARGE}: интеграл = {rect_large_step}')
    print(f'Шаг = {STEP_SMALL}: интеграл = {rect_small_step}')

    # Метод трапеций
    print('\nМетод трапеций:')
    trap_large_step = integrate_by_trapezoid_method(f, INTEGRATION_START, INTEGRATION_END, STEP_LARGE)
    trap_small_step = integrate_by_trapezoid_method(f, INTEGRATION_START, INTEGRATION_END, STEP_SMALL)
    print(f'Шаг = {STEP_LARGE}: интеграл = {trap_large_step}')
    print(f'Шаг = {STEP_SMALL}: интеграл = {trap_small_step}')

    # Метод Симпсона
    print('\nМетод Симпсона:')
    simp_large_step = integrate_by_simpson_method(f, INTEGRATION_START, INTEGRATION_END, STEP_LARGE)
    simp_small_step = integrate_by_simpson_method(f, INTEGRATION_START, INTEGRATION_END, STEP_SMALL)
    print(f'Шаг = {STEP_LARGE}: интеграл = {simp_large_step}')
    print(f'Шаг = {STEP_SMALL}: интеграл = {simp_small_step}')

    # Метод Рунге-Ромберга для уточнения результатов
    print('\nУточнение методом Рунге-Ромберга:')
    print(f'Уточненный интеграл (прямоугольники): {apply_runge_romberg_rule(STEP_LARGE, STEP_SMALL, rect_large_step, rect_small_step, 2)}')
    print(f'Уточненный интеграл (трапеции): {apply_runge_romberg_rule(STEP_LARGE, STEP_SMALL, trap_large_step, trap_small_step, 2)}')
    print(f'Уточненный интеграл (Симпсон): {apply_runge_romberg_rule(STEP_LARGE, STEP_SMALL, simp_large_step, simp_small_step, 4)}')

    