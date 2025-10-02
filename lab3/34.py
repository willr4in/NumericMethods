def df(x_test, x, y):
    """Вычисляет первую производную в точке x_test методом численного дифференцирования"""
    i = None
    # Находим интервал, в котором находится x_test
    for interval in range(len(x) - 1):
        if x[interval] <= x_test <= x[interval + 1]:
            i = interval
            break

    # Проверка, что точка находится внутри диапазона
    if i is None:
        raise ValueError("x* должен находиться в пределах диапазона x.")

    # Проверка граничных условий
    if i == len(x) - 2:  # Последний интервал - используем левую разность
        return (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    elif i == 0:  # Первый интервал - используем правую разность  
        return (y[i + 1] - y[i]) / (x[i + 1] - x[i])
    else:
        # Интерполяционный метод для внутренних точек
        a = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        b = ((y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - a) / (x[i + 2] - x[i]) * (2 * x_test - x[i] - x[i + 1])
        return a + b


def ddf(x_test, x, y):
    """Вычисляет вторую производную в точке x_test методом численного дифференцирования"""
    i = None
    # Находим интервал, в котором находится x_test
    for interval in range(len(x) - 1):
        if x[interval] <= x_test <= x[interval + 1]:
            i = interval
            break

    # Проверка, что точка находится внутри диапазона
    if i is None:
        raise ValueError("x* должен находиться в пределах диапазона x.")

    # Проверка граничных условий
    if i == 0 or i == len(x) - 2:
        raise ValueError("Вторая производная не может быть вычислена для граничных точек этим методом")
    else:
        # Интерполяционный метод для второй производной
        numerator = (y[i + 2] - y[i + 1]) / (x[i + 2] - x[i + 1]) - (y[i + 1] - y[i]) / (x[i + 1] - x[i])
        return 2 * numerator / (x[i + 2] - x[i])


if __name__ == '__main__':
    # Исходные данные
    x = [1.0, 1.2, 1.4, 1.6, 1.8]
    y = [2.0, 2.1344, 2.4702, 2.9506, 3.5486]
    x_test = 1.4  # Точка, в которой вычисляем производные

    try:
        # Вычисление и вывод первой производной
        print('Первая производная:')
        result_df = df(x_test, x, y)
        print(f'df({x_test}) = {result_df}')

        # Вычисление и вывод второй производной
        print('Вторая производная:')
        result_ddf = ddf(x_test, x, y)
        print(f'ddf({x_test}) = {result_ddf}')
    except (ValueError, IndexError) as e:
        print(f"Ошибка: {e}")