import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

class EllipticEquationSolver:
    def __init__(self, Lx=np.pi, Ly=1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.analytic_solution = lambda x, y: np.sin(x) * np.exp(y)
    
    def exact_solution(self, x, y):
        return np.sin(x) * np.exp(y)
    
    def solve_jacobi(self, Nx, Ny, bc_type='two_point_first_order', max_iter=10000, tol=1e-8):
        """
        Метод простых итераций (Якоби) с исправленной правой границей
        """
        hx = self.Lx / Nx
        hy = self.Ly / Ny
        r = hx / hy
        r2 = r * r
        
        print(f"Сетка: {Nx}x{Ny}, hx={hx:.4f}, hy={hy:.4f}, r={r:.4f}")
        print(f"Аппроксимация граничных условий: {bc_type}")
        
        x = np.linspace(0, self.Lx, Nx+1)
        y = np.linspace(0, self.Ly, Ny+1)
        u = np.zeros((Nx+1, Ny+1))
        
        # Начальные условия (по y) - ДИРИХЛЕ
        for i in range(Nx+1):
            u[i, 0] = np.sin(x[i])      # u(x, 0) = sin(x)
            u[i, Ny] = np.exp(1) * np.sin(x[i])  # u(x, 1) = e * sin(x)
        
        # Граничные условия (по x) - НЕЙМАНА
        if bc_type == 'two_point_first_order':
            for j in range(Ny+1):
                # u_x(0, y) = exp(y) - впередная разность
                u[0, j] = u[1, j] - hx * np.exp(y[j])
                # u_x(π, y) = -exp(y) - назадная разность
                u[Nx, j] = u[Nx-1, j] - hx * np.exp(y[j])
                
        elif bc_type == 'three_point_second_order':
            for j in range(Ny+1):
                # u_x(0, y) = exp(y) - трехточечная вперед
                u[0, j] = (4*u[1, j] - u[2, j] - 2*hx*np.exp(y[j])) / 3
                # u_x(π, y) = -exp(y) - трехточечная назад
                u[Nx, j] = (4*u[Nx-1, j] - u[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
        
        denominator = 2.0 * (1 + r2)
        errors = []
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            u_new = u.copy()
            
            # Внутренние точки
            for i in range(1, Nx):
                for j in range(1, Ny):
                    u_new[i, j] = (u[i-1, j] + u[i+1, j] + r2 * (u[i, j-1] + u[i, j+1])) / denominator
            
            # Обновляем граничные условия Неймана
            if bc_type == 'two_point_first_order':
                for j in range(1, Ny):
                    u_new[0, j] = u_new[1, j] - hx * np.exp(y[j])
                    u_new[Nx, j] = u_new[Nx-1, j] - hx * np.exp(y[j])
                    
            elif bc_type == 'three_point_second_order':
                for j in range(1, Ny):
                    u_new[0, j] = (4*u_new[1, j] - u_new[2, j] - 2*hx*np.exp(y[j])) / 3
                    u_new[Nx, j] = (4*u_new[Nx-1, j] - u_new[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
            
            # Восстанавливаем начальные условия Дирихле
            for i in range(Nx+1):
                u_new[i, 0] = np.sin(x[i])
                u_new[i, Ny] = np.exp(1) * np.sin(x[i])
            
            error = np.max(np.abs(u_new - u))
            errors.append(error)
            u = u_new
            
            if error < tol:
                execution_time = time.time() - start_time
                print(f"Сходимость за {iteration+1} итераций, погрешность: {error:.2e}")
                print(f"Время выполнения: {execution_time:.4f} секунд")
                break
                
            if iteration % 1000 == 0 and iteration > 0:
                print(f"Итерация {iteration}, погрешность: {error:.2e}")
        
        if iteration == max_iter - 1:
            execution_time = time.time() - start_time
            print(f"Достигнут максимум итераций {max_iter}, погрешность: {error:.2e}")
            print(f"Время выполнения: {execution_time:.4f} секунд")
        
        return x, y, u, 'Метод Якоби', errors, execution_time

    def solve_gauss_seidel(self, Nx, Ny, bc_type='two_point_first_order', max_iter=10000, tol=1e-8):
        """Метод Зейделя"""
        hx = self.Lx / Nx
        hy = self.Ly / Ny
        r = hx / hy
        r2 = r * r
        
        print(f"Сетка: {Nx}x{Ny}, hx={hx:.4f}, hy={hy:.4f}, r={r:.4f}")
        print(f"Аппроксимация граничных условий: {bc_type}")
        
        x = np.linspace(0, self.Lx, Nx+1)
        y = np.linspace(0, self.Ly, Ny+1)
        u = np.zeros((Nx+1, Ny+1))
        
        # Начальные и граничные условия
        for i in range(Nx+1):
            u[i, 0] = np.sin(x[i])
            u[i, Ny] = np.exp(1) * np.sin(x[i])
        
        if bc_type == 'two_point_first_order':
            for j in range(Ny+1):
                u[0, j] = u[1, j] - hx * np.exp(y[j])
                u[Nx, j] = u[Nx-1, j] - hx * np.exp(y[j])
                
        elif bc_type == 'three_point_second_order':
            for j in range(Ny+1):
                u[0, j] = (4*u[1, j] - u[2, j] - 2*hx*np.exp(y[j])) / 3
                u[Nx, j] = (4*u[Nx-1, j] - u[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
                
        denominator = 2.0 * (1 + r2)
        errors = []
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            u_old = u.copy()
            max_error = 0.0
            
            for i in range(1, Nx):
                for j in range(1, Ny):
                    old_val = u[i, j]
                    u[i, j] = (u[i-1, j] + u_old[i+1, j] + r2 * (u[i, j-1] + u_old[i, j+1])) / denominator
                    max_error = max(max_error, abs(u[i, j] - old_val))
            
            # Обновляем границы
            if bc_type == 'two_point_first_order':
                for j in range(1, Ny):
                    u[0, j] = u[1, j] - hx * np.exp(y[j])
                    u[Nx, j] = u[Nx-1, j] - hx * np.exp(y[j])
                    
            elif bc_type == 'three_point_second_order':
                for j in range(1, Ny):
                    u[0, j] = (4*u[1, j] - u[2, j] - 2*hx*np.exp(y[j])) / 3
                    u[Nx, j] = (4*u[Nx-1, j] - u[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
            
            # Восстанавливаем Дирихле
            for i in range(Nx+1):
                u[i, 0] = np.sin(x[i])
                u[i, Ny] = np.exp(1) * np.sin(x[i])
            
            errors.append(max_error)
            
            if max_error < tol:
                execution_time = time.time() - start_time
                print(f"Сходимость за {iteration+1} итераций, погрешность: {max_error:.2e}")
                print(f"Время выполнения: {execution_time:.4f} секунд")
                break
                
            if iteration % 1000 == 0 and iteration > 0:
                print(f"Итерация {iteration}, погрешность: {max_error:.2e}")
        
        if iteration == max_iter - 1:
            execution_time = time.time() - start_time
            print(f"Достигнут максимум итераций {max_iter}, погрешность: {max_error:.2e}")
            print(f"Время выполнения: {execution_time:.4f} секунд")
        
        return x, y, u, 'Метод Зейделя', errors, execution_time

    def solve_sor(self, Nx, Ny, bc_type='two_point_first_order', max_iter=10000, tol=1e-8, omega=1.5):
        """Метод верхней релаксации (SOR)"""
        hx = self.Lx / Nx
        hy = self.Ly / Ny
        r = hx / hy
        r2 = r * r
        
        print(f"Сетка: {Nx}x{Ny}, hx={hx:.4f}, hy={hy:.4f}, r={r:.4f}, ω={omega}")
        print(f"Аппроксимация граничных условий: {bc_type}")
        
        x = np.linspace(0, self.Lx, Nx+1)
        y = np.linspace(0, self.Ly, Ny+1)
        u = np.zeros((Nx+1, Ny+1))
        
        # Начальные и граничные условия
        for i in range(Nx+1):
            u[i, 0] = np.sin(x[i])
            u[i, Ny] = np.exp(1) * np.sin(x[i])
        
        if bc_type == 'two_point_first_order':
            for j in range(Ny+1):
                u[0, j] = u[1, j] - hx * np.exp(y[j])
                u[Nx, j] = u[Nx-1, j] - hx * np.exp(y[j])
                
        elif bc_type == 'three_point_second_order':
            for j in range(Ny+1):
                u[0, j] = (4*u[1, j] - u[2, j] - 2*hx*np.exp(y[j])) / 3
                u[Nx, j] = (4*u[Nx-1, j] - u[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
                
        denominator = 2.0 * (1 + r2)
        errors = []
        
        start_time = time.time()
        
        for iteration in range(max_iter):
            u_old = u.copy()
            max_error = 0.0
            
            for i in range(1, Nx):
                for j in range(1, Ny):
                    old_val = u[i, j]
                    u_gs = (u[i-1, j] + u_old[i+1, j] + r2 * (u[i, j-1] + u_old[i, j+1])) / denominator
                    u[i, j] = old_val + omega * (u_gs - old_val)
                    max_error = max(max_error, abs(u[i, j] - old_val))
            
            # Обновляем границы
            if bc_type == 'two_point_first_order':
                for j in range(1, Ny):
                    u[0, j] = u[1, j] - hx * np.exp(y[j])
                    u[Nx, j] = u[Nx-1, j] - hx * np.exp(y[j])
                    
            elif bc_type == 'three_point_second_order':
                for j in range(1, Ny):
                    u[0, j] = (4*u[1, j] - u[2, j] - 2*hx*np.exp(y[j])) / 3
                    u[Nx, j] = (4*u[Nx-1, j] - u[Nx-2, j] - 2*hx*np.exp(y[j])) / 3
            
            # Восстанавливаем Дирихле
            for i in range(Nx+1):
                u[i, 0] = np.sin(x[i])
                u[i, Ny] = np.exp(1) * np.sin(x[i])
            
            errors.append(max_error)
            
            if max_error < tol:
                execution_time = time.time() - start_time
                print(f"Сходимость за {iteration+1} итераций, погрешность: {max_error:.2e}")
                print(f"Время выполнения: {execution_time:.4f} секунд")
                break
                
            if iteration % 1000 == 0 and iteration > 0:
                print(f"Итерация {iteration}, погрешность: {max_error:.2e}")
        
        if iteration == max_iter - 1:
            execution_time = time.time() - start_time
            print(f"Достигнут максимум итераций {max_iter}, погрешность: {max_error:.2e}")
            print(f"Время выполнения: {execution_time:.4f} секунд")
        
        return x, y, u, f'Метод SOR (ω={omega})', errors, execution_time

    def calculate_error(self, u_numeric, u_exact):
        return np.abs(u_numeric - u_exact)
    
    def plot_selected_graphs(self, x, y, u, method_name, bc_type, errors):
        """
        Только выбранные графики:
        1. 3D график численного решения
        2. Срез по x
        3. Сравнение аналитического и численного при конечном y
        4. Зависимость погрешности от итераций
        """
        X, Y = np.meshgrid(x, y, indexing='ij')
        u_exact = self.exact_solution(X, Y)
        error = self.calculate_error(u, u_exact)
        
        # Информация об аппроксимации для заголовка
        approx_info = {
            'two_point_first_order': ' (2-точечная 1-го порядка)',
            'three_point_second_order': ' (3-точечная 2-го порядка)'
        }
        
        fig = plt.figure(figsize=(20, 10))
        
        # 1. 3D график численного решения
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, u, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('u(x,y)')
        ax1.set_title(f'3D: Численное решение\n{method_name}{approx_info[bc_type]}')
        plt.colorbar(surf, ax=ax1, shrink=0.5, aspect=20)
        
        # 2. Срез по x (при разных y)
        plt.subplot(2, 2, 2)
        y_indices = [0, len(y)//4, len(y)//2, 3*len(y)//4, -1]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        labels = [f'y = {y[i]:.2f}' for i in y_indices]
        
        for idx, color, label in zip(y_indices, colors, labels):
            plt.plot(x, u[:, idx], color=color, label=label, linewidth=2)
        
        plt.xlabel('x')
        plt.ylabel('u(x,y)')
        plt.title('Срез по x при разных y')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Сравнение аналитического и численного при конечном y (y = 1)
        plt.subplot(2, 2, 3)
        y_final_index = -1  # y = 1
        plt.plot(x, u[:, y_final_index], 'b-', linewidth=3, label='Численное решение')
        plt.plot(x, u_exact[:, y_final_index], 'r--', linewidth=2, label='Аналитическое решение')
        plt.xlabel('x')
        plt.ylabel('u(x,y)')
        plt.title(f'Сравнение при y = {y[y_final_index]:.2f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Зависимость погрешности от итераций
        plt.subplot(2, 2, 4)
        if errors is not None:
            plt.semilogy(errors, 'r-', linewidth=2)
            plt.xlabel('Номер итерации')
            plt.ylabel('Максимальное изменение')
            plt.title('Сходимость метода')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Вывод информации о погрешности
        max_error = np.max(error)
        mean_error = np.mean(error)
        print(f"\nИНФОРМАЦИЯ О ПОГРЕШНОСТИ:")
        print(f"Максимальная погрешность: {max_error:.2e}")
        print(f"Средняя погрешность: {mean_error:.2e}")

    def plot_comparison_chart(self, results):
        """
        Создает результирующий график сравнения времени выполнения всех методов
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Подготовка данных для графиков
        methods = []
        times_first_order = []
        times_second_order = []
        iterations_first_order = []
        iterations_second_order = []
        
        for result in results:
            method_name = result['method']
            bc_type = result['bc_type']
            execution_time = result['execution_time']
            iterations = result['iterations']
            
            if method_name not in methods:
                methods.append(method_name)
            
            if bc_type == 'two_point_first_order':
                times_first_order.append(execution_time)
                iterations_first_order.append(iterations)
            else:
                times_second_order.append(execution_time)
                iterations_second_order.append(iterations)
        
        # График времени выполнения
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, times_first_order, width, label='2-точечная 1-го порядка', 
                       color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, times_second_order, width, label='3-точечная 2-го порядка', 
                       color='lightcoral', alpha=0.8)
        
        ax1.set_xlabel('Метод решения')
        ax1.set_ylabel('Время выполнения (секунды)')
        ax1.set_title('Сравнение времени выполнения методов')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        # График количества итераций
        bars3 = ax2.bar(x - width/2, iterations_first_order, width, label='2-точечная 1-го порядка', 
                       color='lightgreen', alpha=0.8)
        bars4 = ax2.bar(x + width/2, iterations_second_order, width, label='3-точечная 2-го порядка', 
                       color='gold', alpha=0.8)
        
        ax2.set_xlabel('Метод решения')
        ax2.set_ylabel('Количество итераций')
        ax2.set_title('Сравнение количества итераций методов')
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Добавление значений на столбцы
        for bar in bars3:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        for bar in bars4:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Вывод сводной таблицы
        print(f"\n{'='*80}")
        print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
        print(f"{'='*80}")
        print(f"{'Метод':<20} {'Аппроксимация':<25} {'Время (с)':<12} {'Итерации':<10} {'Эффективность':<15}")
        print(f"{'-'*80}")
        
        for i, method in enumerate(methods):
            time1 = times_first_order[i]
            time2 = times_second_order[i]
            iter1 = iterations_first_order[i]
            iter2 = iterations_second_order[i]
            efficiency1 = iter1 / time1 if time1 > 0 else 0
            efficiency2 = iter2 / time2 if time2 > 0 else 0
            
            print(f"{method:<20} {'2-точечная 1-го порядка':<25} {time1:.4f}     {iter1:<10} {efficiency1:.1f} итер/с")
            print(f"{method:<20} {'3-точечная 2-го порядка':<25} {time2:.4f}     {iter2:<10} {efficiency2:.1f} итер/с")
            print(f"{'-'*80}")

def main():
    solver = EllipticEquationSolver()
    
    print("=" * 70)
    print("РЕШЕНИЕ ЭЛЛИПТИЧЕСКОГО УРАВНЕНИЯ")
    print("Уравнение: d²u/dx² + d²u/dy² = 0")
    print("Граничные условия: u_x(0,y) = exp(y), u_x(π,y) = -exp(y)")
    print("Начальные условия: u(x,0) = sin(x), u(x,1) = e*sin(x)")
    print("Аналитическое решение: u(x,y) = sin(x)*exp(y)")
    print("=" * 70)
    
    # Параметры сетки
    Nx, Ny = 40, 40
    
    methods = [
        ('Метод Якоби', solver.solve_jacobi),
        ('Метод Зейделя', solver.solve_gauss_seidel),
        ('Метод SOR (ω=1.5)', solver.solve_sor),
    ]
    
    # Собираем результаты для сравнения
    results = []
    
    # Тестируем оба типа аппроксимации
    bc_types = ['two_point_first_order', 'three_point_second_order']
    
    for bc_type in bc_types:
        for method_name, method_func in methods:
            print(f"\n{'='*60}")
            print(f"МЕТОД: {method_name}")
            print(f"АППРОКСИМАЦИЯ ГРАНИЧНЫХ УСЛОВИЙ: {bc_type}")
            print(f"{'='*60}")
            
            try:
                if 'SOR' in method_name:
                    x, y, u, name, errors, exec_time = method_func(Nx, Ny, bc_type, max_iter=5000, omega=1.5)
                else:
                    x, y, u, name, errors, exec_time = method_func(Nx, Ny, bc_type, max_iter=5000)
                
                # Построение выбранных графиков
                solver.plot_selected_graphs(x, y, u, method_name, bc_type, errors)
                
                # Сохраняем результаты для сравнения
                results.append({
                    'method': method_name,
                    'bc_type': bc_type,
                    'execution_time': exec_time,
                    'iterations': len(errors) if errors else 0
                })
                
            except Exception as e:
                print(f"Ошибка: {e}")
                import traceback
                traceback.print_exc()
    
    # Строим результирующий график сравнения
    if results:
        solver.plot_comparison_chart(results)

if __name__ == "__main__":
    main()