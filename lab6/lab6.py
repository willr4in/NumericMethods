import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time
from mpl_toolkits.mplot3d import Axes3D

class HyperbolicEquationSolver:
    def __init__(self, L=1.0, T=2.0):
        self.L = L
        self.T = T
        self.analytic_solution = lambda x, t: np.exp(2*x) * np.cos(t)
    
    def exact_solution(self, x, t):
        return np.exp(2*x) * np.cos(t)
    
    def solve_explicit(self, Nx, Nt, bc_type='two_point_first_order', initial_approx='first_order'):
        """
        Явная схема для гиперболического уравнения
        """
        dx = self.L / Nx
        dt = self.T / Nt
        r = dt / dx
        r2 = r * r
        
        # Проверка устойчивости (условие Куранта)
        stability_limit = 1.0 / np.sqrt(1 + 5*dx*dx)
        if r > stability_limit:
            print(f"ПРЕДУПРЕЖДЕНИЕ: r = {r:.3f} > {stability_limit:.3f} → МОЖЕТ БЫТЬ НЕУСТОЙЧИВО")
        else:
            print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r:.3f} ≤ {stability_limit:.3f} → УСТОЙЧИВО")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt+1)
        u = np.zeros((Nt+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = exp(2x)
        u[0, :] = np.exp(2*x)
        
        # ВТОРОЕ НАЧАЛЬНОЕ УСЛОВИЕ: u_t(x,0) = 0
        if initial_approx == 'first_order':
            # Аппроксимация первого порядка
            u[1, :] = u[0, :]
        else:
            # Аппроксимация второго порядка
            for i in range(1, Nx):
                u[1, i] = u[0, i] + (r2/2)*(u[0, i-1] - 2*u[0, i] + u[0, i+1]) - (5/2)*dt*dt*u[0, i]
            
            # Граничные точки для второго слоя
            self.apply_boundary_conditions(u, 1, x, t[1], dx, dt, bc_type)
        
        # ЯВНАЯ СХЕМА КРЕСТ
        for n in range(1, Nt):
            for i in range(1, Nx):
                u[n+1, i] = 2*u[n, i] - u[n-1, i] + r2*(u[n, i-1] - 2*u[n, i] + u[n, i+1]) - 5*dt*dt*u[n, i]
            
            # ГРАНИЧНЫЕ УСЛОВИЯ
            self.apply_boundary_conditions(u, n+1, x, t[n+1], dx, dt, bc_type)
        
        return x, t, u, 'Явная схема'
    
    def solve_implicit(self, Nx, Nt, bc_type='two_point_first_order', initial_approx='first_order'):
        """
        Неявная схема для гиперболического уравнения
        """
        dx = self.L / Nx
        dt = self.T / Nt
        r = dt / dx
        r2 = r * r
        
        print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r:.3f} → УСТОЙЧИВО (неявная схема)")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt+1)
        u = np.zeros((Nt+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = exp(2x)
        u[0, :] = np.exp(2*x)
        
        # ВТОРОЕ НАЧАЛЬНОЕ УСЛОВИЕ
        if initial_approx == 'first_order':
            u[1, :] = u[0, :]
        else:
            for i in range(1, Nx):
                u[1, i] = u[0, i] + (r2/2)*(u[0, i-1] - 2*u[0, i] + u[0, i+1]) - (5/2)*dt*dt*u[0, i]
            self.apply_boundary_conditions(u, 1, x, t[1], dx, dt, bc_type)
        
        # НЕЯВНАЯ СХЕМА - ПРАВИЛЬНАЯ ФОРМУЛА
        for n in range(1, Nt):
            # Коэффициенты для матрицы
            A = -r2
            B = 1 + 2*r2 + 5*dt*dt
            C = -r2
            
            # Создаем матрицу системы
            main_diag = B * np.ones(Nx+1)
            lower_diag = A * np.ones(Nx)
            upper_diag = C * np.ones(Nx)
            
            # Правая часть: 2u_i^n - u_i^{n-1}
            b = 2*u[n, :] - u[n-1, :]
            
            # ГРАНИЧНЫЕ УСЛОВИЯ
            if bc_type == 'two_point_first_order':
                # Левая граница: (u1 - u0)/dx - 2u0 = 0
                main_diag[0] = -1/dx - 2
                upper_diag[0] = 1/dx
                b[0] = 0
                
                # Правая граница: (uN - u_{N-1})/dx - 2uN = 0
                main_diag[Nx] = 1/dx - 2
                lower_diag[Nx-1] = -1/dx
                b[Nx] = 0
                
            elif bc_type == 'two_point_second_order':
                # Для двухточечной 2-го порядка используем специальную обработку
                # Левая граница
                main_diag[0] = -1/dx - 2
                upper_diag[0] = 1/dx
                if n >= 2:
                    u_xx_0 = (u[n-2, 0] - u[n-1, 0]) / (dt*dt) + 5*u[n-1, 0]
                    b[0] = -(dx*dx/2) * u_xx_0
                else:
                    b[0] = 0
                
                # Правая граница
                main_diag[Nx] = 1/dx - 2
                lower_diag[Nx-1] = -1/dx
                if n >= 2:
                    u_xx_N = (u[n-2, -1] - u[n-1, -1]) / (dt*dt) + 5*u[n-1, -1]
                    b[Nx] = -(dx*dx/2) * u_xx_N
                else:
                    b[Nx] = 0
                
            elif bc_type == 'three_point_second_order':
                # Левая граница: (-3u0 + 4u1 - u2)/(2dx) - 2u0 = 0
                main_diag[0] = -3/(2*dx) - 2
                upper_diag[0] = 4/(2*dx)
                upper_diag[1] = -1/(2*dx)
                b[0] = 0
                
                # Правая граница: (3uN - 4u_{N-1} + u_{N-2})/(2dx) - 2uN = 0
                main_diag[Nx] = 3/(2*dx) - 2
                lower_diag[Nx-1] = -4/(2*dx)
                lower_diag[Nx-2] = 1/(2*dx)
                b[Nx] = 0
            
            # Решение системы
            try:
                A_matrix = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
                u[n+1, :] = spsolve(A_matrix, b)
            except Exception as e:
                print(f"Ошибка решения СЛАУ: {e}")
                # Резервный вариант - используем явную схему для этого шага
                for i in range(1, Nx):
                    u[n+1, i] = 2*u[n, i] - u[n-1, i] + r2*(u[n, i-1] - 2*u[n, i] + u[n, i+1]) - 5*dt*dt*u[n, i]
                self.apply_boundary_conditions(u, n+1, x, t[n+1], dx, dt, bc_type)
        
        return x, t, u, 'Неявная схема'
    
    def solve_crank_nicolson(self, Nx, Nt, bc_type='two_point_first_order', initial_approx='first_order'):
        """
        Схема Кранка-Николсона для гиперболического уравнения
        """
        dx = self.L / Nx
        dt = self.T / Nt
        r = dt / dx
        r2 = r * r
        
        print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r:.3f} → УСТОЙЧИВО (Кранк-Николсон)")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt+1)
        u = np.zeros((Nt+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ
        u[0, :] = np.exp(2*x)
        
        # ВТОРОЕ НАЧАЛЬНОЕ УСЛОВИЕ
        if initial_approx == 'first_order':
            u[1, :] = u[0, :]
        else:
            for i in range(1, Nx):
                u[1, i] = u[0, i] + (r2/2)*(u[0, i-1] - 2*u[0, i] + u[0, i+1]) - (5/2)*dt*dt*u[0, i]
            self.apply_boundary_conditions(u, 1, x, t[1], dx, dt, bc_type)
        
        # СХЕМА КРАНКА-НИКОЛСОНА - ПРАВИЛЬНАЯ ФОРМУЛА
        for n in range(1, Nt):
            # Коэффициенты для матрицы
            A = -0.5 * r2
            B = 1 + r2 + 2.5*dt*dt
            C = -0.5 * r2
            
            # Создаем матрицу системы
            main_diag = B * np.ones(Nx+1)
            lower_diag = A * np.ones(Nx)
            upper_diag = C * np.ones(Nx)
            
            # Правая часть
            b = np.zeros(Nx+1)
            for i in range(1, Nx):
                spatial_part = 0.5 * r2 * (u[n, i-1] - 2*u[n, i] + u[n, i+1])
                source_part = -2.5 * dt*dt * u[n, i]
                b[i] = 2*u[n, i] - u[n-1, i] + spatial_part + source_part
            
            # ГРАНИЧНЫЕ УСЛОВИЯ
            if bc_type == 'two_point_first_order':
                main_diag[0] = -1/dx - 2
                upper_diag[0] = 1/dx
                b[0] = 0
                
                main_diag[Nx] = 1/dx - 2
                lower_diag[Nx-1] = -1/dx
                b[Nx] = 0
                
            elif bc_type == 'two_point_second_order':
                # Для двухточечной 2-го порядка используем специальную обработку
                main_diag[0] = -1/dx - 2
                upper_diag[0] = 1/dx
                if n >= 2:
                    u_xx_0 = (u[n-2, 0] - u[n-1, 0]) / (dt*dt) + 5*u[n-1, 0]
                    b[0] = -(dx*dx/2) * u_xx_0
                else:
                    b[0] = 0
                
                main_diag[Nx] = 1/dx - 2
                lower_diag[Nx-1] = -1/dx
                if n >= 2:
                    u_xx_N = (u[n-2, -1] - u[n-1, -1]) / (dt*dt) + 5*u[n-1, -1]
                    b[Nx] = -(dx*dx/2) * u_xx_N
                else:
                    b[Nx] = 0
                
            elif bc_type == 'three_point_second_order':
                main_diag[0] = -3/(2*dx) - 2
                upper_diag[0] = 4/(2*dx)
                upper_diag[1] = -1/(2*dx)
                b[0] = 0
                
                main_diag[Nx] = 3/(2*dx) - 2
                lower_diag[Nx-1] = -4/(2*dx)
                lower_diag[Nx-2] = 1/(2*dx)
                b[Nx] = 0
            
            # Решение системы
            try:
                A_matrix = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
                u[n+1, :] = spsolve(A_matrix, b)
            except Exception as e:
                print(f"Ошибка решения СЛАУ: {e}")
                # Резервный вариант
                for i in range(1, Nx):
                    u[n+1, i] = 2*u[n, i] - u[n-1, i] + r2*(u[n, i-1] - 2*u[n, i] + u[n, i+1]) - 5*dt*dt*u[n, i]
                self.apply_boundary_conditions(u, n+1, x, t[n+1], dx, dt, bc_type)
        
        return x, t, u, 'Кранк-Николсон'
    
    def apply_boundary_conditions(self, u, n, x, t, dx, dt, bc_type):
        """
        Применение граничных условий
        """
        if bc_type == 'two_point_first_order':
            # Двухточечная аппроксимация 1-го порядка
            u[n, 0] = u[n, 1] / (1 + 2*dx)
            u[n, -1] = u[n, -2] / (1 - 2*dx)
        
        elif bc_type == 'two_point_second_order':
            # Двухточечная аппроксимация 2-го порядка
            if n >= 2:  # Нужны как минимум 2 предыдущих временных слоя
                # Левая граница
                u_xx_0 = (u[n-2, 0] - u[n-1, 0]) / (dt*dt) + 5*u[n-1, 0]
                u[n, 0] = (u[n, 1] - (dx*dx/2) * u_xx_0) / (1 + 2*dx)
                
                # Правая граница
                u_xx_N = (u[n-2, -1] - u[n-1, -1]) / (dt*dt) + 5*u[n-1, -1]
                u[n, -1] = (u[n, -2] - (dx*dx/2) * u_xx_N) / (1 - 2*dx)
            else:
                # Для первых шагов используем первый порядок
                u[n, 0] = u[n, 1] / (1 + 2*dx)
                u[n, -1] = u[n, -2] / (1 - 2*dx)
        
        elif bc_type == 'three_point_second_order':
            # Трехточечная аппроксимация 2-го порядка
            u[n, 0] = (4*u[n, 1] - u[n, 2]) / (3 + 4*dx)
            u[n, -1] = (4*u[n, -2] - u[n, -3]) / (3 - 4*dx)

    def calculate_error(self, u_numeric, u_exact):
        return np.abs(u_numeric - u_exact)

    def plot_solutions(self, x, t, u, scheme_name, bc_type, initial_approx):
        """Построение графиков решения с 5 графиками"""
        fig = plt.figure(figsize=(20, 12))
        
        scheme_titles = {
            'Явная схема': 'ЯВНАЯ СХЕМА',
            'Неявная схема': 'НЕЯВНАЯ СХЕМА', 
            'Кранк-Николсон': 'СХЕМА КРАНКА-НИКОЛСОНА'
        }
        
        # bc_titles = {
        #     'two_point_first_order': 'Двухточечная аппроксимация 1-го порядка',
        #     'two_point_second_order': 'Двухточечная аппроксимация 2-го порядка',
        #     'three_point_second_order': 'Трехточечная аппроксимация 2-го порядка'
        # }
        
        # approx_titles = {
        #     'first_order': 'Аппроксимация нач. условия 1-го порядка',
        #     'second_order': 'Аппроксимация нач. условия 2-го порядка'
        # }
        
        main_title = f"{scheme_titles[scheme_name]}\n"
        # {bc_titles[bc_type]}\n{approx_titles[initial_approx]}"
        
        # 1. 2D график: решение в разные моменты времени
        plt.subplot(2, 3, 1)
        time_indices = [0, len(t)//4, len(t)//2, 3*len(t)//4, -1]
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        labels = ['t = 0', f't = {t[len(t)//4]:.2f}', f't = {t[len(t)//2]:.2f}', 
                 f't = {t[3*len(t)//4]:.2f}', f't = {t[-1]:.2f}']
        
        for idx, color, label in zip(time_indices, colors, labels):
            if idx < len(u):
                plt.plot(x, u[idx, :], color=color, label=label, linewidth=2)
                plt.plot(x, self.exact_solution(x, t[idx]), color=color, 
                        linestyle='--', alpha=0.5, linewidth=1)
        
        plt.xlabel('x')
        plt.ylabel('u(x,t)')
        plt.title('Решение в разные моменты времени')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. 3D поверхность численного решения
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        
        stride_t = max(1, len(t) // 20)
        stride_x = max(1, len(x) // 30)
        
        t_reduced = t[::stride_t]
        x_reduced = x[::stride_x]
        u_reduced = u[::stride_t, ::stride_x]
        
        X, T_mesh = np.meshgrid(x_reduced, t_reduced)
        
        surf = ax.plot_surface(X, T_mesh, u_reduced, cmap='viridis', alpha=0.9)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('3D-поверхность решения')
        
        # 3. Погрешность по пространству
        plt.subplot(2, 3, 3)
        error_times = [len(t)//4, len(t)//2, -1]
        colors = ['green', 'orange', 'red']
        labels = [f't = {t[i]:.2f}' for i in error_times]
        
        for time_idx, color, label in zip(error_times, colors, labels):
            if time_idx < len(u):
                error = self.calculate_error(u[time_idx, :], self.exact_solution(x, t[time_idx]))
                plt.plot(x, error, color=color, label=label, linewidth=2)
        
        plt.xlabel('x')
        plt.ylabel('Погрешность')
        plt.title('Абсолютная погрешность')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Сравнение при t = T
        plt.subplot(2, 3, 4)
        if len(u) > 0:
            plt.plot(x, u[-1, :], 'b-', linewidth=2, label='Численное')
            plt.plot(x, self.exact_solution(x, t[-1]), 'r--', linewidth=2, label='Аналитическое')
            plt.xlabel('x')
            plt.ylabel('u(x,T)')
            plt.title(f'Сравнение при t = {t[-1]:.2f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 5. Максимальная погрешность от времени
        plt.subplot(2, 3, 6)
        max_errors = []
        times_for_plot = []
        for n in range(0, len(t), max(1, len(t)//20)):
            exact = self.exact_solution(x, t[n])
            error = np.max(np.abs(u[n, :] - exact))
            max_errors.append(error)
            times_for_plot.append(t[n])
        
        plt.plot(times_for_plot, max_errors, 'r-', linewidth=2)
        plt.xlabel('t')
        plt.ylabel('Макс. погрешность')
        plt.title('Зависимость погрешности от времени')
        plt.grid(True, alpha=0.3)
        if max(max_errors) > 0:
            plt.yscale('log')
        
        fig.suptitle(main_title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return max_errors

def main():
    solver = HyperbolicEquationSolver(L=1.0, T=1.0)
    
    Nx = 50
    Nt = 200
    
    schemes = [
        ('explicit', 'Явная схема'),
        ('implicit', 'Неявная схема'),
        ('crank_nicolson', 'Кранк-Николсон')
    ]
    
    # Теперь три типа граничных условий
    bc_types = ['two_point_first_order', 'two_point_second_order', 'three_point_second_order']
    
    for scheme_func, scheme_name in schemes:
        print(f"\n{'='*60}")
        print(f"РЕШЕНИЕ МЕТОДОМ: {scheme_name}")
        print(f"{'='*60}")
        
        for bc_type in bc_types:
            for initial_approx in ['first_order', 'second_order']:
                print(f"\nГраничные условия: {bc_type}, Аппроксимация: {initial_approx}")
                
                try:
                    if scheme_func == 'explicit':
                        x, t, u, name = solver.solve_explicit(Nx, Nt, bc_type, initial_approx)
                    elif scheme_func == 'implicit':
                        x, t, u, name = solver.solve_implicit(Nx, Nt, bc_type, initial_approx)
                    else:
                        x, t, u, name = solver.solve_crank_nicolson(Nx, Nt, bc_type, initial_approx)
                    
                    final_error = np.max(solver.calculate_error(u[-1, :], solver.exact_solution(x, t[-1])))
                    print(f"Максимальная погрешность: {final_error:.6e}")
                    
                    if final_error < 1e6:  # Ослабили критерий для двухточечной 2-го порядка
                        solver.plot_solutions(x, t, u, scheme_name, bc_type, initial_approx)
                    else:
                        print("Погрешность слишком велика - пропускаем графики")
                        
                except Exception as e:
                    print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()