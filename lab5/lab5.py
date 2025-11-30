import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import time
from mpl_toolkits.mplot3d import Axes3D

class HeatEquationSolver:
    def __init__(self, a=1.0, L=np.pi, T=1.0):
        self.a = a
        self.L = L
        self.T = T
        self.analytic_solution = lambda x, t: np.exp(-a*t) * np.sin(x)
    
    def exact_solution(self, x, t):
        return np.exp(-self.a*t) * np.sin(x)
    
    def solve_explicit(self, Nx, Nt, bc_type='two_point_first_order'):
        """
        Явная схема
        """
        dx = self.L / Nx
        dt_original = self.T / Nt
        r_original = self.a * dt_original / dx**2
        
        # Сохраняем оригинальные параметры для вывода
        dt = dt_original
        Nt_final = Nt
        r_final = r_original
        corrected = False
        
        # Автоматическая корректировка для устойчивости
        if r_original > 0.5:
            corrected = True
            dt = 0.4 * dx**2 / self.a
            Nt_final = int(np.ceil(self.T / dt))
            r_final = self.a * dt / dx**2
            
            print(f"КОРРЕКЦИЯ ДЛЯ УСТОЙЧИВОСТИ:")
            print(f"  Было: dt = {dt_original:.6f}, Nt = {Nt}, r = {r_original:.3f} > 0.5 → НЕУСТОЙЧИВО")
            print(f"  Стало: dt = {dt:.6f}, Nt = {Nt_final}, r = {r_final:.3f} ≤ 0.5 → УСТОЙЧИВО")
        else:
            print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r_final:.3f} ≤ 0.5 → УСТОЙЧИВО")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt_final+1)
        u = np.zeros((Nt_final+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = sin(x)
        u[0, :] = np.sin(x)
        
        for n in range(Nt_final):
            # Внутренние точки
            for i in range(1, Nx):
                u[n+1, i] = u[n, i] + r_final * (u[n, i-1] - 2*u[n, i] + u[n, i+1])
            
            # ГРАНИЧНЫЕ УСЛОВИЯ НЕЙМАНА
            if bc_type == 'two_point_first_order':
                u[n+1, 0] = u[n+1, 1] - dx * np.exp(-self.a * t[n+1])
                u[n+1, Nx] = u[n+1, Nx-1] - dx * np.exp(-self.a * t[n+1])
            
            # elif bc_type == 'two_point_second_order':
            #     u[n+1, 0] = u[n+1, 1] - dx * np.exp(-self.a * t[n+1]) + \
            #                (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
            #     u[n+1, Nx] = u[n+1, Nx-1] - dx * np.exp(-self.a * t[n+1]) + \
            #                 (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
            
            elif bc_type == 'three_point_second_order':
                u[n+1, 0] = (4*u[n+1, 1] - u[n+1, 2] - 2*dx*np.exp(-self.a*t[n+1])) / 3
                u[n+1, Nx] = (4*u[n+1, Nx-1] - u[n+1, Nx-2] - 2*dx*np.exp(-self.a*t[n+1])) / 3
        
        return x, t, u, 'Явная схема'
    
    def solve_implicit(self, Nx, Nt, bc_type='two_point_first_order'):
        """
        Неявная схема
        """
        dx = self.L / Nx
        dt = self.T / Nt
        r = self.a * dt / dx**2
        
        # Для неявной схемы ВСЕГДА устойчиво
        stability = "УСТОЙЧИВО (безусловно)"
        print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r:.3f} → {stability}")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt+1)
        u = np.zeros((Nt+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = sin(x)
        u[0, :] = np.sin(x)
        
        for n in range(Nt):
            # Матрица системы
            main_diag = (1 + 2*r) * np.ones(Nx+1)
            lower_diag = -r * np.ones(Nx)
            upper_diag = -r * np.ones(Nx)
            
            # Правая часть
            b = u[n, :].copy()
            
            # ГРАНИЧНЫЕ УСЛОВИЯ НЕЙМАНА
            if bc_type == 'two_point_first_order':
                main_diag[0] = 1.0
                upper_diag[0] = -1.0
                b[0] = -dx * np.exp(-self.a * t[n+1])
                
                main_diag[Nx] = 1.0
                lower_diag[Nx-1] = -1.0
                b[Nx] = -dx * np.exp(-self.a * t[n+1])
            
            # elif bc_type == 'two_point_second_order':
            #     main_diag[0] = 1.0
            #     upper_diag[0] = -1.0
            #     b[0] = -dx * np.exp(-self.a * t[n+1]) + \
            #            (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
                
            #     main_diag[Nx] = 1.0
            #     lower_diag[Nx-1] = -1.0
            #     b[Nx] = -dx * np.exp(-self.a * t[n+1]) + \
            #             (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
            
            elif bc_type == 'three_point_second_order':
                main_diag[0] = 3.0
                upper_diag[0] = -4.0
                upper_diag[1] = 1.0
                b[0] = -2 * dx * np.exp(-self.a * t[n+1])
                
                main_diag[Nx] = 3.0
                lower_diag[Nx-1] = -4.0
                lower_diag[Nx-2] = 1.0
                b[Nx] = -2 * dx * np.exp(-self.a * t[n+1])
            
            # Решение системы
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
            u[n+1, :] = spsolve(A, b)
        
        return x, t, u, 'Неявная схема'
    
    def solve_crank_nicolson(self, Nx, Nt, bc_type='two_point_first_order'):
        """
        Схема Кранка-Николсона
        """
        dx = self.L / Nx
        dt = self.T / Nt
        r = self.a * dt / dx**2
        
        # Для Кранка-Николсона тоже ВСЕГДА устойчиво
        stability = "УСТОЙЧИВО (безусловно)"
        print(f"Параметры: dx = {dx:.4f}, dt = {dt:.6f}, r = {r:.3f} → {stability}")
        
        x = np.linspace(0, self.L, Nx+1)
        t = np.linspace(0, self.T, Nt+1)
        u = np.zeros((Nt+1, Nx+1))
        
        # НАЧАЛЬНОЕ УСЛОВИЕ: u(x,0) = sin(x)
        u[0, :] = np.sin(x)
        
        for n in range(Nt):
            # Матрица системы
            main_diag = (1 + r) * np.ones(Nx+1)
            lower_diag = -r/2 * np.ones(Nx)
            upper_diag = -r/2 * np.ones(Nx)
            
            # Правая часть
            b = np.zeros(Nx+1)
            for i in range(1, Nx):
                b[i] = (r/2)*u[n, i-1] + (1 - r)*u[n, i] + (r/2)*u[n, i+1]
            
            # Граничные точки для правой части
            b[0] = u[n, 0]
            b[Nx] = u[n, Nx]
            
            # ГРАНИЧНЫЕ УСЛОВИЯ
            if bc_type == 'two_point_first_order':
                main_diag[0] = 1.0
                upper_diag[0] = -1.0
                b[0] = -dx * np.exp(-self.a * t[n+1])
                
                main_diag[Nx] = 1.0
                lower_diag[Nx-1] = -1.0
                b[Nx] = -dx * np.exp(-self.a * t[n+1])
            
            # elif bc_type == 'two_point_second_order':
            #     main_diag[0] = 1.0
            #     upper_diag[0] = -1.0
            #     b[0] = -dx * np.exp(-self.a * t[n+1]) + \
            #            (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
                
            #     main_diag[Nx] = 1.0
            #     lower_diag[Nx-1] = -1.0
            #     b[Nx] = -dx * np.exp(-self.a * t[n+1]) + \
            #             (self.a * dx**2 / 2) * np.exp(-self.a * t[n+1])
            
            elif bc_type == 'three_point_second_order':
                main_diag[0] = 3.0
                upper_diag[0] = -4.0
                upper_diag[1] = 1.0
                b[0] = -2 * dx * np.exp(-self.a * t[n+1])
                
                main_diag[Nx] = 3.0
                lower_diag[Nx-1] = -4.0
                lower_diag[Nx-2] = 1.0
                b[Nx] = -2 * dx * np.exp(-self.a * t[n+1])
            
            # Решение системы
            A = diags([lower_diag, main_diag, upper_diag], [-1, 0, 1], format='csc')
            u[n+1, :] = spsolve(A, b)
        
        return x, t, u, 'Кранк-Николсон'

    def calculate_error(self, u_numeric, u_exact):
        """Вычисление абсолютной погрешности"""
        return np.abs(u_numeric - u_exact)

    def plot_solutions(self, x, t, u, scheme_name, bc_type):
        """Построение графиков решения"""
        # Создаем фигуру в полноэкранном режиме
        fig = plt.figure(figsize=(25, 14))
        
        scheme_titles = {
            'Явная схема': 'ЯВНАЯ КОНЕЧНО-РАЗНОСТНАЯ СХЕМА',
            'Неявная схема': 'НЕЯВНАЯ КОНЕЧНО-РАЗНОСТНАЯ СХЕМА', 
            'Кранк-Николсон': 'СХЕМА КРАНКА-НИКОЛСОНА'
        }
        
        bc_titles = {
            'two_point_first_order': 'Двухточечная аппроксимация 1-го порядка',
            'three_point_second_order': 'Трехточечная аппроксимация 2-го порядка'
        }
        
        main_title = f"{scheme_titles[scheme_name]}\n{bc_titles[bc_type]}"
        
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
        
        plt.xlabel('Пространство (x)', fontsize=12)
        plt.ylabel('Температура u(x,t)', fontsize=12)
        plt.title('Решение в разные моменты времени\n(сплошная - численное, пунктир - аналитическое)', fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        
        # 2. График абсолютной погрешности по пространству
        plt.subplot(2, 3, 3)
        error_times = [0, len(t)//4, len(t)//2, -1]
        error_colors = ['blue', 'green', 'orange', 'red']
        error_labels = [f't = {t[i]:.2f}' for i in error_times]
        
        for time_idx, color, label in zip(error_times, error_colors, error_labels):
            if time_idx < len(u):
                error = self.calculate_error(u[time_idx, :], self.exact_solution(x, t[time_idx]))
                plt.plot(x, error, color=color, label=label, linewidth=2)
        
        plt.xlabel('Пространство (x)', fontsize=12)
        plt.ylabel('Абсолютная погрешность', fontsize=12)
        plt.title('Абсолютная погрешность по пространству\nв разные моменты времени', fontsize=11)
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        # 3. Зависимость максимальной погрешности от времени
        plt.subplot(2, 3, 6)
        max_errors = []
        valid_times = []
        for n in range(len(t)):
            exact = self.exact_solution(x, t[n])
            error = np.max(np.abs(u[n, :] - exact))
            max_errors.append(error)
            valid_times.append(t[n])
        
        plt.plot(valid_times, max_errors, 'r-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.xlabel('Время (t)', fontsize=12)
        plt.ylabel('Максимальная погрешность', fontsize=12)
        plt.title('Зависимость максимальной погрешности от времени', fontsize=11)
        plt.grid(True, alpha=0.3)
        if max(max_errors) > 0:
            plt.yscale('log')
        
        # 4. Сравнение численного и аналитического решения при t = T
        plt.subplot(2, 3, 4)
        if len(u) > 0:
            plt.plot(x, u[-1, :], 'b-', linewidth=2, label='Численное решение')
            plt.plot(x, self.exact_solution(x, t[-1]), 'r--', linewidth=2, 
                    label='Аналитическое решение')
            plt.xlabel('Пространство (x)', fontsize=12)
            plt.ylabel('Температура u(x,T)', fontsize=12)
            plt.title(f'Сравнение решений при t = {t[-1]:.2f}', fontsize=11)
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
        
        # 5. 3D ПОВЕРХНОСТЬ РЕШЕНИЯ С HEATMAP
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        
        # Уменьшаем количество точек для 3D графика
        stride_t = max(1, len(t) // 30)
        stride_x = max(1, len(x) // 50)
        
        t_reduced = t[::stride_t]
        x_reduced = x[::stride_x]
        u_reduced = u[::stride_t, ::stride_x]
        
        X, T_mesh = np.meshgrid(x_reduced, t_reduced)
        
        # Создаем surface plot с цветовой картой по температуре
        surf = ax.plot_surface(X, T_mesh, u_reduced, cmap='hot', alpha=0.9, 
                              linewidth=0, antialiased=True)
        
        ax.set_xlabel('Пространство (x)', fontsize=12, labelpad=15)
        ax.set_ylabel('Время (t)', fontsize=12, labelpad=15)
        ax.set_zlabel('Температура u(x,t)', fontsize=12, labelpad=15)
        ax.set_title('3D-поверхность решения', fontsize=11)
        
        # Добавляем colorbar с большим отступом вправо
        cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=25, pad=0.25)
        cbar.set_label('Температура', rotation=270, labelpad=20, fontsize=12)
        
        # Добавляем общее название ПОСЛЕ создания всех subplots
        fig.suptitle(main_title, fontsize=16, fontweight='bold', y=0.97)
        
        # Оптимизированное расположение с максимальными отступами
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, bottom=0.05, left=0.03, right=0.97, 
                          hspace=0.4, wspace=0.4)
        
        # Открываем в полноэкранном режиме
        manager = plt.get_current_fig_manager()
        try:
            manager.window.showMaximized()
        except:
            try:
                manager.window.state('zoomed')
            except:
                pass
        
        plt.show()
        
        return max_errors

def main():
    # Параметры задачи
    a = 1
    L = np.pi
    T = 1.0
    
    solver = HeatEquationSolver(a, L, T)
    
    # Параметры сетки
    Nx = 50
    Nt = 1000
    
    # Типы граничных условий
    bc_types = ['two_point_first_order', 'two_point_second_order', 'three_point_second_order']
    
    schemes = [
        ('explicit', 'Явная схема'),
        ('implicit', 'Неявная схема'),
        ('crank_nicolson', 'Кранк-Николсон')
    ]
    
    for scheme_func, scheme_name in schemes:
        print(f"\n{'='*60}")
        print(f"Решение методом: {scheme_name}")
        print(f"{'='*60}")
        
        for bc_type in bc_types:
            print(f"\n--- Граничные условия: {bc_type} ---")
            
            start_time = time.time()
            
            try:
                if scheme_func == 'explicit':
                    x, t, u, name = solver.solve_explicit(Nx, Nt, bc_type)
                elif scheme_func == 'implicit':
                    x, t, u, name = solver.solve_implicit(Nx, Nt, bc_type)
                else:
                    x, t, u, name = solver.solve_crank_nicolson(Nx, Nt, bc_type)
                
                computation_time = time.time() - start_time
                print(f"Время вычислений: {computation_time:.3f} сек")
                
                # Проверка начального условия
                print(f"  u(0,0) = {u[0, 0]:.6f} (ожидается: {np.sin(0):.6f})")
                print(f"  u(π/2,0) = {u[0, len(x)//2]:.6f} (ожидается: {np.sin(np.pi/2):.6f})")
                print(f"  u(π,0) = {u[0, -1]:.6f} (ожидается: {np.sin(np.pi):.6f})")
                
                # Построение графиков
                max_errors = solver.plot_solutions(x, t, u, scheme_name, bc_type)
                
                # Вывод информации о погрешности
                final_error = np.max(solver.calculate_error(u[-1, :], solver.exact_solution(x, t[-1])))
                print(f"Максимальная погрешность при t={T}: {final_error:.6e}")
                
            except Exception as e:
                print(f"Ошибка при решении: {e}")
                continue

if __name__ == "__main__":
    main()