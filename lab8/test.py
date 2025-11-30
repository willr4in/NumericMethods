import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time
import copy

class Parabolic2DSolver:
    def __init__(self, a=1.0, Lx=np.pi/4, Ly=np.log(2), T=1.0):
        self.a = a
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        
    def analytical_solution(self, x, y, t):
        """Аналитическое решение"""
        return np.cos(2*x) * np.cosh(y) * np.exp(-3*self.a*t)
    
    def initial_condition(self, x, y):
        """Начальное условие"""
        return np.cos(2*x) * np.cosh(y)
    
    def tdma_solve(self, a, b, c, d, n):
        """
        Решение трехдиагональной системы методом прогонки (TDMA)
        """
        # Прямой ход
        c_prime = np.zeros(n-1)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n-1):
            denominator = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denominator
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
        
        d_prime[n-1] = (d[n-1] - a[n-1] * d_prime[n-2]) / (b[n-1] - a[n-1] * c_prime[n-2])
        
        # Обратный ход
        x = np.zeros(n)
        x[n-1] = d_prime[n-1]
        
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x

    def LU_decompose(self, A):
        """LU-разложение матрицы"""
        n = len(A)
        L = [[0 for _ in range(n)] for _ in range(n)]
        U = copy.deepcopy(A)

        for k in range(n):
            if abs(U[k][k]) < 1e-12:
                raise ValueError("Матрица вырождена или требует перестановки строк")
                
            for i in range(k, n):
                L[i][k] = U[i][k] / U[k][k]
            for i in range(k + 1, n):
                for j in range(k, n):
                    U[i][j] -= L[i][k] * U[k][j]

        return L, U

    def solve_system(self, L, U, b):
        """Решение системы LUx = b"""
        n = len(L)
        y = [0 for _ in range(n)]
        for i in range(n):
            sum_val = sum(y[j] * L[i][j] for j in range(i))
            y[i] = b[i] - sum_val

        x = [0 for _ in range(n)]
        for i in range(n - 1, -1, -1):
            if abs(U[i][i]) < 1e-12:
                raise ValueError("Матрица вырождена")
                
            sum_val = sum(x[j] * U[i][j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_val) / U[i][i]
        return x

    def lu_solve(self, A, b):
        """
        Решение системы линейных уравнений методом LU-разложения
        """
        # Преобразуем numpy массивы в списки
        A_list = A.tolist() if isinstance(A, np.ndarray) else A
        b_list = b.tolist() if isinstance(b, np.ndarray) else b
        
        # LU-разложение
        L, U = self.LU_decompose(A_list)
        
        # Решение системы
        x = self.solve_system(L, U, b_list)
        
        return np.array(x)

    def solve_neumann_2nd_order(self, a, b, c, f, n, hy, method='lu'):
        """
        Решение системы с условием Неймана 2-го порядка
        """
        if method == 'lu':
            # Создаем полную матрицу для метода LU-разложения
            A_full = np.zeros((n, n))
            F_full = np.zeros(n)
            
            # Заполняем матрицу для внутренних точек
            for i in range(1, n-1):
                if i > 0:
                    A_full[i, i-1] = a[i]
                A_full[i, i] = b[i]
                if i < n-1:
                    A_full[i, i+1] = c[i]
                F_full[i] = f[i]
            
            # Нижняя граница (Дирихле)
            A_full[0, 0] = 1.0
            F_full[0] = f[0]
            
            # Верхняя граница (Нейман 2-го порядка)
            # (3u_n - 4u_{n-1} + u_{n-2})/(2hy) = f_n
            # => 3u_n - 4u_{n-1} + u_{n-2} = 2hy * f_n
            A_full[n-1, n-1] = 3.0
            A_full[n-1, n-2] = -4.0
            if n > 2:
                A_full[n-1, n-3] = 1.0
            F_full[n-1] = 2 * hy * f[n-1]
            
            return self.lu_solve(A_full, F_full)
        
        else:  # Метод исключения
            # Для 2-го порядка используем фиктивную точку и исключаем ее
            u = np.zeros(n)
            
            # Нижняя граница (Дирихле)
            u[0] = f[0]
            
            if n > 1:
                # Решаем для n-1 точек как трехдиагональную систему
                a_inner = a[1:n-1].copy()
                b_inner = b[1:n-1].copy()
                c_inner = c[1:n-1].copy()
                f_inner = f[1:n-1].copy()
                
                if len(a_inner) > 0:
                    u_inner = self.tdma_solve(a_inner, b_inner, c_inner, f_inner, n-2)
                    u[1:n-1] = u_inner
                
                # Нейман 2-го порядка на верхней границе
                if n >= 3:
                    u[-1] = (4*u[-2] - u[-3] + 2*hy*f[-1]) / 3
                elif n == 2:
                    u[-1] = u[0] + hy * f[-1]  # Вырожденный случай
            
            return u

    def method_variable_directions(self, Nx, Ny, Nt, theta=0.5, neumann_order=1, save_history=True):
        """
        Метод переменных направлений (МПН) с выбором порядка точности для условия Неймана
        """
        print(f"Запуск МПН: Nx={Nx}, Ny={Ny}, Nt={Nt}, theta={theta}, порядок Неймана: {neumann_order}")
        
        # Шаги сетки
        hx = self.Lx / (Nx - 1)
        hy = self.Ly / (Ny - 1)
        tau = self.T / Nt
        
        # Сетка
        x = np.linspace(0, self.Lx, Nx)
        y = np.linspace(0, self.Ly, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Инициализация решения
        u = self.initial_condition(X, Y)
        
        # Коэффициенты
        sigma_x = self.a * tau / (2 * hx**2)
        sigma_y = self.a * tau / (2 * hy**2)
        
        errors = []
        times = []
        
        # Сохраняем историю решений если нужно
        if save_history:
            solutions_history = [u.copy()]
            times_history = [0.0]
        
        for k in range(Nt):
            t = k * tau
            t_half = (k + 0.5) * tau
            t_full = (k + 1) * tau
            
            # ПЕРВЫЙ ДРОБНЫЙ ШАГ (неявно по x, явно по y)
            u_half = np.zeros_like(u)
            
            # Внутренние точки по y (j от 1 до Ny-2)
            for j in range(1, Ny-1):
                A = np.zeros(Nx)
                B = np.zeros(Nx) 
                C = np.zeros(Nx)
                F = np.zeros(Nx)
                
                for i in range(1, Nx-1):
                    A[i] = -theta * sigma_x
                    B[i] = 1 + 2 * theta * sigma_x
                    C[i] = -theta * sigma_x
                    
                    # Явная часть по y (внутренние точки)
                    y_explicit = u[i,j] + sigma_y * (1-theta) * (
                        u[i,j+1] - 2*u[i,j] + u[i,j-1]
                    )
                    F[i] = y_explicit
                
                # Граничные условия по x
                B[0] = 1; C[0] = 0
                F[0] = np.cosh(y[j]) * np.exp(-3*self.a*t_half)
                
                A[-1] = 0; B[-1] = 1
                F[-1] = 0
                
                u_half[:, j] = self.tdma_solve(A, B, C, F, Nx)
            
            # Нижняя граница (y=0)
            j = 0
            A = np.zeros(Nx)
            B = np.zeros(Nx)
            C = np.zeros(Nx)
            F = np.zeros(Nx)
            
            for i in range(1, Nx-1):
                A[i] = -theta * sigma_x
                B[i] = 1 + 2 * theta * sigma_x
                C[i] = -theta * sigma_x
                F[i] = np.cos(2*x[i]) * np.exp(-3*self.a*t_half)  # Граничное условие
                
            B[0] = 1; C[0] = 0
            F[0] = np.cosh(y[j]) * np.exp(-3*self.a*t_half)
            
            A[-1] = 0; B[-1] = 1
            F[-1] = 0
            
            u_half[:, j] = self.tdma_solve(A, B, C, F, Nx)
            
            # Верхняя граница (y=Ly) - условие Неймана
            j = Ny-1
            A = np.zeros(Nx)
            B = np.zeros(Nx)
            C = np.zeros(Nx)
            F = np.zeros(Nx)
            
            for i in range(1, Nx-1):
                A[i] = -theta * sigma_x
                B[i] = 1 + 2 * theta * sigma_x
                C[i] = -theta * sigma_x
                
                if neumann_order == 1:
                    # Аппроксимация 1-го порядка
                    F[i] = u[i,j] + sigma_y * (1-theta) * (
                        2 * (u[i,j-1] - u[i,j]) + hy * 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t)
                    )
                else:
                    # Аппроксимация 2-го порядка
                    F[i] = u[i,j] + sigma_y * (1-theta) * (
                        u[i,j-2] - 4*u[i,j-1] + 3*u[i,j] + hy * 1.5 * np.cos(2*x[i]) * np.exp(-3*self.a*t)
                    )
                
            B[0] = 1; C[0] = 0
            F[0] = np.cosh(y[j]) * np.exp(-3*self.a*t_half)
            
            A[-1] = 0; B[-1] = 1
            F[-1] = 0
            
            u_half[:, j] = self.tdma_solve(A, B, C, F, Nx)
            
            # ВТОРОЙ ДРОБНЫЙ ШАГ (явно по x, неявно по y)
            u_new = np.zeros_like(u)
            
            # Внутренние точки по x (i от 1 до Nx-2)
            for i in range(1, Nx-1):
                A_arr = np.zeros(Ny)
                B_arr = np.zeros(Ny)
                C_arr = np.zeros(Ny)
                F_arr = np.zeros(Ny)
                
                for j in range(1, Ny-1):
                    A_arr[j] = -theta * sigma_y
                    B_arr[j] = 1 + 2 * theta * sigma_y
                    C_arr[j] = -theta * sigma_y
                    
                    # Явная часть по x (внутренние точки)
                    x_explicit = u_half[i,j] + sigma_x * (1-theta) * (
                        u_half[i+1,j] - 2*u_half[i,j] + u_half[i-1,j]
                    )
                    F_arr[j] = x_explicit
                
                # Граничные условия по y
                B_arr[0] = 1; C_arr[0] = 0
                F_arr[0] = np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                # Условие Неймана на верхней границе
                if neumann_order == 1:
                    # Схема 1-го порядка
                    A_arr[-1] = -1/hy
                    B_arr[-1] = 1/hy
                    F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                    
                    u_new[i, :] = self.tdma_solve(A_arr, B_arr, C_arr, F_arr, Ny)
                else:
                    # Схема 2-го порядка - используем специальный решатель
                    A_arr[-1] = -theta * sigma_y
                    B_arr[-1] = 1 + 2 * theta * sigma_y
                    C_arr[-1] = -theta * sigma_y
                    F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                    
                    u_new[i, :] = self.solve_neumann_2nd_order(A_arr, B_arr, C_arr, F_arr, Ny, hy, method='lu')
            
            # Левая граница (x=0)
            i = 0
            A_arr = np.zeros(Ny)
            B_arr = np.zeros(Ny)
            C_arr = np.zeros(Ny)
            F_arr = np.zeros(Ny)
            
            for j in range(1, Ny-1):
                A_arr[j] = -theta * sigma_y
                B_arr[j] = 1 + 2 * theta * sigma_y
                C_arr[j] = -theta * sigma_y
                F_arr[j] = np.cosh(y[j]) * np.exp(-3*self.a*t_full)  # Граничное условие
                
            B_arr[0] = 1; C_arr[0] = 0
            F_arr[0] = np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
            
            # Условие Неймана на верхней границе
            if neumann_order == 1:
                A_arr[-1] = -1/hy
                B_arr[-1] = 1/hy
                F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                u_new[i, :] = self.tdma_solve(A_arr, B_arr, C_arr, F_arr, Ny)
            else:
                A_arr[-1] = -theta * sigma_y
                B_arr[-1] = 1 + 2 * theta * sigma_y
                C_arr[-1] = -theta * sigma_y
                F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                u_new[i, :] = self.solve_neumann_2nd_order(A_arr, B_arr, C_arr, F_arr, Ny, hy, method='lu')
            
            # Правая граница (x=Lx)
            i = Nx-1
            A_arr = np.zeros(Ny)
            B_arr = np.zeros(Ny)
            C_arr = np.zeros(Ny)
            F_arr = np.zeros(Ny)
            
            for j in range(1, Ny-1):
                A_arr[j] = -theta * sigma_y
                B_arr[j] = 1 + 2 * theta * sigma_y
                C_arr[j] = -theta * sigma_y
                F_arr[j] = 0  # Граничное условие
                
            B_arr[0] = 1; C_arr[0] = 0
            F_arr[0] = np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
            
            # Условие Неймана на верхней границе
            if neumann_order == 1:
                A_arr[-1] = -1/hy
                B_arr[-1] = 1/hy
                F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                u_new[i, :] = self.tdma_solve(A_arr, B_arr, C_arr, F_arr, Ny)
            else:
                A_arr[-1] = -theta * sigma_y
                B_arr[-1] = 1 + 2 * theta * sigma_y
                C_arr[-1] = -theta * sigma_y
                F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                u_new[i, :] = self.solve_neumann_2nd_order(A_arr, B_arr, C_arr, F_arr, Ny, hy, method='lu')
            
            u = u_new.copy()
            
            # Сохраняем историю если нужно
            if save_history:
                solutions_history.append(u.copy())
                times_history.append(t_full)
            
            # Вычисление погрешности
            if k % max(1, Nt//10) == 0 or k == Nt-1:
                u_analytical = self.analytical_solution(X, Y, t_full)
                error = np.abs(u - u_analytical).max()
                errors.append(error)
                times.append(t_full)
                print(f"Время {t_full:.3f}, макс. погрешность: {error:.2e}")
        
        if save_history:
            return X, Y, solutions_history, times_history, errors, times
        else:
            return X, Y, u, errors, times

    def method_fractional_steps(self, Nx, Ny, Nt, neumann_order=1, save_history=True):
        """
        Метод дробных шагов (МДШ) с выбором порядка точности для условия Неймана
        """
        print(f"Запуск МДШ: Nx={Nx}, Ny={Ny}, Nt={Nt}, порядок Неймана: {neumann_order}")
        
        # Шаги сетки
        hx = self.Lx / (Nx - 1)
        hy = self.Ly / (Ny - 1)
        tau = self.T / Nt
        
        # Сетка
        x = np.linspace(0, self.Lx, Nx)
        y = np.linspace(0, self.Ly, Ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Инициализация решения
        u = self.initial_condition(X, Y)
        
        # Коэффициенты
        sigma_x = self.a * tau / hx**2
        sigma_y = self.a * tau / hy**2
        
        errors = []
        times = []
        
        # Сохраняем историю решений если нужно
        if save_history:
            solutions_history = [u.copy()]
            times_history = [0.0]
        
        for k in range(Nt):
            t = k * tau
            t_half = (k + 0.5) * tau
            t_full = (k + 1) * tau
            
            # ПЕРВЫЙ ДРОБНЫЙ ШАГ (неявно по x)
            u_half = np.zeros_like(u)
            
            for j in range(Ny):
                A = np.zeros(Nx)
                B = np.zeros(Nx)
                C = np.zeros(Nx)
                F = np.zeros(Nx)
                
                for i in range(1, Nx-1):
                    A[i] = -sigma_x
                    B[i] = 1 + 2 * sigma_x
                    C[i] = -sigma_x
                    F[i] = u[i,j]
                
                # Граничные условия по x
                B[0] = 1; C[0] = 0
                F[0] = np.cosh(y[j]) * np.exp(-3*self.a*t_half)
                
                A[-1] = 0; B[-1] = 1
                F[-1] = 0
                
                u_half[:, j] = self.tdma_solve(A, B, C, F, Nx)
            
            # ВТОРОЙ ДРОБНЫЙ ШАГ (неявно по y)
            u_new = np.zeros_like(u)
            
            for i in range(Nx):
                A_arr = np.zeros(Ny)
                B_arr = np.zeros(Ny)
                C_arr = np.zeros(Ny)
                F_arr = np.zeros(Ny)
                
                for j in range(1, Ny-1):
                    A_arr[j] = -sigma_y
                    B_arr[j] = 1 + 2 * sigma_y
                    C_arr[j] = -sigma_y
                    F_arr[j] = u_half[i,j]
                
                # Граничные условия по y
                B_arr[0] = 1; C_arr[0] = 0
                F_arr[0] = np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                
                # Условие Неймана на верхней границе
                if neumann_order == 1:
                    # Схема 1-го порядка
                    A_arr[-1] = -1/hy
                    B_arr[-1] = 1/hy
                    F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                    
                    u_new[i, :] = self.tdma_solve(A_arr, B_arr, C_arr, F_arr, Ny)
                else:
                    # Схема 2-го порядка - используем специальный решатель
                    A_arr[-1] = -sigma_y
                    B_arr[-1] = 1 + 2 * sigma_y
                    C_arr[-1] = -sigma_y
                    F_arr[-1] = 0.75 * np.cos(2*x[i]) * np.exp(-3*self.a*t_full)
                    
                    u_new[i, :] = self.solve_neumann_2nd_order(A_arr, B_arr, C_arr, F_arr, Ny, hy, method='lu')
            
            u = u_new.copy()
            
            # Сохраняем историю если нужно
            if save_history:
                solutions_history.append(u.copy())
                times_history.append(t_full)
            
            # Вычисление погрешности
            if k % max(1, Nt//10) == 0 or k == Nt-1:
                u_analytical = self.analytical_solution(X, Y, t_full)
                error = np.abs(u - u_analytical).max()
                errors.append(error)
                times.append(t_full)
                print(f"Время {t_full:.3f}, макс. погрешность: {error:.2e}")
        
        if save_history:
            return X, Y, solutions_history, times_history, errors, times
        else:
            return X, Y, u, errors, times

def visualize_results(solver, X, Y, u_numerical, u_analytical, method_name, t):
    """Визуализация результатов"""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. 3D график численного решения
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_numerical, cmap='viridis', alpha=0.8)
    ax1.set_title(f'{method_name}: Численное решение\nпри t = {t:.3f}')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u(x,y,t)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # # 2. 3D график аналитического решения
    # ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # surf2 = ax2.plot_surface(X, Y, u_analytical, cmap='plasma', alpha=0.8)
    # ax2.set_title(f'Аналитическое решение\nпри t = {t:.3f}')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # ax2.set_zlabel('u(x,y,t)')
    # fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    # 3. 3D график погрешности
    ax3 = fig.add_subplot(2, 3, 2, projection='3d')
    error = np.abs(u_numerical - u_analytical)
    surf3 = ax3.plot_surface(X, Y, error, cmap='hot', alpha=0.8)
    ax3.set_title(f'Погрешность {method_name}\nпри t = {t:.3f}')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('Погрешность')
    fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    # 4. Срезы по x при различных y
    ax4 = fig.add_subplot(2, 3, 3)
    y_indices = [0, Y.shape[1]//4, Y.shape[1]//2, 3*Y.shape[1]//4, -1]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for idx, color in zip(y_indices, colors):
        y_val = Y[0, idx]
        ax4.plot(X[:, 0], u_numerical[:, idx], 
                color=color, linestyle='--', linewidth=2, 
                label=f'Числ. y={y_val:.3f}')
        ax4.plot(X[:, 0], u_analytical[:, idx], 
                color=color, linestyle='-', alpha=0.7, 
                label=f'Анал. y={y_val:.3f}')
    
    ax4.set_title('Срезы по x при различных y')
    ax4.set_xlabel('x')
    ax4.set_ylabel('u(x,y,t)')
    ax4.legend()
    ax4.grid(True)
    
    # 5. Срезы по y при различных x
    ax5 = fig.add_subplot(2, 3, 4)
    x_indices = [0, X.shape[0]//4, X.shape[0]//2, 3*X.shape[0]//4, -1]
    
    for idx, color in zip(x_indices, colors):
        x_val = X[idx, 0]
        ax5.plot(Y[0, :], u_numerical[idx, :], 
                color=color, linestyle='--', linewidth=2, 
                label=f'Числ. x={x_val:.3f}')
        ax5.plot(Y[0, :], u_analytical[idx, :], 
                color=color, linestyle='-', alpha=0.7, 
                label=f'Анал. x={x_val:.3f}')
    
    ax5.set_title('Срезы по y при различных x')
    ax5.set_xlabel('y')
    ax5.set_ylabel('u(x,y,t)')
    ax5.legend()
    ax5.grid(True)
    
    # 6. Тепловая карта погрешности
    ax6 = fig.add_subplot(2, 3, 5)
    im = ax6.imshow(error.T, extent=[0, solver.Lx, 0, solver.Ly], 
                   origin='lower', cmap='hot', aspect='auto')
    ax6.set_title('Тепловая карта погрешности')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    fig.colorbar(im, ax=ax6, shrink=0.8)
    
    plt.tight_layout()
    plt.show()

def get_solution_at_time(solutions_history, times_history, desired_time):
    """Получить численное решение в заданное время"""
    # Находим ближайший временной слой
    time_index = min(range(len(times_history)), key=lambda i: abs(times_history[i] - desired_time))
    return solutions_history[time_index], times_history[time_index]

def interactive_time_selection(solver, X, Y, solutions_history, times_history, method_name):
    """Интерактивный выбор времени для визуализации"""
    print(f"\n{'='*50}")
    print(f"ИНТЕРАКТИВНАЯ ВИЗУАЛИЗАЦИЯ - {method_name}")
    print(f"{'='*50}")
    print(f"Доступный диапазон времени: 0.0 - {solver.T:.2f}")
    print("Введите время для визуализации или 'exit' для выхода")
    
    while True:
        try:
            user_input = input("\nВведите время t (0.0-1.0): ").strip()
            
            if user_input.lower() == 'exit':
                print("Выход из интерактивного режима...")
                break
                
            desired_time = float(user_input)
            
            if desired_time < 0 or desired_time > solver.T:
                print(f"Ошибка: время должно быть в диапазоне [0, {solver.T}]")
                continue
                
            # Получаем решение для выбранного времени
            u_numerical, actual_time = get_solution_at_time(solutions_history, times_history, desired_time)
            u_analytical = solver.analytical_solution(X, Y, actual_time)
            
            print(f"Ближайшее доступное время: {actual_time:.3f}")
            
            # Вычисляем погрешность
            error = np.abs(u_numerical - u_analytical).max()
            print(f"Максимальная погрешность: {error:.2e}")
            
            # Визуализация
            visualize_results(solver, X, Y, u_numerical, u_analytical, method_name, actual_time)
            
            print(f"\nГотово! Графики для t = {actual_time:.3f} построены.")
            print("Можете ввести новое время или 'exit' для выхода")
            
        except ValueError:
            print("Ошибка: введите число или 'exit'")
        except KeyboardInterrupt:
            print("\nВыход из программы...")
            break

def debug_solution(solver, Nx=21, Ny=21, Nt=50):
    """Отладочная функция для проверки решения"""
    print("=== ОТЛАДОЧНАЯ ИНФОРМАЦИЯ ===")
    
    # Простая сетка для отладки
    hx = solver.Lx / (Nx - 1)
    hy = solver.Ly / (Ny - 1)
    tau = solver.T / Nt
    
    x = np.linspace(0, solver.Lx, Nx)
    y = np.linspace(0, solver.Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Проверка начального условия
    u0 = solver.initial_condition(X, Y)
    u0_analytical = solver.analytical_solution(X, Y, 0)
    error_initial = np.abs(u0 - u0_analytical).max()
    print(f"Погрешность начального условия: {error_initial:.2e}")
    
    # Проверка граничных условий
    print("\nПроверка граничных условий при t=0:")
    print(f"Левая граница (x=0): u(0,y,0) = cosh(y)")
    print(f"  При y=0: {np.cosh(0):.6f}, ожидается: 1.000000")
    print(f"  При y=Ly: {np.cosh(solver.Ly):.6f}, ожидается: 1.500000")
    
    print(f"Правая граница (x=π/4): u(π/4,y,0) = 0")
    print(f"Нижняя граница (y=0): u(x,0,0) = cos(2x)")
    print(f"  При x=0: {np.cos(0):.6f}, ожидается: 1.000000")
    print(f"  При x=π/4: {np.cos(2*solver.Lx):.6f}, ожидается: 0.000000")
    
    # Проверка аналитического решения
    print(f"\nПроверка аналитического решения:")
    test_points = [(0, 0), (solver.Lx/2, solver.Ly/2), (solver.Lx, solver.Ly)]
    for x_val, y_val in test_points:
        u_anal = solver.analytical_solution(x_val, y_val, 0)
        u_anal_t1 = solver.analytical_solution(x_val, y_val, 1.0)
        print(f"  (x={x_val:.3f}, y={y_val:.3f}): u(0)={u_anal:.6f}, u(1)={u_anal_t1:.6f}")

def main():
    """Основная функция"""
    print("=== Решение двумерной параболической задачи ===")
    print("Уравнение: du/dt = a * (d²u/dx² + d²u/dy²)")
    print("Аналитическое решение: u(x,y,t) = cos(2x) * cosh(y) * exp(-3at)")
    
    # Создание решателя
    solver = Parabolic2DSolver(a=1.0)
    
    # Отладочная информация
    debug_solution(solver)
    
    # Базовые параметры сетки
    Nx, Ny, Nt = 31, 31, 100
    theta = 0.5
    
    print(f"\n=== МЕТОДЫ С 1-М ПОРЯДКОМ ===")
    
    # МПН 1-го порядка
    print(f"\n=== МПН (1-й порядок) ===")
    try:
        X_vd1, Y_vd1, solutions_vd1, times_vd1, errors_vd1, _ = solver.method_variable_directions(
            Nx, Ny, Nt, theta, neumann_order=1, save_history=True)
        
        # Интерактивный выбор времени для МПН 1-го порядка
        interactive_time_selection(solver, X_vd1, Y_vd1, solutions_vd1, times_vd1, "МПН (1-й порядок)")
        
    except Exception as e:
        print(f"Ошибка в МПН 1-го порядка: {e}")
    
    # МДШ 1-го порядка
    print(f"\n=== МДШ (1-й порядок) ===")
    try:
        X_fs1, Y_fs1, solutions_fs1, times_fs1, errors_fs1, _ = solver.method_fractional_steps(
            Nx, Ny, Nt, neumann_order=1, save_history=True)
        
        # Интерактивный выбор времени для МДШ 1-го порядка
        interactive_time_selection(solver, X_fs1, Y_fs1, solutions_fs1, times_fs1, "МДШ (1-й порядок)")
        
    except Exception as e:
        print(f"Ошибка в МДШ 1-го порядка: {e}")

    print(f"\n=== МЕТОДЫ С 2-М ПОРЯДКОМ ===")
    
    # МПН 2-го порядка
    print(f"\n=== МПН (2-й порядок) ===")
    try:
        X_vd2, Y_vd2, solutions_vd2, times_vd2, errors_vd2, _ = solver.method_variable_directions(
            Nx, Ny, Nt, theta, neumann_order=2, save_history=True)
        
        # Интерактивный выбор времени для МПН 2-го порядка
        interactive_time_selection(solver, X_vd2, Y_vd2, solutions_vd2, times_vd2, "МПН (2-й порядок)")
        
    except Exception as e:
        print(f"Ошибка в МПН 2-го порядка: {e}")
    
    # МДШ 2-го порядка
    print(f"\n=== МДШ (2-й порядок) ===")
    try:
        X_fs2, Y_fs2, solutions_fs2, times_fs2, errors_fs2, _ = solver.method_fractional_steps(
            Nx, Ny, Nt, neumann_order=2, save_history=True)
        
        # Интерактивный выбор времени для МДШ 2-го порядка
        interactive_time_selection(solver, X_fs2, Y_fs2, solutions_fs2, times_fs2, "МДШ (2-й порядок)")
        
    except Exception as e:
        print(f"Ошибка в МДШ 2-го порядка: {e}")

if __name__ == "__main__":
    main()