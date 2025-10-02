#include <iostream>
#include <vector>
#include <cmath>

// Функция для решения системы методом Гаусса
std::vector<double> gaussElimination(std::vector<std::vector<double>>& A, std::vector<double>& b) {
    int n = A.size();

    // Прямой ход
    for (int i = 0; i < n; i++) {
        // Находим максимальный элемент в i-том столбце для перестановки
        int max_row = i;
        for (int j = i + 1; j < n; j++) {
            if (std::abs(A[j][i]) > std::abs(A[max_row][i])) {
                max_row = j;
            }
        }

        // Перестановка строк, если нужно
        if (i != max_row) {
            std::swap(A[i], A[max_row]);
            std::swap(b[i], b[max_row]);
        }

        // Приведение к верхнему треугольному виду
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }

    // Обратный ход
    std::vector<double> x(n);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
        x[i] /= A[i][i];
    }

    return x;
}

// Функция для проверки решения системы уравнений подстановкой
void check_solution(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& solution) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        double left = 0;
        for (int j = 0; j < n; ++j) {
            left += A[i][j] * solution[j];
        }
        std::cout << "Подстановка для уравнения " << i + 1 << ":\n";
        std::cout << "Левая часть (вычисленная): " << left << "\n";
        std::cout << "Правая часть (ожидаемая): " << b[i] << "\n";
        std::cout << "Разница: " << std::fabs(left - b[i]) << "\n\n";
    }
}

int main() {
    // Исходная матрица A и вектор b
    std::vector<std::vector<double>> A = {
        {3, -8, 1, -7},
        {6, 4, 8, 5},
        {-1, 1, -9, -3},
        {-6, 6, 9, -4}
    };
    std::vector<double> b = {96, -13, -54, 82};

    std::vector<std::vector<double>> A_copy = A;

    std::vector<double> b_copy = b; 
    // Решение системы методом Гаусса
    std::vector<double> solution = gaussElimination(A, b);

    // Вывод решения
    std::cout << "Решение системы: \n";
    for (double x_i : solution) {
        std::cout << x_i << " ";
    }
    std::cout << std::endl;

    // Проверка решения подстановкой
    std::cout << "----------------------\nПроверка подстановкой:\n";
    check_solution(A_copy, b_copy, solution);

    return 0;
}
