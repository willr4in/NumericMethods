#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <stdexcept>  // Для исключений

const double EPS = 1e-6;  // Увеличиваем точность для ускорения работы

// Функция для проверки симметричности матрицы
void check_symmetry(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(A[i][j] - A[j][i]) > EPS) {
                throw std::invalid_argument("Матрица несимметрична");
            }
        }
    }
}

// Функция для нахождения максимального по модулю элемента в верхнем треугольнике матрицы
std::pair<int, int> find_max_upper_element(const std::vector<std::vector<double>>& X) {
    int n = X.size();
    int i_max = 0, j_max = 1;
    double max_elem = std::abs(X[0][1]);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(X[i][j]) > max_elem) {
                max_elem = std::abs(X[i][j]);
                i_max = i;
                j_max = j;
            }
        }
    }

    return {i_max, j_max};
}

// Функция для вычисления нормы матрицы
double matrix_norm(const std::vector<std::vector<double>>& X) {
    double norm = 0.0;
    int n = X.size();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            norm += X[i][j] * X[i][j];
        }
    }
    return std::sqrt(norm);
}

// Функция для вычисления собственного значения и собственного вектора с помощью метода вращений
void rotation_method(std::vector<std::vector<double>>& A, std::vector<double>& eigen_values, std::vector<std::vector<double>>& eigen_vectors, int& iterations) {
    int n = A.size();
    eigen_vectors = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    // Инициализация собственных векторов (единичная матрица)
    for (int i = 0; i < n; ++i) {
        eigen_vectors[i][i] = 1.0;
    }
    iterations = 0;

    while (matrix_norm(A) > EPS) {
        // Нахождение индексов максимального элемента в верхнем треугольнике
        auto [i_max, j_max] = find_max_upper_element(A);

        // Вычисление угла поворота
        double phi = 0.0;
        if (A[i_max][i_max] == A[j_max][j_max]) {
            phi = M_PI / 4;
        } else {
            double denom = A[i_max][i_max] - A[j_max][j_max];
            if (std::abs(denom) < EPS) {
                phi = M_PI / 4;  // Если разница слишком мала, используем фиксированный угол
            } else {
                phi = 0.5 * std::atan(2 * A[i_max][j_max] / denom);
            }
        }

        // Построение матрицы вращений U
        std::vector<std::vector<double>> U(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            U[i][i] = 1.0;  // Единичная матрица
        }
        U[i_max][j_max] = -std::sin(phi);
        U[j_max][i_max] = std::sin(phi);
        U[i_max][i_max] = std::cos(phi);
        U[j_max][j_max] = std::cos(phi);

        // Обновление матрицы A
        std::vector<std::vector<double>> A_new(n, std::vector<double>(n, 0.0));
        // A_new = U^T * A * U
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    A_new[i][j] += U[k][i] * A[k][j];
                }
            }
        }

        std::vector<std::vector<double>> A_final(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    A_final[i][j] += A_new[i][k] * U[k][j];
                }
            }
        }

        A = A_final;  // Обновление A

        // Обновление собственных векторов
        std::vector<std::vector<double>> eigen_vectors_new(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                    eigen_vectors_new[i][j] += eigen_vectors[i][k] * U[k][j];
                }
            }
        }
        eigen_vectors = eigen_vectors_new;  // Обновление собственных векторов

        iterations++;
    }

    // Собственные значения - элементы на главной диагонали A
    eigen_values.resize(n);
    for (int i = 0; i < n; ++i) {
        eigen_values[i] = A[i][i];
    }
}

// Функция для проверки, равны ли два вектора с точностью EPS
bool are_vectors_equal(const std::vector<double>& v1, const std::vector<double>& v2, double epsilon) {
    if (v1.size() != v2.size()) {
        return false;
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        if (std::abs(v1[i] - v2[i]) > epsilon) {
            return false;
        }
    }

    return true;
}

// Функция для проверки уравнения A * v = λ * v и вывода результата
void check_eigenvector_equation(const std::vector<std::vector<double>>& A, const std::vector<double>& eigen_values, const std::vector<std::vector<double>>& eigen_vectors, double epsilon) {
    int n = A.size();

    for (int i = 0; i < n; ++i) {
        std::vector<double> v(n);
        for (int j = 0; j < n; ++j) {
            v[j] = eigen_vectors[j][i]; 
        }

        std::vector<double> Av(n, 0.0);
        std::vector<double> lambda_v(n, 0.0);

        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                Av[j] += A[j][k] * v[k];
            }
            lambda_v[j] = eigen_values[i] * v[j];
        }

        std::cout << "\nВектор " << i + 1 << ":\n";
        std::cout << "A * v =   ";
        for (double val : Av) {
            std::cout << val << " ";
        }
        std::cout << "\nλ * v =   ";
        for (double val : lambda_v) {
            std::cout << val << " ";
        }

        // Проверка на равенство Av и λv с точностью EPS
        if (are_vectors_equal(Av, lambda_v, epsilon)) {
            std::cout << "\nA * v и λ * v равны.\n";
        } else {
            std::cout << "\nA * v и λ * v не равны.\n";
        }

        std::cout << '\n';
    }
}

int main() {
    try {
        // Фиксированная матрица A
        std::vector<std::vector<double>> A = {
            {-8,  -4,  8},
            {-4,  -3,  9},
            {8,  9,  -5}
        };

        // std::vector<std::vector<double>> A = {
        //     {0,  -7,  7},
        //     {-7,  -9,  -5},
        //     {7,  -5,  -1}
        // };

        // Создаем копию матрицы A для использования в проверке
        std::vector<std::vector<double>> A_copy = A;

        // Проверка симметричности матрицы
        check_symmetry(A);

        // Размерность матрицы
        int n = A.size();

        // Вычисления
        std::vector<double> eigen_values;
        std::vector<std::vector<double>> eigen_vectors;
        int iterations = 0;

        rotation_method(A, eigen_values, eigen_vectors, iterations);

        // Вывод результатов
        std::cout << "Собственные значения:" << '\n';
        for (double value : eigen_values) {
            std::cout << value << " ";
        }
        std::cout << '\n';

        std::cout << "Собственные векторы:" << '\n';
        for (const auto& row : eigen_vectors) {
            for (double value : row) {
                std::cout << value << " ";
            }
            std::cout << '\n';
        }

        std::cout << "Итерации: " << iterations << '\n';

        // Проверка уравнения A * v = λ * v
        std::cout << "\nПроверка уравнения A * v = λ * v:\n";
        check_eigenvector_equation(A_copy, eigen_values, eigen_vectors, EPS);

    } catch (const std::invalid_argument& e) {
        std::cout << "Ошибка: " << e.what() << '\n';
    }

    return 0;
}
