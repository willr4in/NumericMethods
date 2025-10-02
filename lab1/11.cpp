#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

// LU-разложение с выбором главного элемента и заполнением массива перестановок
void LU(const std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& L, std::vector<std::vector<double>>& U, std::vector<int>& P) {
    int n = A.size();
    L = std::vector<std::vector<double>>(n, std::vector<double>(n, 0));
    U = A;
    P = std::vector<int>(n);

    for (int i = 0; i < n; i++) {
        P[i] = i;
    }

    for (int k = 0; k < n - 1; k++) {
        int max_row = k;
        for (int i = k + 1; i < n; i++) {
            if (std::abs(U[i][k]) > std::abs(U[max_row][k])) {
                max_row = i;
            }
        }

        if (std::abs(U[max_row][k]) < 1e-12) {
            throw std::runtime_error("Обнаружен нулевой или почти нулевой ведущий элемент на шаге " + std::to_string(k) + ". Матрица вырождена.");
        }

        if (max_row != k) {
            std::swap(U[k], U[max_row]);
            std::swap(P[k], P[max_row]);

            for (int j = 0; j < k; ++j) {
                std::swap(L[k][j], L[max_row][j]);
            }
        }

        for (int i = k + 1; i < n; i++) {
            L[i][k] = U[i][k] / U[k][k];
            for (int j = k; j < n; j++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        L[i][i] = 1;
    }
}

std::vector<double> solve_Ly_b(const std::vector<std::vector<double>>& L, const std::vector<double>& b) {
    int n = L.size();
    std::vector<double> y(n, 0);

    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < i; j++) {
            sum += L[i][j] * y[j];
        }
        y[i] = (b[i] - sum) / L[i][i];
    }

    return y;
}

std::vector<double> solve_Ux_y(const std::vector<std::vector<double>>& U, const std::vector<double>& y) {
    int n = U.size();
    std::vector<double> x(n, 0);

    for (int i = n - 1; i >= 0; i--) {
        double sum = 0;
        for (int j = n - 1; j > i; j--) {
            sum += U[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / U[i][i];
    }

    return x;
}

std::vector<double> solve_system(const std::vector<std::vector<double>>& L, const std::vector<std::vector<double>>& U, const std::vector<double>& b, const std::vector<int>& P) {
    int n = L.size();
    std::vector<double> permuted_b(n);

    for (int i = 0; i < n; i++) {
        permuted_b[i] = b[P[i]];
    }

    std::vector<double> y = solve_Ly_b(L, permuted_b);
    return solve_Ux_y(U, y);
}

double determinant(const std::vector<std::vector<double>>& U) {
    double det = 1;
    for (int i = 0; i < U.size(); i++) {
        det *= U[i][i];
    }
    return det;
}

std::vector<std::vector<double>> inverse_matr(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> A_inversion(n, std::vector<double>(n));

    for (int i = 0; i < n; i++) {
        std::vector<std::vector<double>> L, U;
        std::vector<int> P;
        std::vector<double> e(n, 0);
        e[i] = 1;

        LU(A, L, U, P);
        std::vector<double> col = solve_system(L, U, e, P);

        for (int j = 0; j < n; j++) {
            A_inversion[j][i] = col[j];
        }
    }

    return A_inversion;
}

void test_solution(const std::vector<std::vector<double>> A, std::vector<double> b, const std::vector<double>& x) {
    for (int i = 0; i < A.size(); ++i) {
        double left = 0;
        for (int j = 0; j < A[i].size(); ++j) {
            left += A[i][j] * x[j];
        }
        std::cout << "Подстановка для уравнения " << i + 1 << ":\n";
        std::cout << "Левая часть (вычисленная): " << left << "\n";
        std::cout << "Правая часть (ожидаемая): " << b[i] << "\n";
        std::cout << "Разница: " << std::fabs(left - b[i]) << "\n\n";
    }
}

int main() {
    try {
        // std::vector<std::vector<double>> A = {
        //     {3, -8, 1, -7},
        //     {6, 4, 8, 5},
        //     {-1, 1, -9, -3},
        //     {-6, 6, 9, -4}
        // };

        // std::vector<double> b = {96, -13, -54, 82};

        std::vector<std::vector<double>> A = {
            {-7, -2, -1, -4},
            {-4, 6, 0, -4},
            {-8, 2, -9, -3},
            {0, 0, -7, 1}
        };

        std::vector<double> b = {-12, 22, 51, 49};

        std::vector<std::vector<double>> L, U;
        std::vector<int> P;

        std::vector<std::vector<double>> A_copy = A;

        LU(A, L, U, P);

        // Проверка на вырожденность
        for (int i = 0; i < U.size(); ++i) {
            if (std::abs(U[i][i]) < 1e-8) {
                throw std::runtime_error("Матрица вырождена (det A = 0). Решение системы и обратная матрица невозможны.");
            }
        }

        double tolerance = 1e-12;
        for (auto& row : U) {
            for (double& elem : row) {
                if (std::abs(elem) < tolerance) {
                    elem = 0;
                }
            }
        }

        std::vector<double> solution = solve_system(L, U, b, P);

        std::cout << "LU разложение:\nL:\n";
        for (const auto& row : L) {
            for (double elem : row) std::cout << elem << " ";
            std::cout << '\n';
        }

        std::cout << "U:\n";
        for (const auto& row : U) {
            for (double elem : row) std::cout << elem << " ";
            std::cout << '\n';
        }

        std::cout << "----------------------\nРешение системы:\n";
        std::cout << "x: ";
        for (double x_i : solution) std::cout << x_i << " ";
        std::cout << "\n\nПроверка подстановкой:\n";
        test_solution(A_copy, b, solution);

        std::cout << "----------------------\nОпределитель матрицы A: " << determinant(U) << '\n';

        std::cout << "Обратная матрица A:\n";
        std::vector<std::vector<double>> A_inv = inverse_matr(A);
        for (const auto& row : A_inv) {
            for (double elem : row) std::cout << elem << " ";
            std::cout << '\n';
        }

        std::cout << "----------------------\nПроверка A * A⁻¹ = E:\n";
        std::vector<std::vector<double>> identity(A.size(), std::vector<double>(A.size(), 0.0));
        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A.size(); ++j) {
                for (int k = 0; k < A.size(); ++k) {
                    identity[i][j] += A[i][k] * A_inv[k][j];
                }
            }
        }

        bool is_identity = true;
        double eps = 1e-8;
        for (int i = 0; i < A.size(); ++i) {
            for (int j = 0; j < A.size(); ++j) {
                double expected = (i == j) ? 1.0 : 0.0;
                if (std::fabs(identity[i][j] - expected) > eps) {
                    is_identity = false;
                }
                std::cout << identity[i][j] << " ";
            }
            std::cout << '\n';
        }
        std::cout << (is_identity ? "A * A⁻¹ ≈ E\n" : "A * A⁻¹ ≠ E\n");

    } catch (const std::exception& e) {
        std::cerr << "\nОШИБКА: " << e.what() << "\n";
    }

    return 0;
}
