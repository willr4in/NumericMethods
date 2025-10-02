#include <iostream>
#include <vector>
#include <cmath>
#include <complex>

// Функция для вычисления L2-нормы вектора
double L2_norm(const std::vector<double>& vec) {
    double norm = 0.0;
    for (double num : vec) {
        norm += num * num;
    }
    return std::sqrt(norm);
}

// Функция для умножения двух матриц
std::vector<std::vector<double>> multiply_matrices(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int n = A.size();
    std::vector<std::vector<double>> result(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Функция для умножения матрицы на вектор
std::vector<double> multiply_matrix_vector(const std::vector<std::vector<double>>& A, const std::vector<double>& v) {
    int n = A.size();
    std::vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

// Функция для получения матрицы Хаусхолдера
std::vector<std::vector<double>> householder_matrix(const std::vector<std::vector<double>>& A, int col) {
    int n = A.size();
    std::vector<double> v(n, 0.0);
    double norm = 0.0;

    for (int i = col; i < n; ++i) {
        v[i] = A[i][col];
        norm += v[i] * v[i];
    }

    norm = std::sqrt(norm);
    v[col] += (v[col] > 0) ? norm : -norm;

    double vTv = 0.0;
    for (int i = col; i < n; ++i) {
        vTv += v[i] * v[i];
    }

    std::vector<std::vector<double>> H(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        H[i][i] = 1.0;
    }

    for (int i = col; i < n; ++i) {
        for (int j = col; j < n; ++j) {
            H[i][j] -= (2.0 / vTv) * v[i] * v[j];
        }
    }

    return H;
}

// Функция для выполнения QR-разложения
void QR(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& Q, std::vector<std::vector<double>>& R) {
    int n = A.size();
    Q = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    R = A;  // Начинаем с исходной матрицы

    // Инициализация Q как единичной матрицы
    for (int i = 0; i < n; ++i) {
        Q[i][i] = 1.0;
    }

    for (int i = 0; i < n - 1; ++i) {
        std::vector<std::vector<double>> H = householder_matrix(R, i);

        // Умножаем Q на H
        Q = multiply_matrices(Q, H);
        // Умножаем R на H
        R = multiply_matrices(H, R);
    }
}

// Функция для вычисления определителя (правильная реализация)
double compute_determinant(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    if (n == 1) return A[0][0];
    if (n == 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    
    double det = 0.0;
    for (int j = 0; j < n; ++j) {
        // Создаем минор
        std::vector<std::vector<double>> minor(n-1, std::vector<double>(n-1));
        for (int i = 1; i < n; ++i) {
            int col_idx = 0;
            for (int k = 0; k < n; ++k) {
                if (k != j) {
                    minor[i-1][col_idx++] = A[i][k];
                }
            }
        }
        det += (j % 2 == 0 ? 1 : -1) * A[0][j] * compute_determinant(minor);
    }
    return det;
}

// Функция для нахождения собственного значения с помощью детерминанта A - λE
double compute_determinant_of_A_minus_lambda_E(const std::vector<std::vector<double>>& A, double lambda) {
    int n = A.size();
    std::vector<std::vector<double>> A_minus_lambda_E(n, std::vector<double>(n));

    // A - λE
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A_minus_lambda_E[i][j] = A[i][j] - ((i == j) ? lambda : 0);
        }
    }

    return compute_determinant(A_minus_lambda_E);
}

// Функция для проверки сходимости (все внедиагональные элементы)
double check_convergence(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    double off_diagonal_sum = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                off_diagonal_sum += std::abs(A[i][j]);
            }
        }
    }
    return off_diagonal_sum;
}

// Функция для нахождения собственных значений методом QR-итераций
std::vector<std::complex<double>> find_eigenvalues(std::vector<std::vector<double>>& A) {
    int n = A.size();
    std::vector<std::vector<double>> Q, R;
    
    // Сохраняем исходную матрицу для проверок
    std::vector<std::vector<double>> A_original = A;

    // QR-разложение до сходимости
    int max_iterations = 1000;
    for (int iter = 0; iter < max_iterations; ++iter) {
        QR(A, Q, R);
        
        // Обновляем A = R * Q
        A = multiply_matrices(R, Q);

        // Проверяем сходимость (все внедиагональные элементы)
        double off_diagonal_sum = check_convergence(A);

        // Если матрица становится почти диагональной, прекращаем итерации
        if (off_diagonal_sum < 1e-10) {
            std::cout << "Сходимость достигнута на итерации " << iter + 1 << std::endl;
            break;
        }
        
        if (iter == max_iterations - 1) {
            std::cout << "Достигнуто максимальное число итераций" << std::endl;
        }
    }

    // Правильное извлечение собственных значений с учетом комплексных пар
    std::vector<std::complex<double>> eigenvalues;
    int i = 0;
    while (i < n) {
        if (i < n - 1 && std::abs(A[i+1][i]) > 1e-10) {
            // Обнаружен блок 2x2 - возможны комплексные собственные значения
            double a = A[i][i], b = A[i][i+1];
            double c = A[i+1][i], d = A[i+1][i+1];
            
            double trace = a + d;
            double determinant = a * d - b * c;
            double discriminant = trace * trace - 4 * determinant;
            
            if (discriminant < 0) {
                // Комплексно-сопряженная пара
                double real_part = trace / 2.0;
                double imag_part = std::sqrt(-discriminant) / 2.0;
                eigenvalues.emplace_back(real_part, imag_part);
                eigenvalues.emplace_back(real_part, -imag_part);
            } else {
                // Два действительных собственных значения
                eigenvalues.emplace_back(a, 0.0);
                eigenvalues.emplace_back(d, 0.0);
            }
            i += 2;
        } else {
            // Действительное собственное значение
            eigenvalues.emplace_back(A[i][i], 0.0);
            i += 1;
        }
    }

    return eigenvalues;
}

// Функция для вывода собственных значений
void print_eigenvalues(const std::vector<std::complex<double>>& eigenvalues) {
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        const auto& val = eigenvalues[i];
        if (std::abs(val.imag()) < 1e-10) {
            std::cout << "λ" << i+1 << " = " << val.real() << " (действительное)" << std::endl;
        } else {
            std::cout << "λ" << i+1 << " = " << val.real() << " + " << val.imag() << "i" << std::endl;
            std::cout << "λ" << i+2 << " = " << val.real() << " - " << val.imag() << "i" << std::endl;
            i++; // Пропускаем следующий, так как это комплексно-сопряженная пара
        }
    }
}

// Функция для проверки собственных значений через определитель
void verify_eigenvalues(const std::vector<std::vector<double>>& A_original, 
                       const std::vector<std::complex<double>>& eigenvalues) {
    std::cout << "\nПроверка собственных значений (det(A - λI) должно быть ≈ 0):" << std::endl;
    
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        const auto& lambda = eigenvalues[i];
        
        // Для действительных собственных значений
        if (std::abs(lambda.imag()) < 1e-10) {
            double det = compute_determinant_of_A_minus_lambda_E(A_original, lambda.real());
            std::cout << "det(A - λI) = " << det;
            if (std::abs(det) < 1e-5) {
                std::cout << " Выполняется " << std::endl;
            } else {
                std::cout << " Не выполняется " << std::endl;
            }
        }
    }
}

int main() {
    // Пример матрицы
    // std::vector<std::vector<double>> A = {
    //     {5, 8, -2},
    //     {7, -2, -4},
    //     {5, 8, -1}
    // };

    std::vector<std::vector<double>> A = {
        {-3, 1, -1},
        {6, 9, -4},
        {5, -4, -8}
    };

    // Сохраняем копию исходной матрицы для проверок
    std::vector<std::vector<double>> A_original = A;

    std::cout << "Исходная матрица A:" << std::endl;
    for (const auto& row : A_original) {
        for (double val : row) {
            std::cout << val << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Находим собственные значения
    std::vector<std::complex<double>> eigenvalues = find_eigenvalues(A);

    // Выводим собственные значения
    std::cout << "\nСобственные значения:" << std::endl;
    print_eigenvalues(eigenvalues);

    // Проверяем собственные значения
    verify_eigenvalues(A_original, eigenvalues);

    return 0;
}
