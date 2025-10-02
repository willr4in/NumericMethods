#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <limits>

#define EPS 1e-9  // Порог для сходимости и вывода значений
#define MAX_ITER 10000  // Максимальное количество итераций

// Функция для вычисления нормы ||X||, где X - матрица или вектор
double calc_norm(const std::vector<std::vector<double>>& X) {
    int n = X.size();
    double l2_norm = std::abs(X[0][0]);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            l2_norm = std::max(std::abs(X[i][j]), l2_norm);
        }
    }

    return l2_norm;
}

double calc_norm(const std::vector<double>& X) {
    int n = X.size();
    double l2_norm = std::abs(X[0]);

    for (int i = 0; i < n; ++i) {
        l2_norm = std::max(std::abs(X[i]), l2_norm);
    }

    return l2_norm;
}

// Функция для вычитания двух векторов
std::vector<double> subtract_vectors(const std::vector<double>& v1, const std::vector<double>& v2) {
    int n = v1.size();
    std::vector<double> result(n);

    for (int i = 0; i < n; ++i) {
        result[i] = v1[i] - v2[i];
    }

    return result;
}

// Функция для проверки диагональной доминантности
bool is_diagonally_dominant(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    for (int i = 0; i < n; ++i) {
        double sum = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                sum += std::abs(A[i][j]);
            }
        }
        if (std::abs(A[i][i]) <= sum) {
            return false;
        }
    }
    return true;
}

// Итеративный метод решения уравнения Ax = b
std::pair<std::vector<double>, int> solve_iterative(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int n = A.size();
    
    std::vector<std::vector<double>> alpha(n, std::vector<double>(n, 0));
    std::vector<double> beta(n, 0);
    
    // 1. Ax = b -> x_k = alpha * x_(k-1) + beta
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                alpha[i][j] = 0;
            } else {
                alpha[i][j] = -A[i][j] / A[i][i];
            }
        }
        beta[i] = b[i] / A[i][i];
    }

    // 2. Итерируем
    int iterations = 0;
    std::vector<double> cur_x = beta;
    bool converge = false;
    
    while (!converge && iterations < MAX_ITER) {
        std::vector<double> prev_x = cur_x;
        cur_x.resize(n);
        
        for (int i = 0; i < n; ++i) {
            cur_x[i] = beta[i];
            for (int j = 0; j < n; ++j) {
                cur_x[i] += alpha[i][j] * prev_x[j];
            }
        }
        
        iterations++;
        if (calc_norm(subtract_vectors(cur_x, prev_x)) <= EPS) {
            converge = true;
        }
    }
    
    return {cur_x, iterations};
}

// Метод Зейделя для решения уравнения Ax = b
std::pair<std::vector<double>, int> solve_seidel(const std::vector<std::vector<double>>& A, const std::vector<double>& b) {
    int n = A.size();
    
    // 1. Ax = b -> x = alpha * x + beta
    std::vector<std::vector<double>> alpha(n, std::vector<double>(n, 0));
    std::vector<double> beta(n, 0);
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                alpha[i][j] = 0;
            } else {
                alpha[i][j] = -A[i][j] / A[i][i];
            }
        }
        beta[i] = b[i] / A[i][i];
    }

    // 2. Итерируем
    int iterations = 0;
    std::vector<double> cur_x = beta;
    bool converge = false;
    
    while (!converge && iterations < MAX_ITER) {
        std::vector<double> prev_x = cur_x;
        for (int i = 0; i < n; ++i) {
            cur_x[i] = beta[i];
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    cur_x[i] += alpha[i][j] * cur_x[j];
                }
            }
        }
        
        iterations++;
        if (calc_norm(subtract_vectors(cur_x, prev_x)) <= EPS) {
            converge = true;
        }
    }
    
    return {cur_x, iterations};
}

// Функция для вывода результатов
void print_result(const std::vector<double>& result) {
    for (double x : result) {
        std::cout << x << " ";  // Просто выводим все значения как есть
    }
    std::cout << std::endl;
}

// Проверка подстановкой для метода Якоби и Зейделя
void test_results(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x) {
    int n = A.size();
    
    for (int i = 0; i < n; ++i) {
        // Вычисление левой части уравнения Ax = b
        double left = 0;
        for (int j = 0; j < n; ++j) {
            left += A[i][j] * x[j];
        }
        
        double right = b[i];
        double diff = std::abs(left - right);
        
        std::cout << "Подстановка для уравнения " << i + 1 << ":\n";
        std::cout << "Левая часть (вычисленная): " << left << "\n";
        std::cout << "Правая часть (ожидаемая): " << right << "\n";
        std::cout << "Разница: " << diff << "\n\n";
    }
}

int main() {
    std::vector<std::vector<double>> A = {
        {-25, 4, -4, 9},
        {-9, 21, 5, -6},
        {9, 2, 19, -7},
        {-7, 4, -7, 25}
    };
    
    std::vector<double> b = {86, 29, 28, 68};

    // std::vector<std::vector<double>> A = {
    //     {20, 5, 7, 1},
    //     {-1, 13, 0, -7},
    //     {4, -6, 17, 5},
    //     {-9, 8, 4, -25}
    // };
    
    // std::vector<double> b = {-117, -1, 49, -21};
    
    // Проверка на диагональную доминантность
    if (!is_diagonally_dominant(A)) {
        std::cout << "Матрица не является диагонально доминирующей.\n";
        return 1;
    }

    // Решение методом Якоби
    std::vector<double> res_yacobi;
    int iters_yacobi;
    res_yacobi = solve_iterative(A, b).first;
    iters_yacobi = solve_iterative(A, b).second;

    // Решение методом Зейделя
    std::vector<double> res_seidel;
    int iters_seidel;
    res_seidel = solve_seidel(A, b).first;
    iters_seidel = solve_seidel(A, b).second;

    // Вывод результатов
    std::cout << "-------------\n";
    std::cout << "Метод простых итераций\n";
    print_result(res_yacobi);
    std::cout << "Итерации: " << iters_yacobi << "\n";
    std::cout << "Проверка подстановкой:\n";
    test_results(A, b, res_yacobi);
    
    std::cout << "-------------\n";
    std::cout << "Метод Зейделя\n";
    print_result(res_seidel);
    std::cout << "Итерации: " << iters_seidel << "\n";
    std::cout << "Проверка подстановкой:\n";
    test_results(A, b, res_seidel);

    return 0;
}
