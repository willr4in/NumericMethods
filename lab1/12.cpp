#include <iostream>
#include <vector>
#include <cmath>  // Для вычисления погрешности

bool is_tridiagonal(const std::vector<std::vector<double>>& A) {
    int n = A.size();
    
    // Проверяем, что матрица действительно тридиагональна
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            // Если элемент не на главной, верхней или нижней диагонали, он должен быть равен 0
            if (i != j && i != j - 1 && i != j + 1) {
                if (A[i][j] != 0) {
                    return false; // Матрица не тридиагональна
                }
            }
        }
    }
    return true;
}

std::vector<double> tridiagonal_solve(const std::vector<double>& a, const std::vector<double>& b, const std::vector<double>& c, const std::vector<double>& d) {
    int n = b.size();  // количество элементов в главной диагонали (n x n матрица)
    std::vector<double> v(n - 1, 0), u(n, 0);

    // Прямой ход
    if (b[0] == 0) {
        std::cerr << "Ошибка: деление на ноль в первой строке!" << std::endl;
        return {};  // Возвращаем пустой вектор в случае ошибки
    }

    v[0] = -c[0] / b[0]; // p = v
    u[0] = d[0] / b[0]; // q = u

    for (int i = 1; i < n; i++) {
        double denominator = b[i] + a[i - 1] * v[i - 1]; // a[i-1] для нижней диагонали
        if (denominator == 0) {
            std::cerr << "Ошибка: деление на ноль в строке " << i << std::endl;
            return {};  // Возвращаем пустой вектор в случае ошибки
        }
        if (i < n - 1) {
            v[i] = -c[i] / denominator;
        }
        u[i] = (d[i] - a[i - 1] * u[i - 1]) / denominator;
    }

    // Обратный ход
    std::vector<double> x(n, 0);
    x[n - 1] = u[n - 1];

    for (int i = n - 2; i >= 0; i--) {
        x[i] = v[i] * x[i + 1] + u[i];
    }

    return x;
}

bool is_solution_correct(const std::vector<std::vector<double>>& A, const std::vector<double>& b, const std::vector<double>& x) {
    int n = A.size();
    
    // Проверяем, что решение x подставляется в систему и дает правую часть b
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += A[i][j] * x[j];  // Вычисляем левую часть для i-го уравнения
        }
        
        // Сравниваем с правой частью с допустимой погрешностью
        std::cout << "Подстановка для уравнения " << i + 1 << ":\n";
        std::cout << "Левая часть (вычисленная): " << sum << "\n";
        std::cout << "Правая часть (ожидаемая): " << b[i] << "\n";
        std::cout << "Разница: " << std::fabs(sum - b[i]) << "\n\n";

        if (std::fabs(sum - b[i]) > 1e-6) {
            std::cerr << "Ошибка в решении! Уравнение " << i << " не выполнено. Получено: " << sum << ", ожидается: " << b[i] << std::endl;
            return false;  // Если хотя бы одно уравнение не выполнено, решение неверно
        }
    }

    return true;  // Если все уравнения выполнены
}

int main() {
    std::vector<std::vector<double>> A = {
        {-11, 9, 0, 0, 0},
        {-9, 17, 6, 0, 0},
        {0, 5, 20, 8, 0},
        {0, 0, -6, -20, 7},
        {0, 0, 0, 2, 8}
    };

    std::vector<double> b = {-117, -97, -6, 59, -86};

    // std::vector<std::vector<double>> A = {
    //     {8, 4, 0, 0, 0},
    //     {-5, 22, 8, 0, 0},
    //     {0, -5, -11, 1, 0},
    //     {0, 0, -9, -15, 1},
    //     {0, 0, 0, 1, 7}
    // };

    // std::vector<double> b = {48, 125, -43, 18, -23};

    // Проверяем, что матрица тридиагональна
    if (!is_tridiagonal(A)) {
        std::cerr << "Ошибка: матрица не тридиагональна! Такая матрица не поддерживается." << std::endl;
        return 1;  // Выход из программы с ошибкой
    }

    // Извлекаем диагонали
    int n = A.size();  // Размерность матрицы A
    std::vector<double> a(n - 1), b_diag(n), c(n - 1), d = b;

    // Заполняем диагонали
    for (int i = 0; i < n; i++) {
        b_diag[i] = A[i][i];  // Главная диагональ
        if (i < n - 1) {
            a[i] = A[i + 1][i]; // Нижняя диагональ
            c[i] = A[i][i + 1]; // Верхняя диагональ
        }
    }

    // Решаем систему уравнений
    std::vector<double> solution = tridiagonal_solve(a, b_diag, c, d);

    // Проверяем, были ли ошибки в решении
    if (solution.empty()) {
        std::cerr << "Ошибка решения системы! Проверьте матрицу на корректность." << std::endl;
        return 1;
    }

    // Выводим решение
    std::cout << "Решение уравнения методом прогонки:" << '\n';
    for (double x : solution) {
        std::cout << x << " ";
    }
    std::cout << '\n';

    // Проверка решения
    if (is_solution_correct(A, b, solution)) {
        std::cout << "Решение системы корректно." << std::endl;
    } else {
        std::cout << "Решение системы некорректно." << std::endl;
    }

    return 0;
}
