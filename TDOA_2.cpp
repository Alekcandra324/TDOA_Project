#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

// Функция для вычисления Евклидова расстояния
double evclid(Vector2d point1, Vector2d point2) {
    return (point1 - point2).norm();
}

// Функция ошибки
double loss_function(Vector2d A, Vector2d D, Vector2d E, Vector2d F, const vector<double>& distance_diff) {
    double mistake = 0.0;
    mistake += pow(evclid(A, D) - evclid(A, E) - distance_diff[0], 2);
    mistake += pow(evclid(A, D) - evclid(A, F) - distance_diff[1], 2);
    mistake += pow(evclid(A, E) - evclid(A, F) - distance_diff[2], 2);
    return mistake;
}

// Градиентный спуск
void Gradient_spusk(Vector2d& A, Vector2d D, Vector2d E, Vector2d F, const vector<double>& distance_diff, int iterations) {
    double step_sizes[2] = { 0.001,0.001 };
    double alpha_increase = 1.1;  // Во сколько раз увеличивать шаг, если ошибка уменьшилась
    double alpha_decrease = 0.5;  // Во сколько раз уменьшать шаг, если ошибка увеличилась
    for (int iter = 0; iter < iterations; iter++) {
        double prev_loss = loss_function(A, D, E, F, distance_diff);
        Vector2d A_old = A;
        for (int coord = 0; coord < 2; coord++) {
            double grad = 0.0;

            double normA_D = (A - D).norm();
            double normA_E = (A - E).norm();
            double normA_F = (A - F).norm();
            grad += 2 * (normA_D - normA_E - distance_diff[0]) * ((A - D)[coord] / normA_D - (A - E)[coord] / normA_E);
            grad += 2 * (normA_D - normA_F - distance_diff[1]) * ((A - D)[coord] / normA_D - (A - F)[coord] / normA_F);
            grad += 2 * (normA_E - normA_F - distance_diff[2]) * ((A - E)[coord] / normA_E - (A - F)[coord] / normA_F);


            A[coord] -= step_sizes[coord] * grad;
        }
        double new_loss = loss_function(A, D, E, F, distance_diff);  // Новая ошибка после изменения A

        if (new_loss < prev_loss) {
            // Ошибка уменьшилась -> увеличиваем шаг
            step_sizes[0] *= alpha_increase;
            step_sizes[1] *= alpha_increase;
        }
        else {
            // Ошибка увеличилась -> уменьшаем шаг и откатываем A
            step_sizes[0] *= alpha_decrease;
            step_sizes[1] *= alpha_decrease;
            A = A_old;  // Возвращаемся к предыдущему значению
        }
    }
}

int main() {
    // Известные координаты источников
    Vector2d D(5, 10), E(8, 8), F(10, 19);

    // Истинное положение A (для вычисления разностей)
    Vector2d A_v(4, 5);

    // Разности расстояний
    vector<double> distance_diff;
    distance_diff.push_back((A_v - D).norm() - (A_v - E).norm());
    distance_diff.push_back((A_v - D).norm() - (A_v - F).norm());
    distance_diff.push_back((A_v - E).norm() - (A_v - F).norm());
    int idontknow = 2;
    // Начальная догадка
    Vector2d A(2, 2);

    // Запускаем градиентный спуск
    Gradient_spusk(A, D, E, F, distance_diff, 500);

    // Вывод результата
    cout << "A: (" << A.x() << ", " << A.y() << ")\n";
    cout << loss_function(A, D, E, F, distance_diff) << "\n";

    return 0;
}
