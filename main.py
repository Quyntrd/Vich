import numpy as np
import pandas as pd

# Коэффициенты системы
A = np.array([
    [2.7, 3.3, 1.3],
    [3.5, -1.7, 2.8],
    [4.1, 5.8, -1.7]
])

b = np.array([2.1, 1.7, 0.8])

# Функция для проверки достаточного условия сходимости (норма матрицы B < 1)
def check_convergence(A):
    # Преобразуем систему к виду x = Bx + c
    B = np.zeros_like(A)
    n = len(b)
    for i in range(n):
        B[i, :] = -A[i, :] / A[i, i]
        B[i, i] = 0  # Убираем диагональные элементы для матрицы B
    
    # Проверяем норму матрицы B
    norm_B = np.linalg.norm(B, ord=np.inf)
    if norm_B < 1:
        return True
    return False

# Функция для перестановки строк с целью улучшения диагональной доминантности
def rearrange_system(A, b):
    n = len(b)
    for i in range(n):
        # Находим строку с максимальным по модулю диагональным элементом
        max_row = max(range(i, n), key=lambda k: abs(A[k, i]))
        # Меняем строки, если нужно
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
    return A, b

# Преобразование системы для метода простых итераций
def simple_iteration_method(A, b, tol=1e-3, max_iterations=100):
    x = np.zeros_like(b)
    n = len(b)
    
    # Преобразуем систему к виду x = Bx + c
    B = np.zeros_like(A)
    c = np.zeros_like(b)
    for i in range(n):
        B[i, :] = -A[i, :] / A[i, i]
        B[i, i] = 0  # Убираем диагональные элементы
        c[i] = b[i] / A[i, i]
    
    iteration_data = []
    for k in range(max_iterations):
        x_new = np.dot(B, x) + c
        iteration_data.append(list(x_new))
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    
    return x, iteration_data

# Проверка условия сходимости
if check_convergence(A):
    print("Условие сходимости выполнено. Начинаем итерации...")
    
    # Решение методом простых итераций
    solution, iterations = simple_iteration_method(A, b)

    # Формируем таблицу с итерациями
    df = pd.DataFrame(iterations, columns=['x1', 'x2', 'x3'])
    print("\nТаблица итераций:")
    print(df)
    
    # Проверка решения с помощью np.linalg.solve
    exact_solution = np.linalg.solve(A, b)
    print("\nТочное решение методом np.linalg.solve:", exact_solution)

    # Проверка точности
    print("\nРешение методом простых итераций:", solution)
    print("\nПогрешность:", np.linalg.norm(exact_solution - solution, ord=np.inf))

else:
    print("Условие сходимости не выполнено. Преобразуем систему...")

    # Преобразуем систему для улучшения сходимости
    A_new, b_new = rearrange_system(A, b)

    if check_convergence(A_new):
        print("После преобразования условие сходимости выполнено. Начинаем итерации...")
        
        # Решение методом простых итераций после преобразования
        solution, iterations = simple_iteration_method(A_new, b_new)

        # Формируем таблицу с итерациями
        df = pd.DataFrame(iterations, columns=['x1', 'x2', 'x3'])
        print("\nТаблица итераций после преобразования:")
        print(df)
        
        # Проверка решения с помощью np.linalg.solve
        exact_solution = np.linalg.solve(A, b)
        print("\nТочное решение методом np.linalg.solve:", exact_solution)

        # Проверка точности
        print("\nРешение методом простых итераций после преобразования:", solution)
        print("\nПогрешность:", np.linalg.norm(exact_solution - solution, ord=np.inf))
    else:
        print("Даже после преобразования система не сходится.")
