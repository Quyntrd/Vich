import numpy as np

# 1. Создать квадратную матрицу из случайных вещественных чисел (2, 4) размера 10.
matrix1 = np.random.uniform(2, 4, (10, 10))

# Найти скалярное произведение 4 строки на 5 столбец. Использовать срезы матриц.
row_4 = matrix1[3, :]  # 4-я строка (индекс 3, т.к. индексация с 0)
col_5 = matrix1[:, 4]  # 5-й столбец (индекс 4)

scalar_product_1 = np.dot(row_4, col_5)
print("Скалярное произведение 4 строки на 5 столбец:", scalar_product_1)


# 2. Создать две матрицы из случайных целых чисел [2, 7) подходящего размера.
matrix2 = np.random.randint(2, 7, (3, 3))
matrix3 = np.random.randint(2, 7, (3, 3))

# Скалярный алгоритм умножения матриц.
def scalar_multiplication(A, B):
    result = np.zeros((A.shape[0], B.shape[1]))
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += A[i, k] * B[k, j]
    return result

# Векторный алгоритм.
def vector_multiplication(A, B):
    return np.dot(A, B)

# Умножение через np.dot.
result_scalar = scalar_multiplication(matrix2, matrix3)
result_vector = vector_multiplication(matrix2, matrix3)
result_np_dot = np.dot(matrix2, matrix3)

print("\nПроизведение матриц (скалярный алгоритм):\n", result_scalar)
print("\nПроизведение матриц (векторный алгоритм):\n", result_vector)
print("\nПроизведение матриц (np.dot):\n", result_np_dot)


# 3. Создать вектор-строку 1x10 из случайных целых чисел.
vector = np.random.randint(1, 10, 10)

# Вычислить норму ‖x‖2 самостоятельно написанной функцией.
def norm_2(vector):
    return np.sqrt(np.sum(vector**2))

norm_2_custom = norm_2(vector)
norm_2_np = np.linalg.norm(vector)

print("\nНорма вектора ‖x‖2 (собственная реализация):", norm_2_custom)
print("Норма вектора ‖x‖2 (np.linalg.norm):", norm_2_np)


# 4. Создать матрицу из случайных целых чисел.
matrix4 = np.random.randint(1, 10, (5, 5))

# Найти норму матрицы ‖A‖∞ (максимальная сумма по строкам).
def norm_inf(matrix):
    return np.max(np.sum(np.abs(matrix), axis=1))

norm_inf_custom = norm_inf(matrix4)
norm_inf_np = np.linalg.norm(matrix4, ord=np.inf)

print("\nНорма матрицы ‖A‖∞ (собственная реализация):", norm_inf_custom)
print("Норма матрицы ‖A‖∞ (np.linalg.norm):", norm_inf_np)


# 5. Найти отражение Хаусхолдера для вектора, которое обнуляет его координаты с 6 по 10.
def householder_reflection(vector):
    v = vector[5:]  # часть вектора с 6 по 10 элемент
    e = np.zeros_like(v)
    e[0] = np.linalg.norm(v)
    u = v - e
    u = u / np.linalg.norm(u)  # нормализуем вектор

    H = np.eye(len(v)) - 2 * np.outer(u, u)
    v_new = H @ v
    vector[5:] = v_new
    return vector

vector_reflected = householder_reflection(vector.copy())
print("\nОтражение Хаусхолдера для вектора:\n", vector_reflected)


# 6. LU-разложение самостоятельно написанной функцией.
def lu_decomposition(matrix):
    n = matrix.shape[0]
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = matrix[i, j] - L[i, :i] @ U[:i, j]
        L[i, i] = 1
        for j in range(i + 1, n):
            L[j, i] = (matrix[j, i] - L[j, :i] @ U[:i, i]) / U[i, i]

    return L, U

L, U = lu_decomposition(matrix4)
LU_multiplied = np.dot(L, U)

print("\nLU-разложение:")
print("L:\n", L)
print("U:\n", U)
print("Проверка LU-разложения (L*U):\n", LU_multiplied)


# 7. QR-разложение всеми методами и проверка.
# Метод Грама-Шмидта
def qr_gram_schmidt(matrix):
    n = matrix.shape[0]
    Q = np.zeros_like(matrix, dtype=float)
    R = np.zeros((n, n), dtype=float)

    for j in range(n):
        v = matrix[:, j]
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], matrix[:, j])
            v = v - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

Q_gs, R_gs = qr_gram_schmidt(matrix4)
QR_multiplied_gs = np.dot(Q_gs, R_gs)

print("\nQR-разложение методом Грама-Шмидта:")
print("Q:\n", Q_gs)
print("R:\n", R_gs)
print("Проверка QR-разложения (Q*R):\n", QR_multiplied_gs)

# QR-разложение через np.linalg.qr
Q_np, R_np = np.linalg.qr(matrix4)
QR_multiplied_np = np.dot(Q_np, R_np)

print("\nQR-разложение (np.linalg.qr):")
print("Q:\n", Q_np)
print("R:\n", R_np)
print("Проверка QR-разложения (Q*R):\n", QR_multiplied_np)
