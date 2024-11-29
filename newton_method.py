import numpy as np

# Часткові похідні для матриці Якобі
def jacobian(x1, x2):
    df1_dx1 = 1 - np.cos((x1 + x2) / 3)
    df1_dx2 = -np.cos((x1 + x2) / 3)
    df2_dx1 = (5 * np.sin((x1 - x2) / 4)) / 4
    df2_dx2 = 1 - (5 * np.sin((x1 - x2) / 4)) / 4
    return np.array([[df1_dx1, df1_dx2], [df2_dx1, df2_dx2]])

# Функції системи рівнянь
def f1(x1, x2):
    return x1 - 3 * np.sin((x1 + x2) / 3)

def f2(x1, x2):
    return x2 - 5 * np.cos((x1 - x2) / 4)

# Метод Ньютона
def newton_method(x0, epsilon):
    x = np.array(x0, dtype=float)
    iteration = 0
    print("Розпис ітерацій:")
    print("-" * 90)

    while True:
        # Вектор функцій
        F = np.array([f1(x[0], x[1]), f2(x[0], x[1])])
        # Матриця Якобі
        J = jacobian(x[0], x[1])
        
        # Перевірка на виродженість
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-10:
            raise ValueError(f"Вироджена матриця Якобі на ітерації {iteration}. Спробуйте змінити початкове наближення.")
        
        # Розв'язок системи лінійних рівнянь J * z = -F
        z = np.linalg.solve(J, -F)
        # Оновлення наближення
        new_x = x + z
        iteration += 1

        # Вивід ітерації у стилі документа
        print(f"Ітерація №{iteration}:")
        print(f"x = ({x[0]:.6f}, {x[1]:.6f})")
        print(f"F(x) = ({F[0]:.6f}, {F[1]:.6f})")
        print("Матриця Якобі:")
        print(f"[{J[0, 0]:.6f}, {J[0, 1]:.6f}]")
        print(f"[{J[1, 0]:.6f}, {J[1, 1]:.6f}]")
        print(f"Поправки: z = ({z[0]:.6f}, {z[1]:.6f})")
        print(f"Нове наближення: x = ({new_x[0]:.6f}, {new_x[1]:.6f})")
        print("-" * 90)
        
        # Перевірка умови зупинки
        if np.linalg.norm(z, ord=np.inf) <= epsilon:
            x = new_x
            break
        
        x = new_x

    return x, iteration

# Ввід користувача
epsilon = float(input("Введіть точність (наприклад, 1e-5): "))
x0 = list(map(float, input("Введіть початкове наближення у форматі x1 x2: ").split()))

# Запуск методу
try:
    solution, iterations = newton_method(x0, epsilon)
    # Результат
    print("Завершення:")
    print(f"Розв'язок: x1 = {solution[0]:.6f}, x2 = {solution[1]:.6f}")
    print(f"Кількість ітерацій: {iterations}")
except ValueError as e:
    print(f"Помилка: {e}")
