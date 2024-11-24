import math

def f1(x, y):
    return x - math.tan(x * y + 0.1) / x

def f2(x, y):
    return y - (1 - x * x) / (2 * y)

def f1dx(x, y):
    return 1 - (2 * x * y - math.sin(2 * x * y + 0.2)) / (2 * x * x * math.cos(x * y + 0.1) ** 2)

def f1dy(x, y):
    return -1 / (math.cos(x * y + 0.1) ** 2)

def f2dx(x, y):
    return x / y

def f2dy(x, y):
    return 1 - (x * x - 1) / (2 * y * y)

def gaus(matrix, n):
    ans = [0] * n
    for k in range(n):
        q = matrix[k][k]
        for i in range(k, n + 1):
            matrix[k][i] /= q
        for i in range(k + 1, n):
            q = matrix[i][k]
            for j in range(k, n + 1):
                matrix[i][j] -= matrix[k][j] * q

    ans[n - 1] = matrix[n - 1][n]
    for i in range(n - 2, -1, -1):
        s = sum(matrix[i][j] * ans[j] for j in range(i + 1, n))
        ans[i] = matrix[i][n] - s

    return ans

def newton(x, y, it_cnt):
    for i in range(it_cnt):
        matrix = [
            [f1dx(x, y), f1dy(x, y), -f1(x, y)],
            [f2dx(x, y), f2dy(x, y), -f2(x, y)]
        ]

        ans = gaus(matrix, 2)
        x -= ans[0]
        y -= ans[1]

        print(f"Iteration {i + 1}: x = {x:.6f}, y = {y:.6f}")

    print(f"Final result: x = {x:.6f}, y = {y:.6f}")

def main():
    print("This program is solving the system of non-linear equations:")
    print("tan(xy + 0.1) = x^2, x^2 + 2y^2 = 1 by using the Newton method.")

    x = float(input("Please enter your start approximation for x: "))
    y = float(input("Please enter your start approximation for y: "))
    it_cnt = int(input("Now please enter the number of iterations you want to perform: "))

    newton(x, y, it_cnt)

if __name__ == "__main__":
    main()
