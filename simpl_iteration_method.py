import numpy as np

x_0 = np.array([0, 0])
epsilon = 1e-6


def F(x):
    x1, x2 = x[0], x[1]
    return np.array([
        np.cos(x1) - 4 * x2 + 3,
        x1 + np.sin(x2)
    ])

def compute_C(x):
    x1, x2 = x[0], x[1]
    return np.array([
        [-0.1 + 0.01 * np.cos(x1), 0],
        [0, -0.1 + 0.01 * np.sin(x2)]
    ])


def simple_iteration_method(x_0, epsilon):
    x_i = x_0
    iteration = 0

    while True:
        C = compute_C(x_i)

        F_x_i = F(x_i)
        x_next = x_i + np.dot(C, F_x_i)

        condition_number = np.max(np.abs(x_next - x_i))
        comparison_sign = ">=" if condition_number > epsilon else "<="

        print(f"Iteration {iteration + 1}:")
        print(f"x_{iteration + 1} = [{x_next[0]:.8f}, {x_next[1]:.8f}]")
        print(f"||x_{iteration + 1} - x_{iteration}|| = {condition_number:.8f}")
        print(f"Check: {condition_number:.8f} {comparison_sign} {epsilon:.8f}")
        print("------------------------")

        if condition_number < epsilon:
            return x_next

        x_i = x_next
        iteration += 1

solution = simple_iteration_method(x_0, epsilon)
print("\nResponse:")
print(f"x1 = {solution[0]:.8f}")
print(f"x2 = {solution[1]:.8f}")
