import matplotlib.pyplot as plt

OPTIMAL_X = 0
KAI = 25
SQRT_KAI = 5

def func1(x):
    return 25 * x ** 2

def func1_grad(x):
    return 50 * x

def func2(x):
    return x ** 2 + 48 * x - 24

def func2_grad(x):
    return 2 * x + 48

def func3(x):
    return 25 * x ** 2 - 48 * x + 72

def func3_grad(x):
    return 50 * x - 48

def get_func_val(x):
    if x < 1:
        return func1(x)
    elif x <= 2:
        return func2(x)
    else:
        return func3(x)

def get_func_grad(x):
    if x < 1:
        return func1_grad(x)
    elif x <= 2:
        return func2_grad(x)
    else:
        return func3_grad(x)

def gradient_descent(x, learning_rate, num_iterations):
    sub_optimality_gaps = []
    optimality_distances = []
    for i in range(num_iterations):
        sub_optimality_gaps.append(abs(get_func_val(x) - get_func_val(OPTIMAL_X)))
        optimality_distances.append(abs(x - OPTIMAL_X))

        x = x - learning_rate * get_func_grad(x)
    return sub_optimality_gaps, optimality_distances

def heavy_ball(x, learning_rate, num_iterations, momentum):
    v = 0
    sub_optimality_gaps = []
    optimality_distances = []
    for i in range(num_iterations):
        sub_optimality_gaps.append(abs(get_func_val(x) - get_func_val(OPTIMAL_X)))
        optimality_distances.append(abs(x - OPTIMAL_X))

        v = -learning_rate * get_func_grad(x) + momentum * v
        x = x + v
    return sub_optimality_gaps, optimality_distances

def nestrov_accelerated_gradient(x, learning_rate, num_iterations, momentum):
    y = x
    sub_optimality_gaps = []
    optimality_distances = []
    for i in range(num_iterations):
        sub_optimality_gaps.append(abs(get_func_val(x) - get_func_val(OPTIMAL_X)))
        optimality_distances.append(abs(x - OPTIMAL_X))

        x_ = y - learning_rate * get_func_grad(y)
        y = x_ - momentum * (x_ - x)
        x = x_
    return sub_optimality_gaps, optimality_distances

def get_gradient_descent_upper_bound(x, num_iterations):
    optimality_distances = [abs(x - OPTIMAL_X)]
    for i in range(1, num_iterations):
        optimality_distances.append(optimality_distances[-1] * (KAI - 1) / (KAI + 1))
    return optimality_distances

def get_heavy_ball_upper_bound(x, num_iterations):
    optimality_distances = [abs(x - OPTIMAL_X)]
    for i in range(1, num_iterations):
        optimality_distances.append(optimality_distances[-1] * (SQRT_KAI - 1) / (SQRT_KAI + 1))
    return optimality_distances


def plot_results(all_sub_optimality_gaps, all_optimality_distances, title):
    line_specs = {
        'Gradient Descent': 'r-',
        'Heavy Ball': 'g-',
        'Nestrov Accelerated Gradient': 'b-',
        'Gradient Descent Upper Bound': 'r--',
        'Heavy Ball Upper Bound': 'g--',
    }
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Optimality Gaps and Distances')

    axs[0].set_title('Sub-optimality Gaps $f(x_t) - f(x^*)$')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Sub-optimality Gap $f(x_t) - f(x^*)$')

    axs[1].set_title('Optimality Distances $||x_t - x^*||$')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Optimality Distance $||x_t - x^*||$')

    for i in range(len(title)):
        if 'Bound' not in title[i]:
            axs[0].plot(all_sub_optimality_gaps[i], line_specs[title[i]], label=title[i])
        axs[1].plot(all_optimality_distances[i], line_specs[title[i]], label=title[i])

    axs[0].legend()
    axs[1].legend()
    plt.savefig('optimality_gaps_and_distances.png')
    plt.show()

def main():
    x = 3
    num_iterations = 50
    parameters = [
        (gradient_descent, x, 1/50, num_iterations),
        (heavy_ball, x, 1/18, num_iterations, 4/9),
        (nestrov_accelerated_gradient, x, 1/50, num_iterations, 2/3)
    ]

    results = [arg[0](*arg[1:]) for arg in parameters]
    titles = ['Gradient Descent', 'Heavy Ball', 'Nestrov Accelerated Gradient', 'Gradient Descent Upper Bound', 'Heavy Ball Upper Bound']

    all_sub_optimality_gaps, all_optimality_distances = zip(*results)
    all_optimality_distances += (get_gradient_descent_upper_bound(x, num_iterations), get_heavy_ball_upper_bound(x, num_iterations))
    plot_results(all_sub_optimality_gaps, all_optimality_distances, titles)



if __name__ == '__main__':
    main()
