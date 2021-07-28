import matplotlib.pyplot as plt
from pymop import ZDT1
from pymop.factory import get_uniform_weights


def get_problem_paretofront(problem):


    if problem.name() == 'DTLZ1' or problem.name() == 'DTLZ2' or problem.name() == 'DTLZ3' \
            or problem.name() == 'DTLZ4':
        ref_dir = get_uniform_weights(100, problem.n_obj)
        pf = problem.pareto_front(ref_dir)
    else:
        pf = problem.pareto_front(n_pareto_points=100)
    return pf


def process_visualisation2D(ax, problem, popf):
    plt.cla()
    pf = get_problem_paretofront(problem)

    ax.scatter(pf[:, 0], pf[:, 1], c='g', s=60, label='pareto front')
    ax.scatter(popf[:, 0], popf[:, 1], c='r', s=60, label='current pop')
    plt.legend()
    plt.pause(0.01)

def process_visualisation3D(ax, problem, popf):
    plt.cla()
    pf = get_problem_paretofront(problem)

    ax.scatter3D(pf[:, 0], pf[:, 1], pf[:, 2], c='g', s=60, label='pareto front')
    ax.scatter3D(popf[:, 0], popf[:, 1], popf[:, 2], c='r', s=60, label='current pop')
    plt.legend()
    plt.pause(0.01)



