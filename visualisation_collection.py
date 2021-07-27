import matplotlib.pyplot as plt
from pymop import ZDT1


def get_problem_paretofront(problem):
    pf = problem.pareto_front(n_pareto_points=100)
    return pf


def process_visualisation2D(ax, problem, popf):
    plt.cla()
    pf = get_problem_paretofront(problem)

    ax.scatter(pf[:, 0], pf[:, 1], c='g', s=60, label='pareto front')
    ax.scatter(popf[:, 0], popf[:, 1], c='r', s=60, label='current pop')
    plt.legend()
    plt.pause(1)



def process_visualisation3D(ax, problem, popf):
    print('not implemented')
