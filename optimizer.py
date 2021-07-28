import numpy as np
import time
import pygmo as pg
from collections import Sequence
from itertools import repeat
from surrogate_problems import branin
from pymop import ZDT1, ZDT2, ZDT3,  DTLZ1, DTLZ2
import matplotlib.pyplot as plt
from visualisation_collection import process_visualisation2D, process_visualisation3D
from copy import deepcopy



def create_children(pop, dim, pop_size, cp, mp, eta):
    '''
    This reproduce function use DE operator and polynormial mutation operator
    :param pop: normlaised x population
    :param dim:  x dimensions
    :param pop_size: how many child to generated
    :param cp: crossover rate
    :param mp:  mutation rate
    :param eta:  polynormial mutation index
    :return:  child population of size pop_size
    '''
    # (1) prepare parents index list for DE parents
    parents_id = np.zeros((pop_size, 3))
    parents_id[:, 0] = np.arange(pop_size)
    tmp_child = []

    for i in range(pop_size):
        remaining_ids = np.setdiff1d(np.arange(pop_size), np.array(parents_id[i, 0]))

        cand_id = np.random.permutation(pop_size - 1)
        parents_id[i, 1] = remaining_ids[cand_id[0]]
        parents_id[i, 2] = remaining_ids[cand_id[1]]


    # (2) use parent list to generate child one by one
    i = 0
    while len(tmp_child) < pop_size * dim:  # this is a vector
        off_spring = de_operator(parents_id[i, :], pop, cp, mp, eta, dim)

        # make sure offspring is unique w.r.t parents & children population
        tmpx = np.append(tmp_child, off_spring)
        tmpx_2d = np.atleast_2d(tmpx).reshape(-1, dim)

        # only add to new child population when the new child is unique
        unique_arr = np.unique(tmpx_2d, axis=1)
        if unique_arr.shape[0] == tmpx_2d.shape[0]:
            tmp_child = np.append(tmp_child, off_spring)
            i = i + 1

    return np.atleast_2d(tmp_child).reshape(-1, dim)


def de_operator(parents_id, popx, cp, mp, eta, dim):
    '''
    :param parents_id:  row rector, indicating parents id in x population
    :param popx:  current x population
    :param cp:  crossover rate
    :param mp:  mutation rate
    :param eta:  polynormial mutation index
    :param dim: x dimenstion
    :return: one child x solution
    '''
    crossover_site = np.random.rand(1, dim)
    crossover_site = (crossover_site < cp)

    # deepcopy is important or else error will generate
    offspring = deepcopy(np.atleast_2d(popx[int(parents_id[0]), :]))
    p2 = deepcopy(np.atleast_2d(popx[int(parents_id[1]), :]))
    p3 = deepcopy(np.atleast_2d(popx[int(parents_id[2]), :]))

    # DE crossover
    offspring[crossover_site] = offspring[crossover_site] + cp * (p2[crossover_site] - p3[crossover_site])

    # polynomial mutation
    low = [0] * dim
    up = [1] * dim
    offspring = offspring.flatten()
    offspring = mutPolynomialBounded(offspring, eta, low, up, mp)

    return offspring


def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a :term:`python:sequence` of values that
                is the lower bound of the search space.
    :param up: A value or a :term:`python:sequence` of values that
               is the upper bound of the search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(range(size), low, up):

        if np.random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - xl) / (xu - xl)
            delta_2 = (xu - x) / (xu - xl)
            rand = np.random.random()
            mut_pow = 1.0 / (eta + 1.)

            if rand < 0.5:
                xy = 1.0 - delta_1
                val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                delta_q = val ** mut_pow - 1.0
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow

            # boundary check
            x = x + delta_q * (xu - xl)
            x = min(max(x, xl), xu)
            individual[i] = x

        else:
            # boundary check
            x = individual[i]
            x = min(max(x, xl), xu)
            individual[i] = x

    return individual


def sort_population(popsize,nobj,ncon,infeasible, feasible, all_cv,all_f):
    l2 =[]
    l1=[]
    sl=[]
    ff=[]
    if ncon!=0:
        infeasible=np.asarray(infeasible)
        infeasible=infeasible.flatten()
        index1 = all_cv[infeasible].argsort()
        index1=index1.tolist()
        l2=infeasible[index1]
    if len(feasible)>=1:
        ff = all_f[feasible, :]
        if nobj==1:
            ff=ff.flatten()
            index1 = ff.argsort()
            index1=index1.tolist()
            l1=feasible[index1]
        if nobj>1:
            sl = pg.sort_population_mo(ff)
            l1 = feasible[sl]
    order=np.append(l1, l2, axis=0)
    order=order.flatten()
    selected=order[0:popsize]
    selected=selected.flatten()
    selected=selected.astype(int)
    return selected


def optimizer(problem, nobj, ncon, bounds, mut, crossp, popsize, popgen, visual,  **kwargs):
    '''
    This is a general EA optimizer
    :param problem (class instance): problem to be optimized, defined with pymop problem class definition
    :param nobj (int): number of objective
    :param ncon (int): number of constraints
    :param bounds (list): x bounds
    :param mut (float): mutation probability
    :param crossp (float): crossover probability
    :param popsize (int): population size
    :param popgen (int): generation size
    :param visual (binary): whether to plot process
    :param kwargs:
    :return:
    '''

    # initialization and settings
    if visual is True:
        if nobj < 3:
            print('2d plot')
            plothandle = eval('process_visualisation2D')
            fig, ax = plt.subplots()
        elif nobj == 3:
            print('3d plot')
            plothandle = eval('process_visualisation3D')
            ax = plt.axes(projection='3d')
        else:
            print('higher than 3D, no plot')
    if 'ranking_scheme' in kwargs.keys():
        sorting = eval('ranking_scheme')
    else:
        sorting = eval('sort_population')


    dimensions = len(bounds)
    a = np.linspace(0, 2 * popsize - 1, 2 * popsize, dtype=int)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_x = min_b + pop * diff
    archive_x = pop

    # initial population
    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)

    if ncon == 0:
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)
        pop_g = []

    # archive initialization
    archive_g = pop_g
    archive_f = pop_f

    # evolution
    for i in range(popgen):
        if visual is True:
            plothandle(ax, problem, pop_f)

        # generate children
        child_x = create_children(pop, dimensions, popsize, crossp, mut, 30)

        # Evaluating the offspring
        trial_denorm = min_b + child_x * diff
        if ncon != 0:
            child_f, child_g = problem.evaluate(trial_denorm, return_values_of=["F", "G"], **kwargs)
        if ncon == 0:
            child_f = problem.evaluate(trial_denorm, return_values_of=["F"], **kwargs)

        # Parents and offspring 2N size population for sorting
        all_x = np.append(pop, child_x, axis=0)
        all_f = np.append(pop_f, child_f, axis=0)

        if ncon != 0:
            all_g = np.append(pop_g, child_g, axis=0)
            all_g[all_g <= 0] = 0
            all_cv = all_g.sum(axis=1)
            infeasible = np.nonzero(all_cv)
            feasible = np.setdiff1d(a, infeasible)
        if ncon == 0:
            feasible = a
            infeasible = []
            all_cv = []

        feasible = np.asarray(feasible)
        feasible = feasible.flatten()

        # Selecting the parents for the next generation
        selected = sorting(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f)

        pop = all_x[selected, :]
        pop_f = all_f[selected, :]
        if ncon != 0:
            pop_g = all_g[selected, :]

        # Storing all solutions in tha archive
        archive_x = np.append(archive_x, child_x, axis=0)
        archive_f = np.append(archive_f, child_f)
        if ncon != 0:
            archive_g = np.append(archive_g, child_g)

    # Getting the variables in appropriate bounds
    pop_x = min_b + pop * diff
    archive_x = min_b + archive_x * diff
    return pop_x, pop_f, pop_g, archive_x, archive_f, archive_g


if __name__ == "__main__":
    np.seterr(over='raise')
    np.random.seed(1)
    problem = DTLZ1(n_var=7, n_obj=3)
    bounds = np.vstack((problem.xl, problem.xu)).T.tolist()
    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer(problem, 3, 0, bounds, 0.2, 0.8, 100, 100, visual=True)
