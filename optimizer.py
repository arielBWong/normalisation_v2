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
from sklearn.utils.validation import check_array



def create_childrenMix(pop, dim, pop_size, cp, mp, eta):
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
    offspring = min(max(list(offspring), low), up)
    offspring = mutPolynomialBounded(offspring, eta, low, up, mp)

    return offspring


def create_childrenDE(pop, pop_f, dim, pop_size, cp, *args, **kwargs):
    '''
    Use DE operator to generate offspring, implementation is based on matrix operation from
    the DE (differential evolution) algorithm of Rainer Storn (matlab version, reference in reference folder)

    :param pop: population x, normalized into [0, 1], for less parameter passing
    :param pop_f: population objective (real objectives)
    :param dim:  x dimension
    :param pop_size: population size
    :param cp: crossover rate
    :param args: no mutation
    :param kwargs:
    :return: child population ui
    '''

    if pop_f.shape[1] > 1:
        strategy = 7  # multiple objective, no exponential crossover
        bm = None
    else:
        strategy = 6  # single objective /best/1 scheme, no exponenetial crossover
        bm_id = np.argmin(pop_f)
        bm = np.tile(pop[bm_id, :], (pop_size,1)) # best solution

    F = 0.8  # DE-stepsize F from interval [0, 2]
    # rotating index array (size NP)
    rot = np.arange(0, pop_size)
    # rotating index array (size D)
    rotd = np.arange(0, dim)  # (0:1:D-1);

    oldpop_x = pop.copy()  # no need deepcopy wasting of memory

    # index pointer array
    ind = np.random.permutation(4) + 1

    # shuffle locations of vectors
    a1 = np.random.permutation(pop_size)

    # rotate indices by ind(1) positions
    rt = np.remainder(rot + ind[0], pop_size)
    # rotate vector locations
    a2 = a1[rt]

    rt = np.remainder(rot + ind[1], pop_size)
    a3 = a2[rt]

    rt = np.remainder(rot + ind[2], pop_size)
    a4 = a3[rt]

    rt = np.remainder(rot + ind[3], pop_size)
    a5 = a4[rt]

    # shuffled population 1
    pm1 = oldpop_x[a1, :]
    pm2 = oldpop_x[a2, :]
    pm3 = oldpop_x[a3, :]
    pm4 = oldpop_x[a4, :]
    pm5 = oldpop_x[a5, :]

    # population filled with the best member of the last iteration
    mui = np.random.rand(pop_size, dim) < cp  # select from new
    if strategy > 5:
        st = strategy - 5
    else:
        # exponential crossover
        st = strategy
        # transpose, collect 1's in each column
        mui = np.sort(mui, axis=1)
        mui = mui.T
        for i in range(pop_size):
            n = np.floor(np.random.rand() * dim)
            if n > 0:
                # determine crossover decision
                # rotate
                rtd = np.remainder(rotd + n, dim)
                mui[:, i] = mui[rtd, i]
        mui = mui.T



    # inverse mask to mui
    # mpo = ~mui same as following one
    mpo = mui < 0.5  # select from old

    if st == 1:  # DE/best/1
        # differential variation
        ui = bm + F * (pm1 - pm2)  # permutate best member population
        # crossover
        ui = oldpop_x * mpo + ui * mui  # partially old population, partially new population

    if st == 2:  # DE/rand/1
        # differential variation
        ui = pm3 + F * (pm1 - pm2)
        # crossover
        ui = oldpop_x * mpo + ui * mui
    if st == 3:  # DE/rand-to-best/1
        ui = oldpop_x + F * (bm - oldpop_x) + F * (pm1 - pm2)
        ui = oldpop_x * mpo + ui * mui
    if st == 4:  # DE/best/2
        ui = bm + F * (pm1 - pm2 + pm3 - pm4)
        ui = oldpop_x * mpo + ui * mui
    if st == 5:  # DE/rand/2
        ui = pm5 + F * (pm1 - pm2 + pm3 - pm4)
        ui = oldpop_x * mpo + ui * mui

    # boundary fix, x is in normalised space
    ub_mask = ui <= 1.0
    lb_mask = ui >= 0.0
    ui = ui * ub_mask * lb_mask + 1.0 * (~ub_mask) + 0.0 * (~lb_mask) # bit operator, but applicable in logical array

    return ui






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
                try:
                    check_array(np.atleast_2d(delta_q))
                except ValueError as e:
                    print(e)
            else:
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                delta_q = 1.0 - val ** mut_pow
                try:
                    check_array(np.atleast_2d(delta_q))
                except ValueError as e:
                    print(e)


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



def corner_sort(popsize, nobj, ncon, infeasible, feasible, all_cv, all_f):
    # return ordered
    l1 = []
    order = []

    if ncon != 0:
        print('this sorting does not deal with constraints')
    else:
        # sort all f assume all f is of form  [f1, f2, f3, f1^2+f2^2, f1^2+f3^2, f1^2+f3^2]
        n = all_f.shape[0]
        a = np.linspace(0, n - 1, n, dtype=int)
        uniq_f, indx_unique = np.unique(all_f, return_index=True, axis=0)
        indx_same = np.setdiff1d(a, indx_unique)

        #
        for i in range(nobj):
            single_colid = np.argsort(uniq_f[:, i])
            l1 = np.append(l1, single_colid)

        # l1 is unique objectives' sorted ID
        l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

        # corner sort unique objective's ID
        # candidate_id is selecting from each objective
        # order is the list of selected ID
        i = 0
        while len(order) < uniq_f.shape[0]:
            # print(order)
            candidate_id = l1[0, i]

            order = np.append(order, candidate_id)
            # rearrange l1 remove candidate id
            l1 = l1.flatten(order='F')
            remove_id = np.where(l1 == candidate_id)
            l1 = np.delete(l1, remove_id)
            l1 = np.atleast_2d(l1).reshape(-1, nobj, order='F')

            # cycled pointer
            i = i + 1
            if i >= nobj:
                i = 0

    # convert back to original population ID
    selected = indx_unique[order.astype(int)]
    selected = np.append(selected, indx_same).flatten()
    selected = selected[0:popsize]
    selected = selected.astype(int)

    return selected


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


def optimizer(problem, nobj, ncon, mut, crossp, popsize, popgen, insertx=None, visual=False,  **kwargs):
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
        sorting = eval(kwargs['ranking_scheme'])
    else:
        sorting = eval('sort_population')

    if insertx is not None:
        new_popsize = popsize - insertx.shape[0]
    else:
        new_popsize = popsize

    dimensions = problem.n_var
    a = np.linspace(0, 2 * popsize - 1, 2 * popsize, dtype=int)
    pop = np.random.rand(new_popsize, dimensions)
    child_x = np.zeros((popsize, dimensions))
    min_b = problem.xl
    diff = np.fabs(problem.xu - problem.xl)
    pop_x = min_b + pop * diff

    if insertx is not None:
        insertx_pop = (insertx - problem.xl)/(problem.xu - problem.xl)
        pop = np.vstack((insertx_pop, pop))  # order matters w.r.t. pop_x
        pop_x = np.vstack((insertx, pop_x))


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
        # child_x = create_children(pop,  problem.n_var, popsize, crossp, mut, 30)
        child_x = create_childrenDE(pop, pop_f,  problem.n_var, popsize, crossp, mut)

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
    pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer(problem, 3, 0,  0.2, 0.8, 100, 100, visual=True)
