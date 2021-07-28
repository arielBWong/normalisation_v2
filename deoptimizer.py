import numpy as np
from sklearn.utils.validation import check_array

def optimizer_DE(problem, ncon, insertpop, NP, itermax, visflag, **kwargs):
    #  NP: number of population members/popsize
    #  itermax: number of generation
    #  kwargs for this method is for plot, keys are train_y
    ax = None
    dimensions = problem.n_var
    # Check input variables
    VTR = -np.inf
    refresh = 0
    F = 0.8
    CR = 0.8
    strategy = 6
    use_vectorize = 1

    if NP < 5:
        NP = 5
        print('pop size is increased to minimize size 5')

    if CR < 0 or CR > 1:
        CR = 0.5
        print('CR should be from interval [0,1]; set to default value 0.5')

    if itermax <= 0:
        itermax = 200
        print('generation size is set to default 200')

    # insert guide population
    if insertpop is not None:
        check_array(insertpop)
        n_insertpop = len(insertpop)
        n_rest = NP - n_insertpop
        if n_insertpop > NP:  # rear situation where nd is larger than evolution population size
            n_rest = 1
            insertpop = insertpop[0:NP-1, :]
    else:
        n_rest = NP

    # Initialize population and some arrays
    # if pop is a matrix of size NPxD. It will be initialized with random
    # values between the min and max values of the parameters

    min_b = problem.xl
    max_b = problem.xu
    pop = np.random.rand(n_rest, dimensions)
    pop_x = min_b + pop * (max_b - min_b)
    if insertpop is not None:  # attach guide population
        pop_x = np.vstack((pop_x, insertpop))

    XVmin = np.repeat(np.atleast_2d(min_b), NP, axis=0)
    XVmax = np.repeat(np.atleast_2d(max_b), NP, axis=0)

    if ncon != 0:
        pop_f, pop_g = problem.evaluate(pop_x, return_values_of=["F", "G"], **kwargs)
        tmp = pop_g.copy()
        tmp[tmp <= 0] = 0
        pop_cv = tmp.sum(axis=1)

    if ncon == 0:
        # np.savetxt('test_x.csv', pop_x, delimiter=',')
        pop_f = problem.evaluate(pop_x, return_values_of=["F"], **kwargs)

    # best member of current iteration
    bestval = np.min(pop_f)  # single objective only
    ibest = np.where(pop_f == bestval)  # what if multiple best values?
    bestmemit = pop_x[ibest[0][0]]  # np.where return tuple of (row_list, col_list)

    # save best_x ever
    bestmem = bestmemit

    # DE-Minimization
    # popold is the population which has to compete. It is static through one
    # iteration. pop is the newly emerging population
    # initialize bestmember  matrix
    bm = np.zeros((NP, dimensions))

    # intermediate population of perturbed vectors
    ui = np.zeros((NP, dimensions))

    # rotating index array (size NP)
    rot = np.arange(0, NP)

    # rotating index array (size D)
    rotd = np.arange(0, dimensions)  # (0:1:D-1);

    iter = 1
    while iter < itermax and bestval > VTR:



        # save the old population
        # print('iteration: %d' % iter)
        oldpop_x = pop_x.copy()

        # index pointer array
        ind = np.random.permutation(4) + 1

        # shuffle locations of vectors
        a1 = np.random.permutation(NP)

        # rotate indices by ind(1) positions
        rt = np.remainder(rot + ind[0], NP)
        # rotate vector locations
        a2 = a1[rt]

        rt = np.remainder(rot + ind[1], NP)
        a3 = a2[rt]

        rt = np.remainder(rot + ind[2], NP)
        a4 = a3[rt]

        rt = np.remainder(rot + ind[3], NP)
        a5 = a4[rt]
        # for test
        # a5 = np.loadtxt('a5.csv', delimiter=',')
        # a5 = np.array(list(map(int, a5)))-1

        # shuffled population 1
        pm1 = oldpop_x[a1, :]
        pm2 = oldpop_x[a2, :]
        pm3 = oldpop_x[a3, :]
        pm4 = oldpop_x[a4, :]
        pm5 = oldpop_x[a5, :]

        # population filled with the best member of the last iteration
        # print(bestmemit)
        for i in range(NP):
            bm[i, :] = bestmemit

        mui = np.random.rand(NP, dimensions) < CR
        if strategy > 5:
            st = strategy - 5
        else:
            # exponential crossover
            st = strategy
            # transpose, collect 1's in each column
            # did not implement following strategy process

        # inverse mask to mui
        # mpo = ~mui same as following one
        mpo = mui < 0.5

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
        if st == 5:  #DE/rand/2
            ui = pm5 + F * (pm1 - pm2 + pm3 - pm4)
            ui = oldpop_x * mpo + ui * mui


        # correcting violations on the lower bounds of the variables
        # validate components
        maskLB = ui > XVmin
        maskUB = ui < XVmax

        # part one: valid points are saved, part two/three beyond bounds are set as bounds
        ui = ui * maskLB * maskUB + XVmin * (~maskLB) + XVmax * (~maskUB)

        # Select which vectors are allowed to enter the new population
        if use_vectorize == 1:

            if ncon != 0:
                pop_f_temp, pop_g_temp = problem.evaluate(ui, return_values_of=["F", "G"], **kwargs)
                tmp = pop_g_temp.copy()
                tmp[tmp <= 0] = 0
                pop_cv_temp = tmp.sum(axis=1)

            if ncon == 0:
                # np.savetxt('test_x.csv', pop_x, delimiter=',')
                pop_f_temp = problem.evaluate(ui, return_values_of=["F"], **kwargs)

            # if competitor is better than value in "cost array"
            indx = pop_f_temp <= pop_f
            # replace old vector with new one (for new iteration)
            change = np.where(indx)
            pop_x[change[0], :] = ui[change[0], :]
            pop_f[change[0], :] = pop_f_temp[change[0], :]

            # we update bestval only in case of success to save time
            indx = pop_f_temp < bestval
            if np.sum(indx) != 0:
                # best member of current iteration
                bestval = np.min(pop_f_temp)  # single objective only
                ibest = np.where(pop_f_temp == bestval)  # what if multiple best values?
                if len(ibest[0]) > 1:
                    print(
                        "multiple best values, selected first"
                    )
                bestmem = ui[ibest[0][0], :]
            # freeze the best member of this iteration for the coming
            # iteration. This is needed for some of the strategies.
            bestmemit = bestmem.copy()

            if visflag and (problem.name in 'all but one'):
                ax.cla()
                ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.3)
                ax.scatter3D(pop_x[:, 0], pop_x[:, 1], pop_f[:, 0], marker=7, c='black')


        if refresh == 1:
            print('Iteration: %d,  Best: %.4f,  F: %.4f,  CR: %.4f,  NP: %d' % (iter, bestval, F, CR, NP))

        iter = iter + 1
        del oldpop_x

    return np.atleast_2d(bestmem), np.atleast_2d(bestval), pop_x, pop_f


