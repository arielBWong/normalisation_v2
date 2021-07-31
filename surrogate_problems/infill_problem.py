import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt

from surrogate_problems.sur_problem_base import Problem
from surrogate_assisted import krg_believer
from pymop import ZDT1
from visualisation_collection import get_problem_paretofront
from utility import init_solutions, model_building, update_archive, get_ndfront, normalization_with_nd
from optimizer import optimizer
from deoptimizer import optimizer_DE


class hv_infill(Problem):
    def __init__(self, n_var, n_obj, n_constr, upper_bound, lower_bound, name):
        self.n_var = n_var
        self.n_constr = n_constr
        self.n_obj = n_obj
        self.xl = anp.array(lower_bound)
        self.xu = anp.array(upper_bound)
        self.name = name + '_hv_infill'
        self.krg = None
        self.nd_front = None
        self.ref = None
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        if self.krg is None:
            raise('set up kriging model before evaluation, using method: evaluation_prepare')
        if self.nd_front is None:
            raise('set up ND front solutions before evaluation, using method: evaluation_prepare')
        if self.ref is None:
            raise('set up reference point before evaluation, using method: evaluation_prepare')

        fit = krg_believer(x, self.krg, self.nd_front, self.ref)
        out["F"] = -fit



    def evaluation_prepare(self, krg, nd_front, ref):
        self.krg = krg
        self.nd_front = nd_front
        self.ref = ref


def infill_visulisation(ax, problem, infillf, archivef):
    plt.cla()
    pf = get_problem_paretofront(problem)
    ax.scatter(pf[:, 0], pf[:, 1], c='g', s=60, label='pareto front')
    ax.scatter(archivef[:, 0], archivef[:, 1], c='orange', s=30, label='archive')
    ax.scatter(infillf[:, 0], infillf[:, 1], c='r', s=60, label='infill')
    plt.pause(0.01)


def infill_test():

    print('test hv infill')
    fig, ax = plt.subplots()
    problem = ZDT1(n_var=2)
    ref = [1.1] * problem.n_obj
    # (1) init x y
    trgx, trgy, trgc = init_solutions(21, problem)
    trgy_norm = normalization_with_nd(trgy)

    # (2) train krg for init, krg operate on 0-1 space
    krg, krgc = model_building(trgx, trgy_norm, trgc)

    # (3) propose next x
    infill_problem = hv_infill(n_var=problem.n_var, n_obj=1, n_constr=0, upper_bound=problem.xu, lower_bound=problem.xl,
                               name=problem.name())

    for i in range(10):
        _, ndf_norm = get_ndfront(trgx, trgy_norm)
        infill_problem.evaluation_prepare(krg, ndf_norm, ref)

        # pop_x, pop_f, pop_g, archive_x, archive_f, archive_g = optimizer(infill_problem, nobj=1, ncon=0, mut=0.2,
        #                                                                   crossp=0.8, popsize=100, popgen=100, insertx=None)
        # nextx = np.atleast_2d(pop_x[0, :])

        nextx, _, _, _ = optimizer_DE(infill_problem, ncon=0, insertpop=None, NP=100, itermax=100, visflag=False)

        if problem.n_constr != 0:
            nexty, nextc = problem.evaluate(nextx, return_values_of=['F', 'G'])
        else:
            nexty = problem.evaluate(nextx, return_values_of=['F'])
            nextc = None

        # infill_visulisation(ax, problem, nexty, trgy)

        trgx, trgy, trgc = update_archive(trgx, trgy, trgc, nextx, nexty, nextc)
        trgy_norm = normalization_with_nd(trgy)
        krg, krgc = model_building(trgx, trgy_norm, trgc)

if __name__=="__main__":
    import cProfile

    cProfile.run('infill_test()')
    # infill_test()





