import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.validation import check_array

from surrogate_problems.sur_problem_base import Problem
from surrogate_assisted import krg_believer
from pymop import ZDT1
from visualisation_collection import get_problem_paretofront
from utility import init_solutions, model_building, update_archive, get_ndfront, normalization_with_nd
from optimizer import optimizer



class corner_problem(Problem):
    def __init__(self, krg, n_var, n_obj, upper_bound, lower_bound, name):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = anp.array(lower_bound)
        self.xu = anp.array(upper_bound)
        self.name = name + '_corner_problem'
        self.krg = krg
        self.n_constr = 0
        super().__init__(n_var=self.n_var,
                         n_obj=self.n_obj,
                         n_constr=self.n_constr,
                         xl=self.xl,
                         xu=self.xu,
                         type_var=anp.double)


    def _evaluate(self, x, out, *args, **kwargs):
        check_array(x)
        f = []
        if self.n_obj > 1:
            for i in range(self.n_obj):
                y, _ = self.krg[i].predict(x)
                f = np.append(f, y)
        else:
            f = self.krg.predict(x)
        out['F'] = np.atleast_2d(f).reshape(-1, self.n_obj, order='F')

