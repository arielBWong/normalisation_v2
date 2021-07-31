import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg
from scipy import special
import copy

import surrogate_problems.corner_problems
import utility
import optimizer
from surrogate_problems import corner_problems

def gaussiancdf(x):
    y = 0.5 * (1 + special.erf(x / np.sqrt(2)))
    return y

def gausspdf(x):
    y = 1/np.sqrt(2*np.pi) * np.exp(-np.square(x)/2)
    return y


def krg_believer(x, krg, nd_front, ref):
    '''
    This function return x's evaluation results using kriging believer and hypervolume
    :param x: population of design variables to be evaluated with kriging
    :param krg(list):   kriging models for each objective
    :param nd_front:  current nd front for mo problems
    :param ref:  reference point for calculating hypervolume
    :return:
    '''


    org = nd_front.shape[0]
    nd_front_check = copy.deepcopy(nd_front)
    nd_front_check[nd_front_check <= 1.1] = 0
    nd_front_check = nd_front_check.sum(axis=1)
    deleterows = np.nonzero(nd_front_check)

    nd_front = np.delete(nd_front, deleterows, 0)
    if nd_front.shape[0] == 0:
        ndhv_value = 0
        nd_front = None
        print('external normalization caused ALL real ND front fall out of normalization boundary')
    else:
        if nd_front.shape[0] < org:
            print('external normalization caused SOME real ND front fall out of normalization boundary')
        hv_class = pg.hypervolume(nd_front)
        ndhv_value = hv_class.compute(ref)

    # --------------------
    x = np.atleast_2d(x)
    n_samples = x.shape[0]
    n_obj = len(krg)
    pred_obj = []
    for model in krg:
        y, _ = model.predict(x)
        pred_obj = np.append(pred_obj, y)

    pred_obj = np.atleast_2d(pred_obj).reshape(-1, n_obj,  order='F')

    fit = np.zeros((n_samples, 1))
    for i in range(n_samples):
        pred_instance = pred_obj[i, :]
        if np.any(pred_instance - ref >= 0):
            fit[i] = 0
        else:
            if nd_front is not None:
                hv_class = pg.hypervolume(np.vstack((nd_front, pred_instance)))
            else:
                pred_instance = np.atleast_2d(pred_instance)
                hv_class = pg.hypervolume(pred_instance)

            fit[i] = hv_class.compute(ref) - ndhv_value

    fit = np.atleast_2d(fit)
    return fit

def confirmsearch(nexty, trgy):
    obj_min = np.min(trgy, axis=0)
    diff = nexty - obj_min
    if np.any(diff < 0):
        return True
    else:
        return False


def corner_adjustment(corner_var, trgx, trgy, trgc, norm_scheme, target_problem, hv_ref, **kwargs):
    '''
    This function conduct the appointed corner search scheme
    :param corner_var: (int) specifying which scheme to use
    :param trgx: archive x
    :param trgy: archive y
    :param trgc: archive c
    :param norm_scheme: normalisation scheme, 'normalization with nd'
    :param target_problem: problem to be solved
    :param kwargs: additional variables for record silhouette analysis
    :return: expanded archive
    '''
    str_method = method_swtich(corner_var)
    corner_method = eval(str_method)
    trgx, trgy, silinfo1, silinfo2 = corner_method(trgx, trgy, trgc, norm_scheme, hv_ref, target_problem, **kwargs)
    # trgx, trgy, trgc, norm_scheme, hv_ref, target_problem,  **kwargs
    return trgx, trgy, silinfo1, silinfo2


def extreme_search(trgx, trgy, trgc, norm_scheme, hv_ref, target_problem,  **kwargs):
    trgy_norm = norm_scheme(trgy)
    krg, krgc = utility.model_building(trgx, trgy_norm, trgc)

    best_index = np.argmin(trgy, axis=0)
    guide_x = np.atleast_2d(trgx[best_index, :])

    # run estimated new best x on each objective
    x_out1 = check_krg_ideal_points(krg, target_problem.n_var, 0, target_problem.n_obj,
                                    target_problem.xl, target_problem.xu, guide_x, target_problem.name())

    trgx, trgy = additional_evaluation(x_out1, trgx, trgy, target_problem)

    return trgx, trgy, kwargs['sil_record'], kwargs['silcount']


def corner_search_all(trgx, trgy, trgc, norm_scheme, hv_ref, target_problem,  **kwargs):
    candx, candf, _, _ = corner_search_prep(trgx, trgy, trgc, norm_scheme, target_problem)
    if candx.shape[0] >= target_problem.n_obj:
        candx = candx[0: target_problem.n_obj, :]
        candf = candf[0:target_problem.n_obj, :]

    # evaluate all
    trgx,trgy = additional_evaluation(candx, trgx, trgy, target_problem)
    return trgx, trgy, kwargs['sil_record'], kwargs['silcount']


def corner_search_sel(trgx, trgy, trgc, norm_scheme, hv_ref, target_problem,  **kwargs):
    '''
    normalization scheme for corner search and selective evaluation
    :param trgx: archive x
    :param trgy: archive y
    :param trgc: archive c
    :param norm_scheme:  fixed scheme normalization with ND
    :param hv_ref: [1.1] * n_obj parameter for selective evaluation
    :param target_problem: true problem
    :param kwargs: variables for silhouette analysis
    :return: expanded archive x, f and silhouette records
    '''
    candx, candf, _, _ = corner_search_prep(trgx, trgy, trgc, norm_scheme, target_problem)
    if candx.shape[0] >= target_problem.n_obj:
        candx = candx[0: target_problem.n_obj, :]
        candf = candf[0:target_problem.n_obj, :]

    # selective evaluation
    trgy_norm = norm_scheme(trgy)
    before_size = trgy.shape[0]
    trgx, trgy = selective_evaluation(candx, candf, hv_ref, trgx, trgy, trgy_norm, target_problem)
    after_size = trgy.shape[0]
    print('corner search evaluation size for confirm %d' % int(after_size - before_size))

    return trgx, trgy, kwargs['sil_record'], kwargs['silcount']


def corner_search_selsil(trgx, trgy, trgc, norm_scheme, hv_ref, target_problem, **kwargs):
    '''
    corner search plus selective evaluation
    :param trgx: archive x
    :param trgy: archive f
    :param trgc: archive c
    :param norm_scheme: normalisation_with_nd fixed
    :param hv_ref: reference point
    :param target_problem: true problem
    :param kwargs: silhouette analysis parameters
    :return: expanded archive x f and silhouette analysis
    '''
    ndx, ndf, popx, popf = corner_search_prep(trgx, trgy, trgc, norm_scheme, target_problem)
    if ndx.shape[0] > target_problem.n_obj:  # clustering is only for top 2M
        x_out, f_out = utility.Silhouette(popf, popx, target_problem.n_obj)
        x_out = np.atleast_2d(x_out).reshape(-1, target_problem.n_var)
    else:
        x_out = ndx
        f_out = ndf

    if 'sil_record' not in kwargs.keys():
        raise('using function corner_search_selsil should pass in sil_record parameter')
    else:
        sil_record = kwargs['sil_record']
        silcount = kwargs['silcount']
        sil_record[str(silcount)] = x_out.shape[0]
        silcount = silcount + 1

    trgy_norm = norm_scheme(trgy)
    before_size = trgy.shape[0]
    trgx, trgy = selective_evaluation(x_out, f_out, hv_ref, trgx, trgy, trgy_norm, target_problem)
    after_size = trgy.shape[0]
    print('selective evaluation size %d' % int(after_size - before_size))

    return trgx, trgy, sil_record, silcount


def corner_search_prep(trgx, trgy, trgc, norm_scheme, target_probem):
    '''
    common process before corner search, use corner sorting to identify candidate solutions
    :param trgx: archive x
    :param trgy: archive f
    :param trgc: archive c
    :param norm_scheme: normalization scheme fixed to normalisation with ND
    :param target_probem: true problem to be solved
    :return:  nd front of corner sorting based search, and the last population
    '''
    trgy_norm = norm_scheme(trgy)
    krg, krgc = utility.model_building(trgx, trgy_norm, trgc)
    corner_problem = corner_problems.corner_problem(krg, n_var=target_probem.n_var, n_obj=target_probem.n_obj, upper_bound=target_probem.xu,
                                                    lower_bound=target_probem.xl, name=target_probem.name())
    ndx, _ = utility.get_ndfront(trgx, trgy_norm)
    opt_param = {'ranking_scheme': 'corner_sort'}
    last_popx, last_popf, _, _, _, _ = optimizer.optimizer(corner_problem, nobj=corner_problem.n_obj, ncon=0, mut=0.2,
                                                                     crossp=0.8, popsize=100, popgen=100, insertx=ndx, **opt_param)

    last_popf = utility.close_adjustment(last_popf)
    last_popndx, last_popndf = utility.get_ndfront(last_popx, last_popf)
    sorted_id = optimizer.corner_sort(last_popndf.shape[0], last_popndf.shape[1], 0, None, None, None, last_popndf)

    last_popndx = last_popndx[sorted_id, :]
    last_popndf = last_popndf[sorted_id, :]
    return last_popndx, last_popndf, last_popx, last_popf



def method_swtich(corner_var):
    if corner_var in [1, 3, 4, 5]:
        return {
            1: 'extreme_search',
            3: 'corner_search_all',
            4: 'corner_search_sel',
            5: 'corner_search_selsil'
        }[corner_var]
    else:
        raise('corner search option out of range, should be in 1, 3, 4, 5')


def selective_evaluation(candx, candf, hv_ref, trgx, trgy, trgf_norm, target_problem):
    '''
    For proposed corners, which ones should be evaluated
    only select those non-dominated and out of reference hypercube
    :param candx: corners x
    :param candf: corners f (surrogate prediction on normalised space)
    :param hv_ref: reference point
    :param trgx: archive x
    :param trgy: archive f
    :param trgf_norm: normalized archive f
    :param target_problem: true problem
    :return: expanded archive
    '''
    n = candx.shape[0]
    ndx, ndf_norm = utility.get_ndfront(trgx, trgf_norm)
    for i in range(n):
        x = candx[i, :]
        f = candf[i, :]
        if np.any(f > hv_ref):
            newf = np.vstack((ndf_norm, f))
            newx = np.vstack((ndx, x))
            _, new_ndf = utility.get_ndfront(newx, newf)
            tmp = np.abs(np.sum(new_ndf - f, axis=1))
            if np.any(tmp < 1e-5):   # usually do not use == 0
                trgx, trgy = additional_evaluation(x, trgx, trgy, target_problem)
            else:
                print('corner point dominated, ignore')

    return trgx, trgy


def additional_evaluation(x_krg, train_x, train_y, problem):
    '''
    this method only deal with unconstraint mo
    it does closeness
    :return: add kriging estimated x to training data.
    '''
    n_var = problem.n_var
    x_krg = np.atleast_2d(x_krg)
    n = x_krg.shape[0]

    for i in range(n):
        x_i = np.atleast_2d(x_krg[i]).reshape(-1, n_var)
        y_i = problem.evaluate(x_i, return_values_of=['F'])
        train_x = np.vstack((train_x, x_i))
        train_y = np.vstack((train_y, y_i))
    train_y = utility.close_adjustment(train_y)
    return train_x, train_y


def check_krg_ideal_points(krg, n_var, n_constr, n_obj, low, up, guide_x, problem_name):
    '''This function uses  krging model to search for a better x
    krg(list): krging model
    n_var(int): number of design variable for kriging
    n_constr(int): number of constraints
    n_obj(int): number of objective function
    low(list):
    up(list)
    guide_x(row vector): starting point to insert to initial population
    '''

    last_x_pop = []
    last_f_pop = []
    x_pop_size = 100
    x_pop_gen = 100

    # identify ideal x and f for each objective
    for k_i, k in enumerate(krg):
        problem = surrogate_problems.corner_problems.corner_problem(k, n_var=n_var, n_obj=1, upper_bound=up, lower_bound = low, name=problem_name)



        guide = np.atleast_2d(guide_x[k_i, :])
        pop_x, pop_f, _, _, _, _ = optimizer.optimizer(problem, problem.n_obj, problem.n_constr, 0.8, 0.2, 100, 100, insertx=guide, visual=False)

        # save the last population for lexicon sort
        last_x_pop = np.append(last_x_pop, pop_x)
        last_f_pop = np.append(last_f_pop, pop_f)  # var for test

    # long x
    last_x_pop = np.atleast_2d(last_x_pop).reshape(n_obj, -1)
    x_estimate = []
    # lex sort because
    # considering situation when f1 min has multiple same values
    # choose the one with smaller f2 value, so that nd can expand

    for i in range(n_obj):
        x_pop = last_x_pop[i, :]
        x_pop = x_pop.reshape(x_pop_size, -1)
        all_f = []
        # all_obj_f under current x pop
        for k in krg:
            f_k, _ = k.predict(x_pop)
            all_f = np.append(all_f, f_k)

        # reorganise all f in obj * popsize shape
        all_f = np.atleast_2d(all_f).reshape(n_obj, -1)
        # select an x according to lexsort
        x_index = lexsort_with_certain_row(all_f, i)
        # x_index = lexsort_specify_baserow(all_f, i)

        x_estimate = np.append(x_estimate, x_pop[x_index, :])

    x_estimate = np.atleast_2d(x_estimate).reshape(n_obj, -1)

    return x_estimate


def lexsort_with_certain_row(f_matrix, target_row_index):
    '''
    problematic function, given lexsort, it does not matter how upper
    rows are shuffled, sort is according to the last row.
    sort matrix according to certain row in fact last row
    e.g. sort the last row, the rest rows move its elements accordingly
    however, the matrix except last row is also sorted row wise
    according to number of min values each row has
    '''

    # f_matrix should have the size of [n_obj * popsize]
    # determine min
    target_row = f_matrix[target_row_index, :].copy()
    f_matrix = np.delete(f_matrix, target_row_index, axis=0)  # delete axis is opposite to normal

    f_min = np.min(f_matrix, axis=1)
    f_min = np.atleast_2d(f_min).reshape(-1, 1)
    # according to np.lexsort, put row with largest min values last row
    # -- this following sort is useless, can just go straight in 3d
    f_min_count = np.count_nonzero(f_matrix == f_min, axis=1)
    f_min_accending_index = np.argsort(f_min_count)
    # adjust last_f_pop
    last_f_pop = f_matrix[f_min_accending_index, :]

    # add saved target
    last_f_pop = np.vstack((last_f_pop, target_row))

    # apply np.lexsort (works row direction)
    lexsort_index = np.lexsort(last_f_pop)
    # print(last_f_pop[:, lexsort_index])
    selected_x_index = lexsort_index[0]

    return selected_x_index