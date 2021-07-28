import numpy as np
from scipy.stats import norm
from sklearn.utils.validation import check_array
import pygmo as pg
from scipy import special
import copy

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

