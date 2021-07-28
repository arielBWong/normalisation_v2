import numpy as np
import pyDOE
from krige_dace import krige_dace
import pygmo as pg
from sklearn.utils.validation import check_array

def init_solutions(number_of_initial_samples, target_problem):
    '''
    Initialise certain number of solutions
    :param number_of_initial_samples:  (int)
    :param target_problem: (pymop style problems)
    :return: desgin vectors x and its objective and constraint
    '''
    n_vals = target_problem.n_var
    n_sur_cons = target_problem.n_constr

    # initial samples with hyper cube sampling
    train_x = pyDOE.lhs(n_vals, number_of_initial_samples, criterion='maximin')

    xu = np.atleast_2d(target_problem.xu).reshape(1, -1)
    xl = np.atleast_2d(target_problem.xl).reshape(1, -1)

    train_x = xl + (xu - xl) * train_x

    if n_sur_cons != 0:
        train_y, cons_y = target_problem.evaluate(train_x, return_values_of=['F', 'G'])
    else:
        train_y = target_problem.evaluate(train_x, return_values_of='F')
        cons_y = None

    return train_x, train_y, cons_y


def model_building(train_x, train_y, cons_y):
    n_sur_objs = train_y.shape[1]
    if cons_y is not None:
        n_sur_cons = cons_y.shape[1]
    else:
        n_sur_cons = 0

    gpr = []
    for i in range(n_sur_objs):
        one_obj_y = np.atleast_2d(train_y[:, i]).reshape(-1, 1)
        gpr.append(create_krg(train_x, one_obj_y))

    gpr_g = []
    for i in range(n_sur_cons):
        one_cons_g = np.atleast_2d(cons_y[:, i]).reshape(-1, 1)
        gpr_g.append(create_krg(train_x, one_cons_g))

    return gpr, gpr_g


def create_krg(x, y):
    mykriging = krige_dace(x, y)
    mykriging.train()
    return mykriging


def update_archive(x, y, c, newx, newy, newc):
    if c is not None:
        c = np.vstack((c, newc))
    x = np.vstack((x, newx))
    y = np.vstack((y, newy))
    return x, y, c

def get_ndfront(x, y):
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_index = ndf[0]
    nd_y = y[ndf_index, :]
    nd_x = x[ndf_index, :]
    return nd_x, nd_y



def normalization_with_nd(y):
    y = check_array(y)
    ndf, dl, dc, ndr = pg.fast_non_dominated_sorting(y)
    ndf = list(ndf)
    ndf_size = len(ndf)
    # extract nd for normalization
    # nd has to be more than 1 points for normalisation purpose
    if len(ndf[0]) > 1:
        ndf_extend = ndf[0]
    else:
        ndf_extend = np.append(ndf[0], ndf[1])

    nd_front = y[ndf_extend, :]

    # normalization boundary
    min_nd_by_feature = np.amin(nd_front, axis=0)
    max_nd_by_feature = np.amax(nd_front, axis=0)

    # rule out exception nadir and ideal are too close
    # add more fronts to nd front
    if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
        print('nd front aligned problem, cannot form proper normalisation bounds, expanding nd front')
        ndf_index = ndf[0]
        for k in np.arange(1, ndf_size):
            ndf_index = np.append(ndf_index, ndf[k])
            nd_front = y[ndf_index, :]
            min_nd_by_feature = np.amin(nd_front, axis=0)
            max_nd_by_feature = np.amax(nd_front, axis=0)
            if np.any(max_nd_by_feature - min_nd_by_feature < 1e-5):
                continue
            else:
                break
    norm_y = (y - min_nd_by_feature) / (max_nd_by_feature - min_nd_by_feature)
    return norm_y