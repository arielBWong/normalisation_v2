import numpy as np
from surrogate_problems import infill_problem, DTLZs
from utility import model_building, init_solutions, get_ndfront, update_archive, normalization_with_nd, denormalization_with_nd
import surrogate_assisted
import optimizer
import os
from joblib import dump, load
import multiprocessing as mp

def normalisation_adjustment(seed, str_problem, max_eval, str_normscheme, verbose=True):
    '''
    This is the main steps of normalization
    :param seed:
    :param str_problem:  problem to be solved
    :param max_eval: max number of function evaluation
    :param str_normscheme:  which normalization scheme to use
    :param corner_search: whether to apply corner search
    :return:
    '''
    # decode method
    items = str_normscheme.split('_')
    str_normscheme ='_'.join(items[0:-1])
    corner_search = int(items[-1])

    # setting up
    np.random.seed(seed)
    target_problem = eval(str_problem)
    print('Problem %s, seed %d, normalization method %s switch %d' % (target_problem.name(), seed, str_normscheme, corner_search))
    hv_ref = [1.1] * target_problem.n_obj
    hvinfill_problem = infill_problem.hv_infill(n_var=target_problem.n_var, n_obj=1, n_constr=0, upper_bound=target_problem.xu, lower_bound=target_problem.xl,
                               name=target_problem.name())
    num_init = int(max_eval/2)

    trgx, trgy, trgc = init_solutions(num_init, target_problem)
    norm_scheme = eval(str_normscheme)
    denorm_scheme = eval('de'+str_normscheme)

    # record for silhouette and selective evaluation
    activation_count = 0
    silcount = 0
    activation_record = dict()
    sil_record = dict()

    # if corner search start with a corner search hoping to have adjustment first
    if corner_search:
        before_archivesize = trgx.shape[0]
        sil_param = {'sil_record': sil_record, 'silcount': silcount}
        trgx, trgy, sil_record, silcount = surrogate_assisted.corner_adjustment(corner_search, trgx, trgy, trgc, norm_scheme, target_problem, hv_ref, **sil_param)
        after_archivesize = trgx.shape[0]
        activation_record = activation_saveprocess(before_archivesize, after_archivesize, activation_record,
                                                   activation_count)
        activation_count = activation_count + 1

    # head into infill
    while trgy.shape[0] < max_eval:
        if verbose:
            print('Evaluation: ' + str(trgy.shape[0]))
        # create surrogate model & set up evaluation
        trgy_norm = norm_scheme(trgy)
        krg, krgc = model_building(trgx, trgy_norm, trgc)
        _, ndy_norm = get_ndfront(trgx, trgy_norm)
        hvinfill_problem.evaluation_prepare(krg, ndy_norm, hv_ref)

        pop_x, _, _, _, _, _ = optimizer.optimizer(hvinfill_problem, nobj=1, ncon=0, mut=0.2,
                                                                         crossp=0.8, popsize=100, popgen=100,
                                                                         insertx=None)
        nextx = np.atleast_2d(pop_x[0, :])

        if target_problem.n_constr != 0:
            nexty, nextc = target_problem.evaluate(nextx, return_values_of=['F', 'G'])
        else:
            nexty = target_problem.evaluate(nextx, return_values_of=['F'])
            nextc = None
        trgx, trgy, trgc = update_archive(trgx, trgy, trgc, nextx, nexty, nextc)

        if trgy.shape[0] >= max_eval:
            trgx = trgx[0:max_eval, :]
            trgy = trgy[0:max_eval, :]
            break

        if corner_search & surrogate_assisted.confirmsearch(nexty, trgy[0:-1, :]):
            before_archivesize = trgx.shape[0]
            sil_param = {'sil_record': sil_record, 'silcount': silcount}
            trgx, trgy, sil_record, silcount = surrogate_assisted.corner_adjustment(corner_search, trgx, trgy, trgc,
                                                                                    norm_scheme, target_problem, hv_ref,
                                                                                    **sil_param)
            after_archivesize = trgx.shape[0]
            activation_record = activation_saveprocess(before_archivesize, after_archivesize, activation_record,
                                                       activation_count)
            activation_count = activation_count + 1

    # save results
    str_method = '%s_%s' % (str_normscheme, str(corner_search))
    save_results(seed, trgy, target_problem, str_method, activation_record, sil_record)



def make_dir(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def save_results(seed, trgy, problem, str_method, activation_record, sil_record):
    # upper folder
    f = 'result_folder_OBJ%d' % int(problem.n_obj)
    folder1 = os.path.join(os.getcwd(), f)
    make_dir(folder1)

    # second level folder
    f = problem.name() + "_" + str_method
    folder2 = os.path.join(folder1, f)
    make_dir(folder2)

    # save f and activation

    savename = os.path.join(folder2, 'trainy_seed_%d.csv' % int(seed))
    np.savetxt(savename, trgy, delimiter=',')

    # activation
    filename = os.path.join(folder2, 'activationcheck_seed_%d.joblib' % seed)
    dump(activation_record, filename)

    filename = os.path.join(folder2, 'sil_record_seed_%d.joblib' % seed)
    dump(sil_record, filename)


def activation_saveprocess(before_archivesize, after_archivesize, activation_record, activation_count):
    newsolutions = np.arange(before_archivesize, after_archivesize)
    activation_record[str(activation_count)] = newsolutions
    return activation_record

def run_experiments():
    import json
    folder  = os.path.join(os.getcwd(), 'run_settings')
    settings = os.path.join(folder, 'run_settings_obj3.json')

    args = []
    seedmax = 29
    for problem_setting in settings:
        with open(problem_setting, 'r') as data_file:
            hyp = json.load(data_file)
        target_problems = hyp['MO_target_problems']
        method_selection = hyp['method_selection']
        max_eval = hyp['max_eval']
        num_pop = hyp['num_pop']
        num_gen = hyp['num_gen']
        for problem in target_problems:
            for seed in range(seedmax):
                for method in method_selection:
                    args.append((seed, problem, max_eval, method))
    num_workers = 48
    pool = mp.Pool(processes=num_workers)
    pool.starmap(normalisation_adjustment, ([arg for arg in args]))

    # normalisation_adjustment(1, "DTLZs.DTLZ1(n_var=6, n_obj=3)", 60, "normalization_with_nd", 4)



if __name__=="__main__":
    # print(0)
    # import cProfile

    normalisation_adjustment(1, "DTLZs.DTLZ1(n_var=6, n_obj=3)", 60, "normalization_with_self_0")
    # cProfile.run('run_experiments()')

    # seed, str_problem, max_eval, str_normscheme, corner_search