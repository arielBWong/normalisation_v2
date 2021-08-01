import json
import numpy as np
from surrogate_problems import WFG, DTLZs, MAF
import os


def problem_split():
    nv = 6
    no = 3
    max_eval = 400
    para_settings = dict()
    # check no parameter is wrongly written
    # maf is 1-6
    problem = []

    for i in np.arange(1, 7):
        problem = np.append(problem, 'MAF.MAF%d(n_var=%d, n_obj=%d)'%(i, nv, no))

    dltz_id = [1,2,3,4,7]
    for i in dltz_id:
        problem = np.append(problem, 'DTLZs.DTLZ%d(n_var=%d, n_obj=%d)'%(i, nv, no))
    # wfg 1-9
    for i in np.arange(1, 10):
        problem = np.append(problem, 'WFG.WFG_%d(n_var=%d, n_obj=%d, K=4)' % (i, nv, no))

    problem = problem.tolist()

    para_settings["MO_target_problems"] = problem
    para_settings["max_eval"] = max_eval
    para_settings["num_pop"] = 100
    para_settings["num_gen"] = 100


    methods =[
        'normalization_with_self_0',
        'normalization_with_nd_0',
        'normalization_with_nd_1',
        'normalization_with_nd_3',
        'normalization_with_nd_4',
        'normalization_with_nd_5',
    ]
    para_settings["method_selection"] = methods


    path = os.getcwd()
    settingfolder = os.path.join(path, 'run_settings')
    filename = os.path.join(settingfolder, 'run_settings_obj%d.json' % no)


    with open(filename, 'w') as wf:
        json.dump(para_settings, wf)


if __name__ =="__main__":
    problem_split()