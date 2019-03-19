import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess as sp

def get_trajectory_ids(trajectory_dirpath):
    find = ["find", trajectory_dirpath, "-regex", ".*/[0-9]+.txt"]
    proc = sp.run(find, stdout=sp.PIPE) 
    filelist = proc.stdout.decode("utf-8").split("\n")
    filelist = [fn for fn in filelist if len(fn) > 0]
    ids = [int(os.path.splitext(os.path.basename(fn))[0]) for fn in filelist]
    return ids

def get_trajectory_to_asts(trajectory_dirpath):
    traj_to_asts_map = {}
    tid_list = get_trajectory_ids(trajectory_dirpath)
    maxlen = 0
    for tid in tid_list:
        trajectory_filepath = os.path.join(trajectory_dirpath, str(tid)+".txt")
        with open(trajectory_filepath) as fo:
            ast_list = [int(line.rstrip()) for line in fo]
            maxlen = max(len(ast_list), maxlen)
            traj_to_asts_map[tid] = ast_list
    return traj_to_asts_map, maxlen


def current_failures_to_future(failure_list, window=1):
    end = len(failure_list)
    return [1 if \
        sum(failure_list[ind:min(ind+window, end)]) \
        == min(ind+window,end) - ind else 0 \
        for ind in range(end)]


def get_trajectory_to_failures(traj_to_asts, correct_set):
    traj_to_failures_map = {}
    for tid in traj_to_asts:
        traj_to_failures_map[tid] = \
            [1 if (aid not in correct_set) else 0 for aid in traj_to_asts[tid]] 

        # Check if succeeds, it does not submit another failed submission
        if 0 in traj_to_failures_map[tid]:
            assert(traj_to_failures_map[tid][-1] == 0)
    return traj_to_failures_map

def get_trajectory_to_future_failures(traj_to_failures, window=1):
    traj_to_future_map = {}
    for tid in traj_to_failures:
        flist = traj_to_failures[tid]
        traj_to_future_map[tid] = current_failures_to_future(flist, window)
    return traj_to_future_map

def get_correct_asts(result_filepath):
    correct_set = set()
    with open(result_filepath) as fo:
        fo.readline()
        for line in fo:
            row = line.rstrip().split()
            ast_id = int(row[0])
            score = int(row[1])
            if score == 100:
                correct_set.add(ast_id)
    return correct_set

def map_to_matrix(map_to_lists, maxlen):
    matrix = []
    for key in sorted(map_to_lists.keys()): 
        real_part = map_to_lists[key] 
        if len(real_part) == 0:
            continue
        padded_part = [real_part[-1]] * (maxlen - len(real_part))
        matrix.append(real_part + padded_part)
    return np.array(matrix)

def plot_data_balance(Y):
    fractions = np.mean(Y, axis=0)
    plt.plot(fractions)
    plt.xlabel("Timestep")
    plt.ylabel("Fraction of failure labels")

def report_data_balance(Y, imagepath):
    print("Fraction of failure labels", np.mean(Y))
    plot_data_balance(Y)
    plt.savefig(imagepath)

if __name__=="__main__":
    rootpath = "../anonymizeddata/data/hoc4/"
    trajectory_dirpath = os.path.join(rootpath, "trajectories")
    result_filepath = os.path.join(rootpath, "asts/unitTestResults.txt")
    
    correct_set = get_correct_asts(result_filepath)
    traj_to_asts, maxlen = get_trajectory_to_asts(trajectory_dirpath)
    traj_to_fails = get_trajectory_to_failures(traj_to_asts, correct_set) 
    traj_to_futures = get_trajectory_to_future_failures(traj_to_fails, 2)

    ast_matrix = map_to_matrix(traj_to_asts, maxlen) + 1
    fail_matrix = map_to_matrix(traj_to_fails, maxlen)
    future_matrix = map_to_matrix(traj_to_futures, maxlen)
    assert(np.shape(ast_matrix) == np.shape(fail_matrix)) 

    report_data_balance(fail_matrix[:,np.arange(10)], "fail_balance.png")
    report_data_balance(future_matrix[:,np.arange(10)], "fail_win_2_balance.png")
    np.save("traj_ast_matrix.npy", ast_matrix)
    np.save("traj_fail_matrix.npy", fail_matrix)
    np.save("traj_fail_window_2_matrix.npy", future_matrix)
