import os
import json
import subprocess as sp
import numpy as np

def get_ast_ids(ast_dirpath):
    find = ["find", ast_dirpath, "-regex", ".*/[0-9]+.json"]
    proc = sp.run(find, stdout=sp.PIPE) 
    filelist = proc.stdout.decode("utf-8").split("\n")
    filelist = [fn for fn in filelist if len(fn) > 0]
    ids = [int(os.path.splitext(os.path.basename(fn))[0]) for fn in filelist]
    return ids

def parse_json(json_file):
    """ Deserializes the json string in a file and returns the object.

    Refer to the tranaslation table in the Python documentation for json
    library.
    """
    with open(json_file) as fo:
        json_string = fo.read()
    parsed_json = json.loads(json_string)
    return parsed_json

def dfs_traversal(json_tree):
    type_list = []
    id_list = []
    root_node = json_tree
    dfs_stack = [root_node]
    while len(dfs_stack) > 0:
        json_node = dfs_stack.pop()
        if json_node in ["return", "call"]:
            type_list.append(json_node)
            continue

        node_type = json_node["type"]
        node_id = json_node["id"]
        type_list.append(node_type)
        id_list.append(node_id)
        if "children" not in json_node:
            continue
        
        dfs_stack.append("return")
        for child_node in reversed(json_node["children"]):
            dfs_stack.append(child_node)
        dfs_stack.append("call")
    type_list.append("end")
    return type_list

def get_aid_to_blocks(ast_dirpath):
    all_types = set()
    ast_ids = get_ast_ids(ast_dirpath)
    aid_to_blocks = {}
    # collect block lists and block names
    for aid in ast_ids:
        ast_filepath = os.path.join(ast_dirpath, str(aid) + ".json")
        json_tree = parse_json(ast_filepath)
        type_list = dfs_traversal(json_tree)
        all_types = all_types.union(set(type_list))

        aid_to_blocks[aid] = type_list
    # convert block name to id
    block_name_to_id = {b: i for i, b in enumerate(all_types)}
    id_to_blockname = {i: b for b, i in block_name_to_id.items()}
    # HACK: swap "end" with whatever block which is assigned to the last index
    last_block = id_to_blockname[len(id_to_blockname) - 1]
    block_name_to_id[last_block] = block_name_to_id["end"]
    block_name_to_id["end"] = len(id_to_blockname) - 1

    maxlen = 0
    for aid in ast_ids:
        aid_to_blocks[aid] = [block_name_to_id[b] for b in aid_to_blocks[aid]]
        maxlen = max(maxlen, len(aid_to_blocks[aid]))
    return aid_to_blocks, maxlen

def get_aid_to_nextblocks(aid_to_blocks):
    aid_to_nextblocks = {}
    for aid in aid_to_blocks:
        aid_to_nextblocks[aid] = \
            aid_to_blocks[aid][1:] + [aid_to_blocks[aid][-1]]
    return aid_to_nextblocks

def map_to_matrix(map_to_lists, maxlen):
    matrix = []
    for key in sorted(map_to_lists.keys()): 
        real_part = map_to_lists[key] 
        if len(real_part) == 0:
            continue
        padded_part = [real_part[-1]] * (maxlen - len(real_part))
        matrix.append(real_part + padded_part)
    return np.array(matrix)


if __name__=="__main__":
    rootpath = "../anonymizeddata/data/hoc18/"
    ast_dirpath = os.path.join(rootpath, "asts")
    
    aid_to_blocks, maxlen = get_aid_to_blocks(ast_dirpath)
    aid_to_nextblocks = get_aid_to_nextblocks(aid_to_blocks)

    block_matrix = map_to_matrix(aid_to_blocks, maxlen)
    nextblock_matrix = map_to_matrix(aid_to_nextblocks, maxlen)

    from keras.utils import to_categorical
    #block_matrix = to_categorical(block_matrix)
    #block_matrix = np.swapaxes(block_matrix, 0, 2)
    #block_matrix = np.swapaxes(block_matrix, 0, 1)
    np.save("hoc18_ast_block_matrix.npy", block_matrix)

