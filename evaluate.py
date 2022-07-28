

from sup_metrics     import evaluate
from unsup_statistics   import call

from i24_database_api.db_writer import DBWriter
import numpy as np
import os
import time
import torch
import _pickle as pickle

def db_cleanup(dbw,coll_name):
    dbw.delete_collection([coll_name])

    inp = input("Do you want to remove all RAW_TRACKS collections? Type YES if so:")
    if inp == "YES":
        existing_collections = [col["name"] for col in list(dbw.db.list_collections())] # list all collections
        remove = []
        for item in existing_collections:
            if "RAW*" in item:
                remove.append(item)
        inp2 = input("{} collections to be removed ({}). Continue? Type YES if so:".format(len(remove),remove))
        if inp2 == "YES":
            if len(remove) > 0:
                dbw.delete_collection(remove)

adj_list = ["admissible",
            "ostentatious",
            "modest",
            "loquacious",
            "gregarious",
            "cantankerous",
            "bionic",
            "demure",
            "thrifty",
            "quizzical",
            "pragmatic",
            "sibilant",
            "visionary",
            "morose",
            "jubilant",
            "apathetic",
            "stalwart",
            "paradoxical",
            "tantalizing",
            "specious",
            "tautological",
            "hollistic",
            "super",
            "pristine",
            "wobbly",
            "lovely"]

noun_list = ["anteater",
             "zebra",
             "anaconda",
             "aardvark",
             "bison",
             "wallaby",
             "heron",
             "stork",
             "cyborg",
             "vulcan",
             "snek",
             "beluga",
             "panda",
             "lynx",
             "panther",
             "housecat",
             "osprey",
             "bovine",
             "jackalope",
             "yeti",
             "doggo",
             "cheetah",
             "squirrel",
             "axylotl",
             "kangaroo"
             ]


def get_pickle(name):
    with open(name,"rb") as f:
        result = pickle.load(f)
        return result

db_param = {
      "default_host": "10.2.218.56",
      "default_port": 27017,
      "host":"10.2.218.56",
      "port":27017,
      "username":"i24-data",
      "password":"mongodb@i24",
      "default_username": "i24-data",
      "readonly_user":"i24-data",
      "default_password": "mongodb@i24",
      "db_name": "trajectories",      
      "server_id": 1,
      "session_config_id": 1,
      "trajectory_database":"trajectories",
      "timestamp_database":"transformed"
      }

metric_info_dict = {
    "precision":{"best":1,"bad":0.5},
    "recall":{"best":1,"bad":0.5},
    }

if __name__ == "__main__":
    gt_coll = "groundtruth_scene_1"
    IOUT = 0.3
    collection_cleanup = False

    ### overwrite collection_name 
<<<<<<< Updated upstream
    coll_name = "morose_panda--RAW_GT1_boggles"

=======
    coll_name = "morose_panda--RAW_GT1_cajoles" ; append_db = False
    #coll_name = "morose_panda--RAW_GT1_castigates" ; append_db = False
    #coll_name = "morose_panda--RAW_GT1_boggles" ; append_db = False
    coll_name = "morose_panda--RAW_GT1"            ; append_db = True


    dbw   = DBWriter(db_param,collection_name = db_param["db_name"])
    existing_collections = dbw.db.list_collection_names() # list all collections
    
    [print(item) for item in existing_collections]
        
        
>>>>>>> Stashed changes
    # generate comment
    comment = input("Description of run settings / test for storage with evaluation results: ")    

        

    dbw   = DBWriter(db_param,collection_name = db_param["db_name"])
    existing_collections = dbw.db.list_collection_names() # list all collections
    
    [print(item) for item in existing_collections]
    

    result = {}
    result["name"] = coll_name
    result["description"] = comment
    result["bps"]  = 1
    result["iou_threshold"] = IOUT
    
    start = time.time()
    ### supervised evaluation
    if gt_coll is not None:
        metrics = evaluate(db_param,gt_collection = gt_coll,pred_collection = coll_name, sample_freq = 30, break_sample = 2700,append_db = append_db,iou_threshold = IOUT)
        result["gt"] = gt_coll
        #result["metrics"] = metrics
        
        for key in metrics.keys():
            result[key] = metrics[key]
   
    
    ### unsupervised statistics
    statistics = call(db_param,coll_name)
    #result["statistics"] = statistics
    elapsed = time.time() - start
    result["eval_time"] = elapsed
    
    # unroll statistics and metrics so that result is flat
    for key in statistics.keys():
        stat = statistics[key]
        
        if type(stat) == dict:
            for subkey in stat.keys():
                kn = "{}_{}".format(key,subkey)
                result[kn] = stat[subkey]
        else:
            result[key] = stat
    
    # create a few new ones
    result["overlaps_per_object"] = len(result["overlaps"]) / result["traj_count"]
    result["percent_backwards"] = len(result["backward_cars"]) / result["traj_count"]
    result["MAE_x"] = np.mean(np.abs(result["state_error"][:,0]))
    # create spider
    spider = {}
    spider["MOTA"]              = result["mota"]
    spider["Speed"]             = result["bps"] / 30
    spider["X Error"]           = max(0, 1 - result["MAE_x"] / 5.0  )
    spider["% to Extents"]      = result["cause_of_death"]["Exit FOV"]/result["n_pred"]
    spider["Trajectory Length"] = result["x_traveled_avg"] /2000
    spider["Avg Acceleration"]  = max(0, 1 - result["avg_ax_avg"] / 5   )
    spider["Classification"]    = torch.sum(torch.diag(result["confusion_matrix"])) /torch.sum(result["confusion_matrix"])
    spider["Total"]             = ( 2 * spider["MOTA"]             +
                                    3 * spider["Speed"]            +
                                    2 * spider["X Error"]          +
                                    1 * spider["% to Extents"]     +
                                    1 * spider["Trajectory Length"]+
                                    1 * spider["Avg Acceleration"] +
                                    1 * spider["Classification"]   )
    result["spider"] = spider
    
    
     
    ### Save results dict in /data/eval_results
    save_name = "./eval_results/{}.cpkl".format(coll_name)
    with open(save_name, 'wb') as f:
        pickle.dump(result, f)

    ### clean up
    if collection_cleanup: db_cleanup(dbw,coll_name)