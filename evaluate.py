
from sup_metrics     import evaluate
from unsup_statistics  import call

from i24_database_api import DBClient

import numpy as np
import os
import time
import torch
import _pickle as pickle
import warnings
warnings.filterwarnings("ignore")

from eval_dashboard import main as dash




def get_pickle(name):
    with open(name,"rb") as f:
        result = pickle.load(f)
        return result



def save_collection_and_reconciled_descendants(collection_name):
    pass
    
def main(gt_coll = None): #"groundtruth_scene_1_130", TAG = "GT1"):
    pc  = None
    sc = None
    for db_name in ["trajectories","reconciled"]:
        to_remove = []

        db_param = {
              "host":"10.80.4.91",
              "port":27017,
              "username": "mongo-admin",
              "password": "i24-data-access",
              "database_name": db_name,      
              "server_id": 1,
              "session_config_id": 1,
              }
        
        IOUT = 0.5
        coll_name = None
        append_db = False
        if db_name == "trajectories": append_db  = True

        # connect to database
        dbw   = DBClient(**db_param)
        existing_collections = dbw.list_collection_names() # list all collections
        existing_collections.sort()
        print("\n Existing Collections in database -  {}:".format(dbw.database_name)) 
        [print(item) for item in existing_collections]
        print("\n")
        
        if coll_name is None:
            to_evaluate = existing_collections
        else:
            to_evaluate = [coll_name]
            
        for coll_name in to_evaluate:
            if TAG in coll_name:
                
                # check whether has already been evaluated
                save_name = "./data/eval_results/{}.cpkl".format(coll_name)
                if os.path.exists(save_name):
                    continue
                
                # generate comment
                #comment = input("Description of run settings / test for storage with evaluation results: ")    
                result = {}
                #result["description"] = comment
                result["bps"]  = 1
        
                # try to overwrite this stuff with an existing runfile
                base_name = "./data/run_results/"+ coll_name.split("__")[0] + "_run_results.cpkl"
                try:
                    result = get_pickle(base_name)
                    print("Reloaded run_result for {}".format(coll_name))
                except:
                    print("Failed to reload run result for {}".format(coll_name))
    
                result["name"] = coll_name
                result["iou_threshold"] = IOUT
    
                try:
                    if True:
        
                        start = time.time()
                        ### supervised evaluation
                        if gt_coll is not None:
                            metrics = evaluate(db_param,gt_collection = gt_coll,pred_collection = coll_name, sample_freq = 30, break_sample = 2700,append_db = append_db,iou_threshold = IOUT)
                            
                            if metrics is None:
                                to_remove.append(coll_name)
                                continue
                            
                            result["iou_threshold"] = IOUT
                            result["gt"] = gt_coll        
                            for key in metrics.keys():
                                result[key] = metrics[key]
                    
                        ### unsupervised statistics
                        print(db_param)
                        statistics = call(db_param,coll_name)
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
                        result["percens"] = len(result["backward_cars"]) / result["traj_count"]
                        result["MAE_x"] = np.mean(np.abs(result["state_error"][:,0]))
                
                        
                 
                        ## Save results dict in /data/eval_results
                        with open(save_name, 'wb') as f:
                            pickle.dump(result, f)
                except Exception as e:
                    print("Error processing collection {}: {}".format(coll_name,e))
                            
        
        if True and len(to_remove) > 0:
            print("\n The following collections are empty: {}".format(to_remove))
            inp = input("Do you want to remove these collections? (Y/N)")
            if inp == "Y":
                dbw.delete_collections(to_remove)
    
    dash(mode = "latest v latest",close = 1000)
    return result
    
if __name__ == "__main__":
    result = main()
    
    