#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:58:05 2022

@author: yanbing_wang
compare the pickle files of raw and reconciled
"""
import _pickle as pickle
def get_pickle(name):
    with open(name,"rb") as f:
        result = pickle.load(f)
        return result
    
# print metrics side by side
def compare_metrics(raw, rec, attr):
    print("{}: {:.3f}, {:.3f}".format(attr, raw["metrics"][attr], rec["metrics"][attr]))
    
def list_metrics(raw, rec, attr_list):
    for attr in attr_list:
        if attr in raw["metrics"]:
            compare_metrics(raw, rec, attr)
        else:
            compare_stats(raw, rec, attr)
        
def compare_stats(raw, rec, attr):
    if isinstance(raw["statistics"][attr], int):
        print("{}: {}, {}".format(attr, raw["statistics"][attr], rec["statistics"][attr]))
    else:
        for agg in ["min", "max", "avg", "median", "stdev"]:
            print("{}: {:.3f}, {:.3f}".format(attr + " @ " + agg, raw["statistics"][attr][agg], rec["statistics"][attr][agg]))
        
if __name__ == "__main__":
    collection_name = "morose_panda--RAW_GT1"
    rec_suffix = "_boggles"
    metrics = ['traj_count',"mota", "motp", "idp",'idr', 'mostly_tracked', 'partially_tracked', 'mostly_lost', 'num_false_positives', 'num_misses', 'num_switches', 'num_fragmentations',
               "avg_vx", 'min_spacing']
    
    
    raw = get_pickle("eval_results/{}.cpkl".format(collection_name))
    rec = get_pickle("eval_results/{}.cpkl".format(collection_name+rec_suffix))
    
    print("IOUT: {}, {}, {}".format(raw["iou_threshold"], rec["iou_threshold"], rec["description"]))
    list_metrics(raw, rec, metrics)