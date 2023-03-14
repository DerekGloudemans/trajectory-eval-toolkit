#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:27:59 2022

@author: derek
"""
# 0. Imports
from i24_database_api import DBClient as DBReader
from i24_database_api import DBClient as DBWriter
import numpy as np
import statistics
import motmetrics
import torch


from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt

#[col["name"] for col in list(gtd.db.list_collections())] # list all collections
#delete_collection(["garbage_dump_2","groundtruth_scene_2_TEST","groundtruth_scene_1_TEST","garbage_dump","groundtruth_scene_2"])

def plot_trajectories(gt_resampled,pred_resampled):
    FP_shift = 0
    lane_boundaries = [-10,12,24,36,60,84,96,108,130]
    # for each lane, do a subplot
    
    leg = []
    fig, axs = plt.subplots(2, 4, figsize=(20,8), sharex=True)
    
    
    axs[0,0].set_ylabel("x-position (ft)", fontsize=20)
    axs[1,0].set_ylabel("x-position (ft)", fontsize=20)
    axs[1,0].set_xlabel("Time (s)", fontsize=20)
    axs[1,1].set_xlabel("Time (s)", fontsize=20)
    axs[1,2].set_xlabel("Time (s)", fontsize=20)
    axs[1,3].set_xlabel("Time (s)", fontsize=20)

    axs[0,0].set_title("EB Lane 4", fontsize=20)
    axs[0,1].set_title("EB Lane 3", fontsize=20)
    axs[0,2].set_title("EB Lane 2", fontsize=20)
    axs[0,3].set_title("EB Lane 1", fontsize=20)
    axs[1,0].set_title("WB Lane 4", fontsize=20)
    axs[1,1].set_title("WB Lane 3", fontsize=20)
    axs[1,2].set_title("WB Lane 2", fontsize=20)
    axs[1,3].set_title("WB Lane 1", fontsize=20)
    
    for lidx in range(8):
        ridx = int(lidx//4)
        cidx = int(lidx% 4)
        
        if ridx == 1:
            cidx = 3-cidx
        
        ax = axs[ridx,cidx]
        ax.set_xlim([0,55])
        ax.set_ylim([-100,2100])
        ax.tick_params(axis='both', which='major', labelsize=14)
        # for one lane
        lane_extents = [lane_boundaries[lidx],lane_boundaries[lidx+1]]
        
        colors = np.array([[0,0.2,1,1],
                           [1,0.8,0,1],
                           [1,0,0,1]])
        plot_segments = []
        plot_colors   = []
        # iterate through all gt_resampled
        for fidx in range(1,len(gt_resampled)):
            
    
            # for each trajectory, plot each timestep colored by "tval" and masked by y-range
            for id in gt_resampled[fidx]:
                if id in gt_resampled[fidx-1]:
                     
                    y = gt_resampled[fidx][id]["y"] 
                    if y > lane_extents[0] and y < lane_extents[1]:
                        segment = np.array([[gt_resampled[fidx-1][id]["timestamp"],gt_resampled[fidx-1][id]["x"]],[gt_resampled[fidx][id]["timestamp"],gt_resampled[fidx][id]["x"]]])
                        plot_segments.append(segment)
                        plot_colors.append(colors[gt_resampled[fidx][id]["tval"]])
                        
        plot_segments = np.stack(plot_segments)
        plot_colors = np.stack(plot_colors)
        lc = LineCollection(plot_segments, colors=plot_colors,linewidths = 2)
        ax.add_collection(lc)
        
        
        if True: # plot preds as well, they tend to mostly just overlap at this x-scale
            plot_segments = []
            plot_colors   = []
            # iterate through all pred_resampled
            for fidx in range(1,len(pred_resampled)):
                
        
                # for each trajectory, plot each timestep colored by "tval" and masked by y-range
                for id in pred_resampled[fidx]:
                    if id in pred_resampled[fidx-1]:
                         
                        y = pred_resampled[fidx][id]["y"] 
                        if y > lane_extents[0] and y < lane_extents[1] and pred_resampled[fidx][id]["tval"] != 0:
                            segment = np.array([[pred_resampled[fidx-1][id]["timestamp"],pred_resampled[fidx-1][id]["x"]+FP_shift],[pred_resampled[fidx][id]["timestamp"],pred_resampled[fidx][id]["x"]+FP_shift]])
                            plot_segments.append(segment)
                            plot_colors.append(colors[pred_resampled[fidx][id]["tval"]])
            if len(plot_segments) > 0:       
                plot_segments = np.stack(plot_segments)
                plot_colors = np.stack(plot_colors)
                lc2 = LineCollection(plot_segments, colors=plot_colors,linewidths = 1)
                ax.add_collection(lc2)
            
    handle1, = axs[0,-1].plot([0,0],[-1,-1],color = colors[0])
    handle2, = axs[0,-1].plot([0,0],[-1,-1],color = colors[1])
    handle3, = axs[0,-1].plot([0,0],[-1,-1],color = colors[2])
    axs[0,-1].legend([handle1,handle2,handle3],["True Positive","False Negative","False Positive"], loc='upper right',fontsize = 16)
    
    fig.tight_layout()
    #plt.subplots_adjust(wspace=0.25, hspace=0.05)
    plt.savefig("ts_eval.pdf", bbox_inches="tight")
    fig.show()
    

def state_iou(a,b,threshold = 0.3):
    """ a,b are in state formulation (x,y,l,w,h) - returns iou in range [0,1]"""
    # convert to space formulation
    minxa = np.minimum(a[:,0], a[:,0] + a[:,2] * a[:,5])
    maxxa = np.maximum(a[:,0], a[:,0] + a[:,2] * a[:,5])
    minya = a[:,1] - a[:,3]/2.0
    maxya = a[:,1] + a[:,3]/2.0
    
    minxb = np.minimum(b[:,0], b[:,0] + b[:,2] * b[:,5])
    maxxb = np.maximum(b[:,0], b[:,0] + b[:,2] * b[:,5])
    minyb = b[:,1] - b[:,3]/2.0
    maxyb = b[:,1] + b[:,3]/2.0
    
    a_re = np.stack([minxa,minya,maxxa,maxya]).transpose(1,0)
    b_re = np.stack([minxb,minyb,maxxb,maxyb]).transpose(1,0)
    first = torch.tensor(a_re)
    second = torch.tensor(b_re)
    
    f = a_re.shape[0]
    s = b_re.shape[0]
    
    #get weight matrix
    a = second.unsqueeze(0).repeat(f,1,1).double()
    b = first.unsqueeze(1).repeat(1,s,1).double()
    
    area_a = (a[:,:,2]-a[:,:,0]) * (a[:,:,3]-a[:,:,1])
    area_b = (b[:,:,2]-b[:,:,0]) * (b[:,:,3]-b[:,:,1])
    
    minx = torch.max(a[:,:,0], b[:,:,0])
    maxx = torch.min(a[:,:,2], b[:,:,2])
    miny = torch.max(a[:,:,1], b[:,:,1])
    maxy = torch.min(a[:,:,3], b[:,:,3])
    zeros = torch.zeros(minx.shape,dtype=float,device = a.device)
    
    intersection = torch.max(zeros, maxx-minx) * torch.max(zeros,maxy-miny)
    union = area_a + area_b - intersection + 1e-07
    iou = torch.div(intersection,union)
    

    # nan out any ious below threshold so they can't be matched
    iou = torch.where(iou > threshold,iou, torch.zeros(iou.shape,dtype = torch.float64)*torch.nan)
    return iou


def evaluate(db_param,
             gt_collection   = "groundtruth_scene_1",
             pred_collection = "paradoxical_wallaby--RAW_GT1",
             sample_freq     = 30,
             break_sample = 2700,
             append_db = False,
             iou_threshold = 0.3,
             plot_traj = False):
    
    RESULT = {"postprocessed":True}
    
    prd   = DBReader(**db_param,collection_name = pred_collection)
    
    if append_db:
        dbw = DBWriter(**db_param,collection_name = pred_collection)
        print("Appending assigned gt_ids to predicted data collection.")
    
    db_param2 = db_param.copy()
    db_param2["database_name"] = "trajectories"
    gtd   = DBReader(**db_param2,collection_name = gt_collection)
    
    gts = list(gtd.read_query(None))
    preds = list(prd.read_query(None))
    
    if len(preds) == 0:
        print("Collection {} is empty".format(pred_collection))
        return None
    
    n_gt_resample_points = {}
    n_pred_resample_points = {}
    # 2. Parse collections into expected format for MOT EVAL
    
    
    # ensure objects are monotonic
    for obj in gts:
        for i in range(1,len(obj["x_position"])-1):
            if (obj["x_position"][i+1] - obj["x_position"][i])*obj["direction"] < 0:
                obj["x_position"][i+1] = obj["x_position"][i] + 0.5*(obj["x_position"][i] - obj["x_position"][i-1])
    
    # for each object, get the minimum and maximum timestamp
    ####################################################################################################################################################       SUPERVISED
    gt_times = np.zeros([len(gts),2])
    for oidx,obj in enumerate(gts):
        gt_times[oidx,0] = obj["first_timestamp"]
        gt_times[oidx,1] = obj["last_timestamp"]
        
    pred_times = np.zeros([len(preds),2])
    for oidx,obj in enumerate(preds):
        pred_times[oidx,0] = obj["first_timestamp"]
        pred_times[oidx,1] = obj["last_timestamp"]
    
    # get start and end time
    start_time = max(np.min(gt_times),np.min(pred_times))
    stop_time   = min(np.max(gt_times),np.max(pred_times)) # we don't want to penalize in the event that some GT tracks extend beyond the times for which predictions were generated
    
    ####################################################################################################################################################       SUPERVISED
    gt_length = np.max(gt_times) - np.min(gt_times)
    pred_length = np.max(pred_times) - np.min(pred_times)
    iou = (stop_time-start_time)/(-stop_time + start_time + gt_length + pred_length)
    print("TIME ALIGNMENT:    GT length: {}s, Pred length: {}s,   IOU: {:.2}".format(gt_length,pred_length,iou))
        
    # 3. Resample to fixed timestamps - each element is a list of dicts with id,x,y
    gt_resampled = []
    pred_resampled = []
    
    times = np.arange(start_time,stop_time,1/sample_freq)
    # for each sample time, for each object, if in range find the two surrounding trajectory points and interpolate
    
    max_pred_id = 0
    max_gt_id = 0
    # create id_converters
    CONV_GT = {}
    CONV_PRED = {}
    running_counter = 0
    for item in gts:
        _id = item["_id"]
        CONV_GT[running_counter] = _id
        CONV_GT[_id] = running_counter
        running_counter += 1
        

            
    for item in preds:
        _id = item["_id"]
        CONV_PRED[running_counter] = _id
        CONV_PRED[_id] = running_counter
        running_counter += 1

            
    for st in times:
        #print("On sample time {}, {}% done".format(st,(st-times[0])/(times[-1]-times[0])))
        st_gt = {}
        st_pred = {}
        
        ####################################################################################################################################################       SUPERVISED
        for gidx in range(len(gts)):
            if st >= gt_times[gidx,0] and st <= gt_times[gidx,1]:
                # find closest surrounding times to st for this object - we'll sample sample_idx and sample_idx +1
                sample_idx = 0
                try:
                    while gts[gidx]["timestamp"][sample_idx+1] <= st:
                        sample_idx += 1
                except IndexError:
                    print("IndexError")
                    continue
                    
                # linearly interpolate 
                diff = gts[gidx]["timestamp"][sample_idx+1] - gts[gidx]["timestamp"][sample_idx]
                c2 = (st - gts[gidx]["timestamp"][sample_idx]) / diff
                
                x_int = gts[gidx]["x_position"][sample_idx] + c2 * (gts[gidx]["x_position"][sample_idx+1] - gts[gidx]["x_position"][sample_idx])
                y_int = gts[gidx]["y_position"][sample_idx] + c2 * (gts[gidx]["y_position"][sample_idx+1] - gts[gidx]["y_position"][sample_idx])
                t_int = (gts[gidx]["timestamp"][sample_idx]  + c2 * (gts[gidx]["timestamp"][sample_idx+1]  - gts[gidx]["timestamp"][sample_idx])) - start_time
                
                obj = {"id":CONV_GT[gts[gidx]["_id"]],
                        "x":x_int,
                        "y":y_int,
                        "tval":1, # by default false-negative
                        "timestamp":t_int
                        }
                st_gt[obj["id"]] = obj
                
                try:
                    n_gt_resample_points[CONV_GT[gts[gidx]["_id"]]] += 1
                except:
                    n_gt_resample_points[CONV_GT[gts[gidx]["_id"]]] = 1
                
        gt_resampled.append(st_gt)
        
        for pidx in range(len(preds)):
            if st >= pred_times[pidx,0] and st <= pred_times[pidx,1]:
                # find closest surrounding times to st for this object - we'll sample sample_idx and sample_idx +1
                sample_idx = 0
                SKIP = False
                while preds[pidx]["timestamp"][sample_idx+1] <= st:
                    
#                    print(pidx,sample_idx,len(preds[pidx]["timestamp"]))
                    sample_idx += 1
                    if sample_idx+1 >= len(preds[pidx]["timestamp"]):
                        SKIP = True
                        break
                
                if SKIP:
                    continue
                    
                # linearly interpolate 
                diff = preds[pidx]["timestamp"][sample_idx+1] - preds[pidx]["timestamp"][sample_idx]
                c2 = (st - preds[pidx]["timestamp"][sample_idx]) / diff
                
                x_int = preds[pidx]["x_position"][sample_idx] + c2 * (preds[pidx]["x_position"][sample_idx+1] - preds[pidx]["x_position"][sample_idx])
                y_int = preds[pidx]["y_position"][sample_idx] + c2 * (preds[pidx]["y_position"][sample_idx+1] - preds[pidx]["y_position"][sample_idx])
                t_int = (preds[pidx]["timestamp"][sample_idx] + c2 * (preds[pidx]["timestamp"][sample_idx+1] - preds[pidx]["timestamp"][sample_idx])) - start_time
                
                try:
                    conf_int = preds[pidx]["detection_confidence"][sample_idx] + c2 * (preds[pidx]["detection_confidence"][sample_idx+1] - preds[pidx]["detection_confidence"][sample_idx])
                    xvar_int = preds[pidx]["variance"][sample_idx][0] + c2 * (preds[pidx]["variance"][sample_idx+1][0] - preds[pidx]["variance"][sample_idx][0])
                except:
                    xvar_int = -1
                    conf_int = -1
                
                obj = {"id":CONV_PRED[preds[pidx]["_id"]],
                        "x":x_int,
                        "y":y_int,
                        "timestamp":t_int,
                        "tval":2, # by default false-positive
                        "xvar":xvar_int,
                        "conf":conf_int,
                        }
                st_pred[obj["id"]] = obj
                
                try:
                    n_pred_resample_points[CONV_PRED[preds[pidx]["_id"]]] += 1
                except:
                    n_pred_resample_points[CONV_PRED[preds[pidx]["_id"]]] = 1
                
        pred_resampled.append(st_pred)
        
        #if len(gt_resampled) > 100: break
                
    print("Resampling complete")
    
    
    # Assemble dictionary of relevant attributes (median dimensions etc) indexed by id
    
    ####################################################################################################################################################       SUPERVISED
    gt_dict = {}
    for item in gts:
        gt_dict[CONV_GT[item["_id"]]] = item
        if type(item["length"]) == list:
            item["length"] = item["length"][0]
            item["width"] = item["width"][0]
            item["height"] = item["height"][0]
    pred_dict = {}
    for item in preds:
        if type(item["length"]) == list:
            RESULT["postprocessed"] = False
            item["length"] = statistics.median(item["length"])
            item["width"] = statistics.median(item["width"])
            item["height"] = statistics.median(item["height"])
        pred_dict[CONV_PRED[item["_id"]]] = item
    
    # gt_id_converter_dict = {}
    # for frag_id in gt_dict:
    #     gt_id_converter_dict[frag_id] = gt_dict[frag_id]["_id"]
    
    # 4. Run MOT Eval to get MOT-style metrics
    
    ####################################################################################################################################################       SUPERVISED
    # various accumulators
    acc = motmetrics.MOTAccumulator(auto_id = True)
    state_errors = [] # each entry will be a state_size array with state errors for each state variable
    gt_frag_accumulator = {} # for each gt, a list of all pred ides associated with gt
    pred_frag_accumulator = {} # for each pred, a list of all gt ids associated with pred
    prev_event_id = -1
    event_list = []
    
    # we'll run through frames and compute iou for all pairs, which will give us the set of MOT metrics
    for fidx in range(len(gt_resampled)):#[:break_sample]:
        #print("On sample {} ({}s)".format(fidx, times[fidx]))
        
        gt_ids = [item["id"] for item in gt_resampled[fidx].values()]
        pred_ids = [item["id"] for item in pred_resampled[fidx].values()]
        gt_pos = []
        pred_pos = []
        
        for item in gt_resampled[fidx].values():
            id = item["id"]
            pos = np.array([item["x"],item["y"],gt_dict[id]["length"],gt_dict[id]["width"],gt_dict[id]["height"],gt_dict[id]["direction"]])
            gt_pos.append(pos)
            
        for item in pred_resampled[fidx].values():
            id = item["id"]
            pos = np.array([item["x"],item["y"],pred_dict[id]["length"],pred_dict[id]["width"],pred_dict[id]["height"],pred_dict[id]["direction"]])
            pred_pos.append(pos)
       
        # compute IOUs for gts and preds
        if len(pred_pos) == 0 or len(gt_pos) == 0:
            
            ious = np.zeros([len(gt_ids),1])
        else:
            ious = state_iou(np.stack(gt_pos),np.stack(pred_pos),threshold = iou_threshold)
    
        
        acc.update(gt_ids,pred_ids,ious)   
    
    
    #%%##################### Compute Metrics #####################
    
    # Summarize MOT Metrics
    metric_module = motmetrics.metrics.create()
    summary = metric_module.compute(acc) 
    for metric in summary:
        print("{}: {}".format(metric,summary[metric][0]))  
    
    FP = summary["num_false_positives"][0]
    FN = summary["num_misses"][0]
    TP = (summary["recall"][0] * FN) / (1-summary["recall"][0])
    
    ####################################################################################################################################################       SUPERVISED
    
    for metric in summary:
        RESULT[metric] = summary[metric][0]
    RESULT["true_negative_rate"] = 1.0
    RESULT["total_gt_annotations"] = TP + FN
        
    # Now parse events into matchings
    #events = acc.events.values.tolist()
    confusion_matrix = torch.zeros([8,8])
    state_errors = []
    
    match_iou = []
    match_conf = []
    match_xvar = []
    
    # we want to record the position of each object across time and whether it was a FP, FN, or TP at that time
    # additionally we want to record the position of each object switch and fragmentation
    frag_dict = [] # contains [[timestamp,x_pos,y_pos],[]...]
    switch_dirct = [] # contains [[timestamp,x_pos,y_pos],[]...]
    
    match_matrix = np.zeros([len(gts)*5,len(preds)*5])
    gt_counts = np.zeros(len(gts)*5)
    pred_counts = np.zeros(len(preds)*5)
    
    # we care only about
    relevant = ["MATCH","SWITCH","TRANFER","ASCEND","MIGRATE"]
    for event in acc.events.iterrows():
        fidx = event[0][0]
        event = event[1]
        if event[0] in relevant:
            gt_id   = int(event[1])
            pred_id = int(event[2])
            
            if event[0] == "MATCH":
                gt_resampled[fidx][gt_id]["tval"] = 0 # 0 = TP, 1 = FN, 2 = FP
                pred_resampled[fidx][pred_id]["tval"] = 0 # 0 = TP, 1 = FN, 2 = FP
            
            # match_matrix[gt_id,pred_id] += 1
            # gt_counts[gt_id] += 1
            # pred_counts[pred_id] += 1
            
            # store pred_ids in gt_dict
            if "assigned_frag_ids" not in gt_dict[gt_id].keys():
                gt_dict[gt_id]["assigned_frag_ids"] = [pred_id]
            else:
                gt_dict[gt_id]["assigned_frag_ids"].append(pred_id)
            
            # store gt_ids in pred_dict
            if "assigned_gt_ids" not in pred_dict[pred_id].keys():
                pred_dict[pred_id]["assigned_gt_ids"] = [gt_id]
            else:
                pred_dict[pred_id]["assigned_gt_ids"].append(gt_id)
                
            # compute state errors
            for pos in pred_resampled[fidx].values():
                if pos["id"] == pred_id:
                    pred_pos = np.array([pos["x"],pos["y"],pred_dict[pred_id]["length"],pred_dict[pred_id]["width"],pred_dict[pred_id]["height"]])
                    
                    # note that these are interpolated conf and variance values
                    if event[0] == "MATCH":
                        iou = event[3]
                        pred_conf = pos["conf"]
                        pred_xvar = pos["xvar"]
                        match_conf.append(pred_conf)
                        match_xvar.append(pred_xvar)
                        match_iou.append(iou)
                    break
                
            for pos in gt_resampled[fidx].values():
                if pos["id"] == gt_id:
                    gt_pos = np.array([pos["x"],pos["y"],gt_dict[gt_id]["length"],gt_dict[gt_id]["width"],gt_dict[gt_id]["height"]])
                    break
            state_err = pred_pos - gt_pos
            state_err[0] *= gt_dict[gt_id]["direction"]
            state_errors.append(state_err)
            
            gt_cls = gt_dict[gt_id]["coarse_vehicle_class"]
            pred_cls = pred_dict[pred_id]["coarse_vehicle_class"]
            
            confusion_matrix[gt_cls,pred_cls] += 1
        
        # elif event[0] == "MISS":
        #     gt_id = int(event[1])
        #     gt_counts[gt_id] += 1
            
        #     gt_resampled[fidx][gt_id]["tval"] = 1 # 0 = TP, 1 = FN, 2 = FP
            
            
        # elif event[0] == "FP":
        #     pred_id = int(event[2])
        #     pred_counts[pred_id] += 1
        #     pred_resampled[fidx][pred_id]["tval"] = 2 # 0 = TP, 1 = FN, 2 = FP
               
            
    # HOTA_DET_IOU = TP/(TP+FP+FN)
    # HOTA_ASS_IOUs = []
    # for i in range(len(gts)):
    #     for j in range(len(preds)):
            
    #         if match_matrix[i,j] > 0:
    #             match_iou = match_matrix[i,j] / (pred_counts[j] + gt_counts[i] -match_matrix[i,j] )
                
    #             HOTA_ASS_IOUs.append(match_iou)
    # HOTA_ASS_IOU = sum(HOTA_ASS_IOUs)/len(HOTA_ASS_IOUs)
        
    if plot_traj:
        plot_trajectories(gt_resampled,pred_resampled)
    ####################################################################################################################################################       SUPERVISED

    ### State Accuracy Metrics
    state_errors = np.stack(state_errors)
    #print("\n")
    #print("RMSE state error:")
    #print(np.sqrt(np.mean(np.power(state_errors,2),axis = 0)))
    
    RESULT["state_error"] = state_errors
    RESULT["match_overlap"] = {"conf":match_conf,
                                "iou":match_iou,
                                "var":match_xvar
                                }
    
    
    ### Unsupervised metrics  --- statistics
    
    # Total Variation
    travelled = 0
    diffx = 0
    diffy = 0
    for traj in pred_dict.values():
        for idx in range(1,len(traj["x_position"])):
            diffx += np.abs(traj["x_position"][idx] - traj["x_position"][idx-1])
            diffy += np.abs(traj["y_position"][idx] - traj["y_position"][idx-1])
        travelled += np.abs(traj["ending_x"] - traj["starting_x"])
    
    # Total # Trajectories
    total_gt = len(gt_dict)
    total_pred = len(pred_dict)
    
    RESULT["x_variation"] = diffx/travelled
    RESULT["y_variation"] = diffy/total_pred
    
    
    # Roadway extents
    minx = np.inf
    maxx = -np.inf
    
    for traj in gt_dict.values():
        if min(traj["x_position"]) < minx:
            minx = min(traj["x_position"])
        if max(traj["x_position"]) > maxx:
            maxx = max(traj["x_position"])
    RESULT["FOV_extents"] = [minx,maxx]
    
    
    ### Other Trajectory Quantities
    # Average Trajectory length and duration
    avg_gt_length = []
    for traj in gt_dict.values():
        avg_gt_length.append(np.abs(traj["ending_x"] - traj["starting_x"]))
    avg_gt_length = sum(avg_gt_length) / len(avg_gt_length)
    
    avg_pred_length = []
    for traj in pred_dict.values():
        avg_pred_length.append(np.abs(traj["ending_x"] - traj["starting_x"]))
    avg_pred_length = sum(avg_pred_length) / len(avg_pred_length)
    
    avg_gt_dur = []
    for traj in gt_dict.values():
        avg_gt_dur.append(np.abs(traj["last_timestamp"] - traj["first_timestamp"]))
    avg_gt_dur = sum(avg_gt_dur) / len(avg_gt_dur)
    avg_pred_dur = []
    for traj in pred_dict.values():
        avg_pred_dur.append(np.abs(traj["last_timestamp"] - traj["first_timestamp"]))
    avg_pred_dur = sum(avg_pred_dur) / len(avg_pred_dur)
    
    
    
    
    # what percentage of GT trajectories have NO matches
    gt_no_matches = 0
    gt_assigned_id_count = []
    for traj in gt_dict.values():
        if "assigned_frag_ids" not in traj.keys():
            gt_no_matches += 1
        else:
            unique_ids = list(set(traj["assigned_frag_ids"]))
            gt_assigned_id_count.append(len(unique_ids))
    
    gt_no_matches /= len(gt_dict)
    gt_avg_assigned_matches = sum(gt_assigned_id_count)/len(gt_assigned_id_count)
    
    # what percentage of Pred objects have NO matches
    pred_no_matches = 0
    pred_assigned_id_count = []
    for traj in pred_dict.values():
        if "assigned_gt_ids" not in traj.keys():
            pred_no_matches += 1
        else:
            unique_ids = list(set(traj["assigned_gt_ids"]))
            pred_assigned_id_count.append(len(unique_ids))
            
            unique_ids = [CONV_GT[int(id)] for id in unique_ids]            
            if append_db:
                fragment_id = traj["_id"]
                dbw.collection.update_one({'_id':fragment_id},{'$push':{'gt_ids':unique_ids}},upsert=True)
            
    pred_no_matches /= len(pred_dict)
    pred_avg_assigned_matches = sum(pred_assigned_id_count)/len(pred_assigned_id_count)
    
    # what percentage of GT is covered, per GT
    per_gt_recall = []
    for key in gt_dict.keys():
        traj = gt_dict[key]
        try:
            this_traj_recall = len(traj["assigned_frag_ids"]) / n_gt_resample_points[key]
        except:
            this_traj_recall = 0
        per_gt_recall.append(this_traj_recall)
    RESULT["per_gt_recall"] = per_gt_recall
    
    # what percentage of preds cover, per pred
    per_pred_precision = []
    for key in pred_dict.keys():
        traj = pred_dict[key]
        try:
            this_traj_prec = len(traj["assigned_gt_ids"]) / n_pred_resample_points[key]
        except:
            this_traj_prec = 0
        per_pred_precision.append(this_traj_prec)
    RESULT["per_pred_precision"] = per_pred_precision

    
    RESULT["gt_match"]      =    1 - gt_no_matches
    RESULT["pred_match"]    = 1 - pred_no_matches
    RESULT["pred_avg_matches"] = pred_avg_assigned_matches
    RESULT["gt_avg_matches"]   = gt_avg_assigned_matches
    
    # Class confusion matrix
    print("\n")
    print("Class confusion matrix:")
    
    #vec = confusion_matrix.sum(dim = 1) # total number of gts per class
    #confusion_matrix /= (vec.unsqueeze(1).expand(8,8) + 0.001)
    print(torch.round(confusion_matrix,decimals = 2))
    
    RESULT["confusion_matrix"] = confusion_matrix
    
    # Flag counter
    COD = {}
    COD_length = {}
    COD["Active at End"] = 0
    COD_length["Active at End"] = []

    for traj in pred_dict.values():
        death =  traj["flags"][0]
        if death not in COD.keys():
            COD[death] = 1
            COD_length[death] = [len(traj["x_position"])]
        else:
            COD[death] += 1
            COD_length[death].append(len(traj["x_position"]))
    RESULT["cause_of_death"] = COD
    
    RESULT["observations_by_cause_of_death"] = COD_length
    
    
    
    
    # A few more
    n_pred = len(preds)
    n_gt   = len(gts)
    RESULT["n_gt"]   = n_gt
    RESULT["n_pred"] = n_pred
    
    RESULT["num_switches_per_gt"] = RESULT["num_switches"]/n_gt
    RESULT["num_fragmentations_per_gt"] = RESULT["num_fragmentations"]/n_gt
    RESULT["mostly_tracked%"] = RESULT["mostly_tracked"]/n_gt
    RESULT["mostly_lost%"] = RESULT["mostly_lost"]/n_gt
    RESULT["partially_tracked%"] = RESULT["partially_tracked"]/n_gt

    RESULT["gen_time"] = preds[0]["_id"].generation_time   
    
    RESULT["gt_x_traveled"] = [np.abs(max(traj["x_position"]) - min(traj["x_position"])) for traj in gt_dict.values()]
        

    
    #%% COMPUTE HOTA METRICS AT VARYING IOU THRESHOLDS      
    if True:
            print("Computing HOTA.. get comfortable, this could take a bit.")
            all_HOTA = []
            for alpha in np.arange(0.05,1,0.05):
                # various accumulators
                acc = motmetrics.MOTAccumulator(auto_id = True)
                state_errors = [] # each entry will be a state_size array with state errors for each state variable
                
                
                # we'll run through frames and compute iou for all pairs, which will give us the set of MOT metrics
                for fidx in range(len(gt_resampled)):#[:break_sample]:
                    #print("On sample {} ({}s)".format(fidx, times[fidx]))
                    
                    gt_ids = [item["id"] for item in gt_resampled[fidx].values()]
                    pred_ids = [item["id"] for item in pred_resampled[fidx].values()]
                    gt_pos = []
                    pred_pos = []
                    
                    for item in gt_resampled[fidx].values():
                        id = item["id"]
                        pos = np.array([item["x"],item["y"],gt_dict[id]["length"],gt_dict[id]["width"],gt_dict[id]["height"],gt_dict[id]["direction"]])
                        gt_pos.append(pos)
                        
                    for item in pred_resampled[fidx].values():
                        id = item["id"]
                        pos = np.array([item["x"],item["y"],pred_dict[id]["length"],pred_dict[id]["width"],pred_dict[id]["height"],pred_dict[id]["direction"]])
                        pred_pos.append(pos)
                   
                    # compute IOUs for gts and preds
                    if len(pred_pos) == 0 or len(gt_pos) == 0:
                        
                        ious = np.zeros([len(gt_ids),1])
                    else:
                        ious = state_iou(np.stack(gt_pos),np.stack(pred_pos),threshold = alpha)
                
                    
                    acc.update(gt_ids,pred_ids,ious)   
                
                
                #%%##################### Compute Metrics #####################
                
                # Summarize MOT Metrics
                metric_module = motmetrics.metrics.create()
                summary = metric_module.compute(acc) 
                # for metric in summary:
                #     print("{}: {}".format(metric,summary[metric][0]))  
                
                FP = summary["num_false_positives"][0]
                FN = summary["num_misses"][0]
                TP = (summary["recall"][0] * FN) / (1-summary["recall"][0])
                
                ####################################################################################################################################################       SUPERVISED
                

                    
                # Now parse events into matchings              
                match_matrix = np.zeros([len(gts)*5,len(preds)*5])
                gt_counts = np.zeros(len(gts)*5)
                pred_counts = np.zeros(len(preds)*5)
                
                # we care only about
                #relevant = ["MATCH","SWITCH","TRANFER","ASCEND","MIGRATE"]
                for event in acc.events.iterrows():
                    fidx = event[0][0]
                    event = event[1]
                    if event[0] == "MATCH":
                        gt_id   = int(event[1])
                        pred_id = int(event[2])
                        
                        match_matrix[gt_id,pred_id] += 1
                        gt_counts[gt_id] += 1
                        pred_counts[pred_id] += 1
                        
                    elif event[0] == "MISS":
                        gt_id = int(event[1])
                        gt_counts[gt_id] += 1
                    elif event[0] == "FP":
                        pred_id = int(event[2])
                        pred_counts[pred_id] += 1
                    
                        
                HOTA_DET_IOU = TP/(TP+FP+FN)
                HOTA_ASS_IOUs = []
                for i in range(len(gt_counts)):
                    for j in range(len(pred_counts)):
                        
                        if match_matrix[i,j] > 0:
                            match_iou = match_matrix[i,j] / (pred_counts[j] + gt_counts[i] -match_matrix[i,j] )
                            
                            HOTA_ASS_IOUs.append(match_iou)
                HOTA_ASS_IOU = sum(HOTA_ASS_IOUs)/(len(HOTA_ASS_IOUs)+0.1)
                
                HOTA = (HOTA_ASS_IOU * HOTA_DET_IOU)**0.5
                all_HOTA.append(HOTA)
                print("@ alpha = {}: {:.3f},{:.3f},{:.3f}".format(alpha,HOTA, HOTA_DET_IOU,HOTA_ASS_IOU))
        
            HOTA = sum(all_HOTA)/len(all_HOTA)
            RESULT["HOTA"] = HOTA
            print("HOTA: {}".format(HOTA))
    
    return  RESULT

def bonus_evaluate(db_param,
             pred_collection = "paradoxical_wallaby--RAW_GT1",
             sample_freq     = 30,
             append_db = False,
             iou_threshold = 0.3):
    
    
    prd   = DBReader(**db_param,collection_name = pred_collection)
    
    if append_db:
        dbw = DBWriter(**db_param,collection_name = pred_collection)
        print("Appending assigned gt_ids to predicted data collection.")
    
    preds = list(prd.read_query(None))
    
    if len(preds) == 0:
        print("Collection {} is empty".format(pred_collection))
        return None
    
   
    n_pred_resample_points = {}
    # 2. Parse collections into expected format for MOT EVAL
    
    # for each object, get the minimum and maximum timestamp
    ####################################################################################################################################################       SUPERVISE
    pred_times = np.zeros([len(preds),2])
    for oidx,obj in enumerate(preds):
        pred_times[oidx,0] = obj["first_timestamp"]
        pred_times[oidx,1] = obj["last_timestamp"]
    
    # get start and end time
    start_time = np.min(pred_times)
    stop_time   = np.max(pred_times) # we don't want to penalize in the event that some GT tracks extend beyond the times for which predictions were generated
    
    ####################################################################################################################################################       SUPERVISED
    
        
    # 3. Resample to fixed timestamps - each element is a list of dicts with id,x,y
    pred_resampled = []
    
    times = np.arange(start_time,stop_time,1/sample_freq)
    # for each sample time, for each object, if in range find the two surrounding trajectory points and interpolate
    
    
    # create id_converters
    CONV_PRED = {}
    running_counter = 0
   
    for item in preds:
        _id = item["_id"]
        CONV_PRED[running_counter] = _id
        CONV_PRED[_id] = running_counter
        running_counter += 1
    
    
    
    # Assemble dictionary of relevant attributes (median dimensions etc) indexed by id
    
    ####################################################################################################################################################       SUPERVISED
  
    pred_dict = {}
    class_counter = np.zeros(10)
    for item in preds:
        class_counter[item["coarse_vehicle_class"]] += 1
        if type(item["length"]) == list:
            item["length"] = statistics.median(item["length"])
            item["width"] = statistics.median(item["width"])
            item["height"] = statistics.median(item["height"])
        pred_dict[CONV_PRED[item["_id"]]] = item
    
    # gt_id_converter_dict = {}
    # for frag_id in gt_dict:
    #     gt_id_converter_dict[frag_id] = gt_dict[frag_id]["_id"]
    
    # 4. Run MOT Eval to get MOT-style metrics
    
    ####################################################################################################################################################       SUPERVISED
   
    
    ### Unsupervised metrics  --- statistics
    RESULT = {}
    
    # Total Variation
    travelled = 0
    diffx = 0
    diffy = 0
    for traj in pred_dict.values():
        for idx in range(1,len(traj["x_position"])):
            diffx += np.abs(traj["x_position"][idx] - traj["x_position"][idx-1])
            diffy += np.abs(traj["y_position"][idx] - traj["y_position"][idx-1])
        travelled += np.abs(traj["ending_x"] - traj["starting_x"])
    
    # Total # Trajectories
    total_pred = len(pred_dict)
    
    RESULT["x_variation"] = diffx/travelled
    RESULT["y_variation"] = diffy/total_pred
    
    
    # Roadway extents
    minx = np.inf
    maxx = -np.inf
    
   
    
    
    ### Other Trajectory Quantities
    # Average Trajectory length and duration
    
    
    avg_pred_length = []
    for traj in pred_dict.values():
        avg_pred_length.append(np.abs(traj["ending_x"] - traj["starting_x"]))
    avg_pred_length = sum(avg_pred_length) / len(avg_pred_length)
    
   
    avg_pred_dur = []
    for traj in pred_dict.values():
        avg_pred_dur.append(np.abs(traj["last_timestamp"] - traj["first_timestamp"]))
    avg_pred_dur = sum(avg_pred_dur) / len(avg_pred_dur)
    
    # Flag counter
    COD = {}
    COD_length = {}
    COD["Active at End"] = 0
    COD_length["Active at End"] = []

    for traj in pred_dict.values():
        death =  traj["flags"][0]
        if death not in COD.keys():
            COD[death] = 1
            COD_length[death] = [len(traj["x_position"])]
        else:
            COD[death] += 1
            COD_length[death].append(len(traj["x_position"]))
    RESULT["cause_of_death"] = COD
    
    RESULT["observations_by_cause_of_death"] = COD_length
    
    
    
    
    # A few more
    n_pred = len(preds)
    RESULT["n_pred"] = n_pred

    RESULT["gen_time"] = preds[0]["_id"].generation_time   
            
    RESULT["classes"] = class_counter
    
    return  RESULT


if __name__ == "__main__":
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
    r = evaluate(db_param)




# 7. Print out summary
