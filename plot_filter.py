

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
import _pickle as pickle

import matplotlib.gridspec as grid_spec
import matplotlib.pyplot as plt


global classmap
classmap = ["sedan","midsize","van","pickup","semi","truck"]


def evaluate(db_param,
             gt_collection   = "groundtruth_scene_1_130",
             pred_collection = "sanctimonious_beluga--RAW_GT1",
             kf_param_path   = "/home/derek/Documents/i24/i24_track/data/kf_params/kf_params_save3.cpkl",
             n_plot = 100):
    
    
    with open(kf_param_path,"rb") as f:
        kf_params = pickle.load(f)
    meas_cov = kf_params["R"].diag().sqrt()
    #print(meas_cov)
    
    
    
    
    
    prd   = DBReader(**db_param,collection_name = pred_collection)
    
    db_param2 = db_param.copy()
    db_param2["database_name"] = "trajectories"
    gtd   = DBReader(**db_param2,collection_name = gt_collection)
    
    gts = list(gtd.read_query(None))
    preds = list(prd.read_query(None))
    
    if len(preds) == 0:
        print("Collection {} is empty".format(pred_collection))
        return None
    
    
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
    
        
    plot_pred_idx = [i for i in range(n_plot)]
    for p_idx in plot_pred_idx:
        
       # if preds[p_idx]["flags"][0] in ["Active at End", "Exit FOV"]:
       #     continue
        
        pred = preds[p_idx]
        
        pre_shift = 0.01
        
        prior_covariance     = np.array(pred["prior_covariance"])
        prior                = np.array(pred["prior"])
        measurement          = np.array(pred["detection"])
        posterior_covariance = np.array(pred["posterior_covariance"])
        
        try:
            posterior            = np.stack([np.array(pred[key]) for key in ["x_position","y_position","length","width","height","velocity"]]).transpose(1,0)
        except:
            posterior            = np.stack([np.array(pred[key]) for key in ["x_position","y_position","length","width","height"]]).transpose(1,0)  
            
        times                = np.array(pred["timestamp"])
        meas_conf            = np.array(pred["detection_confidence"])
        
        times -= pred_times[p_idx,0]
        
        pre_times            = times  - pre_shift
        
        measurement[0,:] = posterior[0,:5] # this must be true
        #prior = prior[1:,:]
        #prior_covariance = prior_covariance[1:,:] # first prior and prior covariance are garbage
        #pre_times = pre_times[1:]
        
        # interleave prior and posterior covariance
        # ravel_state = []
        # for i in range(len(posterior)):
        #     ravel_state.append(posterior[i])
        #     if i != len(prior):
        #         ravel_state.append(prior[i])
        # state = np.stack(ravel_state)    
            
        # # interleave state (prior and posterior)
        # ravel_cov = []
        # for i in range(len(posterior_covariance)):
        #     ravel_cov.append(posterior_covariance[i])
        #     if i != len(prior_covariance):
        #         ravel_cov.append(prior_covariance[i])
        # covariance = np.stack(ravel_cov)
        
        # # interleave state times
        # state_times = []
        # for i in range(len(times)):
        #     state_times.append(times[i])
        #     if i != len(times)-1:
        #         state_times.append(pre_times[i])
        # state_times = np.stack(state_times)
        
        ravel_state = []
        for i in range(len(posterior)):
            ravel_state.append(prior[i])
            ravel_state.append(posterior[i])
        state = np.stack(ravel_state)    
            
        # interleave state (prior and posterior)
        ravel_cov = []
        for i in range(len(posterior_covariance)):
            ravel_cov.append(prior_covariance[i])
            ravel_cov.append(posterior_covariance[i])
        covariance = np.stack(ravel_cov)
        
        # interleave state times
        state_times = []
        state_times_unshifted = []
        for i in range(len(times)):
            state_times.append(times[i] - pre_shift)
            state_times.append(times[i] + pre_shift)
            state_times_unshifted.append(times[i])
            state_times_unshifted.append(times[i])
        state_times = np.stack(state_times)
        state_times_unshifted = np.stack(state_times_unshifted)

        state_times = state_times[1:]
        state_times_unshifted = state_times_unshifted[1:]
        covariance = covariance[1:,:]
        state = state[1:,:]
        
        
        # get relevant gts (fall within same x_range at same times)
        relevant_gts = []
        for g_idx in range(len(gts)):
            gt = gts[g_idx]
            if gt_times[g_idx,1] > pred_times[p_idx,0] or gt_times[g_idx,0] < pred_times[p_idx,1]: # overlap in time
                if gt["direction"] == pred["direction"]: 
                    
                    
                    # get x and y pos and compare to pred x and y pos at same time
                    
                    pred_time = pred_times[p_idx,0]
                    pred_x = state[1,0]
                    pred_y = state[1,1]
                    
                    gt_time = -np.inf
                    t_idx = -1
                    try:
                        while gt_time < pred_time:
                            t_idx += 1
                            gt_time = gt["timestamp"][t_idx]
                    except:
                        try:
                            pred_time = pred_times[p_idx,1]
                            while gt_time < pred_time:
                                t_idx += 1
                                gt_time = gt["timestamp"][t_idx]
                        except:
                            continue
                            
                    gt_x = gt["x_position"][t_idx]
                    gt_y = gt["y_position"][t_idx]
                    
                    if np.abs(gt_x - pred_x) < 30 and np.abs(gt_y - pred_y) < 20:
                        gt_state = [np.array(gt[key]) for key in ["x_position","y_position"]]
                        gt_state.append( np.ones([len(gt["timestamp"])])*gt["length"]  )
                        gt_state.append( np.ones([len(gt["timestamp"])])*gt["width"]  )
                        gt_state.append( np.ones([len(gt["timestamp"])])*gt["height"]  )
                        gt_state = np.stack(gt_state).transpose(1,0)
                        
                        gt_time =  np.array(gt["timestamp"]) - pred_times[p_idx,0]
                        
                        relevant_gts.append([gt_time,gt_state,gt["_id"]])
            
            
        # get normalizing speed  for x position
        rise = posterior[-1,0] - posterior[0,0]
        run = state_times[-1] - state_times[0]
        slope = rise/run
        intercept = state[0,0]
        
        if False:
            slope = 0
            intercept = 0
        
        state[:,0] = state[:,0] - state_times_unshifted * slope - intercept
        measurement[:,0] =  measurement[:,0] - times*slope - intercept
        
        # covariance windows
        cov_1plus  =     np.sqrt(covariance) + state
        cov_1minus =    -np.sqrt(covariance) + state
        cov_3plus  =   3*np.sqrt(covariance) + state
        cov_3minus =  -3*np.sqrt(covariance) + state
        
        
        
        
                    
                    
                    # if max(gt["starting_x"],gt["ending_x"]) > min(posterior[:,0]) and  min(gt["starting_x"],gt["ending_x"]) < max(posterior[:,0]):
                    #     if max(gt["y_position"]) > min(posterior[:,1]) and  min(gt["y_position"]) < max(posterior[:,1]):
                            
        
        n_plots = len(posterior[0])
        scale = 5
        fig = plt.figure(figsize =(5*scale,n_plots*scale))
        gs = (grid_spec.GridSpec(n_plots,1))
        y_labels = ["speed-norm X (ft)","Y (ft)", "L (ft)", "W (ft)", "H (ft)", "V (ft/s)"]
        #creating empty list
        ax_objs = []
        
        
        gt_colors = np.random.rand(len(relevant_gts),3) * 0.5
        gt_colors[:,1] += 0.5
        
        
        
        
        
        for didx in range(n_plots):
            ax_objs.append(fig.add_subplot(gs[didx:didx+1, 0:]))

                        

            # plot covariance pipes (1 and 3 std devs)
            ax_objs[-1].fill_between(state_times,cov_3plus[:,didx],cov_3minus[:,didx],alpha = 0.1,color = (0.1,0.2,0.8))
            ax_objs[-1].fill_between(state_times,cov_1plus[:,didx],cov_1minus[:,didx],alpha = 0.1,color = (0.1,0.2,0.8))

            # find and plot all relevant gts
            if didx < len(measurement[0]):
                for g,plot_gt in enumerate(relevant_gts):
                    y_val = plot_gt[1][:,didx]
                    if didx == 0:
                        y_val = y_val - plot_gt[0]*slope - intercept
                    
                    plt.plot(plot_gt[0],y_val, ls = "--" , color = gt_colors[g])            

            # plot state (prior and posterior)
            ax_objs[-1].plot(state_times,state[:,didx],color = (0,0.2,0.8),linewidth = 2)
            ax_objs[-1].scatter(state_times[::2],state[::2,didx],color = (0,0.2,0.8))
            ax_objs[-1].scatter(state_times[1::2],state[1::2,didx],edgecolor = (0,0.2,0.8),s =100,facecolor = "none")

            
            if didx < len(measurement[0]):
                # plot measurements
                ax_objs[-1].scatter(times,measurement[:,didx],color = (1,0,0), marker = "x")
                for m_idx in range(len(measurement)):   
                    ax_objs[-1].text(times[m_idx],measurement[m_idx,didx],"{:.2f}".format(meas_conf[m_idx]),va = "bottom", ha = "center",fontsize = 3*scale)
                    
                    # plot meas cov windows
                    ymin = measurement[m_idx,didx] -meas_cov[didx].item()
                    ymax = measurement[m_idx,didx] +meas_cov[didx].item()
                    ax_objs[-1].vlines(x = times[m_idx], ymin=ymin, ymax=ymax,color = (1,0,0), linestyles = "dotted")
                    ymin = measurement[m_idx,didx] -3*meas_cov[didx].item()
                    ymax = measurement[m_idx,didx] +3*meas_cov[didx].item()
                    ax_objs[-1].vlines(x = times[m_idx], ymin=ymin, ymax=ymax,color = (1,0,0), alpha = 0.2, linestyles = "dotted")
                
                
        
        
        
        
        
            # formatting
        
                # setting uniform x and y lims
            #     xr[0] = max(xr[0],xrange_clip[didx][0])
            #     xr[1] = min(xr[1],xrange_clip[didx][1])
        
            ax_objs[-1].set_xlim([0,pred_times[p_idx,1] - pred_times[p_idx,0]])
            ax_objs[-1].set_ylim(min(cov_1minus[:,didx]),max(cov_1plus[:,didx]))
            
            #     # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)
                
            # # remove borders, axis ticks, and labels
            # ax_objs[-1].set_yticklabels([])
            ax_objs[-1].tick_params(axis='y', labelsize=3*scale, length = 3*scale)
            ax_objs[-1].set_ylabel("{}".format(y_labels[didx]),fontsize = 5*scale)
                
            if didx == n_plots - 1:
                ax_objs[-1].tick_params(axis='x', labelsize=3*scale, length = 3*scale)
                ax_objs[-1].set_xlabel("Time (s)",fontsize = 5*scale)
            else:
                ax_objs[-1].set_xticklabels([])
                ax_objs[-1].tick_params(axis='x', length = 0 )
        
                
            spines = []
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)
            #     ax_objs[-1].text(xr[0],max_val/2,MD[to_plot[didx]]["text"],fontweight="bold",fontsize = 1500/scale,va="bottom")
            #     #ax_objs[-1].axvline(x = 0, ymax = 0.8, linestyle = ":",color = (0.2,0.2,0.2))
            
            
        #     # get mean,MAE, stddev, max
        #     mean = np.mean(data[didx])
        #     MA  = np.mean(np.abs(data[didx]))
        #     stddev = np.std(data[didx])
        #     maxx = np.max(np.abs(data[didx]))
            
        #     xv = xr[0] #+  0.2*(xr[1] - xr[0])
        #     ax_objs[-1].text(xv,max_val/2,               "Mean:     {:.1f}{}".format(mean,units[didx]),fontsize = 1000/scale,va="top")
        #     ax_objs[-1].text(xv,max_val/2-(0.05*max_val),"Mean Abs: {:.1f}{}".format(MA,units[didx]),fontsize = 1000/scale,va="top")
        #     ax_objs[-1].text(xv,max_val/2-(0.1*max_val), "Stdev:    {:.1f}{}".format(stddev,units[didx]),fontsize = 1000/scale,va="top")
        #     ax_objs[-1].text(xv,max_val/2-(0.15*max_val),"Max:      {:.1f}{}".format(maxx,units[didx]),fontsize = 1000/scale,va="top")
            
        #     if len(results) > 1:
        #         mean = np.mean(data2[didx])
        #         MA  = np.mean(np.abs(data2[didx]))
        #         stddev = np.std(data2[didx])
        #         maxx = np.max(np.abs(data2[didx]))
                
        #         xv = xr[0] + 0.16*(xr[1] - xr[0])
        #         # may need a black border
                
        #         ax_objs[-1].text(xv,max_val/2,     " ({:.1f}{})".format(mean,units[didx]),fontsize = 1000/scale,va="top",ha="left", color=color_pallette[1]/255,bbox=dict(facecolor="white",edgecolor="white",alpha=0.7))
        #         ax_objs[-1].text(xv,max_val/2-(0.05*max_val)," ({:.1f}{})".format(MA,units[didx]),fontsize = 1000/scale,va="top",ha="left", color= color_pallette[1]/255,bbox=dict(facecolor="white",edgecolor="white",alpha=0.7))
        #         ax_objs[-1].text(xv,max_val/2-(0.1*max_val), " ({:.1f}{})".format(stddev,units[didx]),fontsize = 1000/scale,va="top",ha="left", color= color_pallette[1]/255,bbox=dict(facecolor="white",edgecolor="white",alpha=0.7))
        #         ax_objs[-1].text(xv,max_val/2-(0.15*max_val)," ({:.1f}{})".format(maxx,units[didx]),fontsize = 1000/scale,va="top",ha="left", color= color_pallette[1]/255,bbox=dict(facecolor="white",edgecolor="white",alpha=0.7))
    
        
        gs.update(hspace= 0.02)
        
        duration = run
        dist_duration = np.abs(rise)
        direct = 'WB' if pred["direction"] == -1 else "EB"
        fig.text(0.1,0.93,"{} {} (idx {}), {}".format(classmap[pred["coarse_vehicle_class"]],pred["local_fragment_id"],p_idx,direct),fontsize = 10*scale,ha = "left",va = "bottom")
        fig.text(0.1,0.92,"Duration: {:.1f}s, {:.1f} ft.  Cause of Death: {}".format(duration,dist_duration,pred["flags"][0]),fontsize = 6*scale,ha = "left",va = "top")
        plt.tight_layout()
        
        fig.savefig("./data/filter_plots/{}_{}_{}_{}.png".format(pred_collection,p_idx,classmap[pred["coarse_vehicle_class"]],pred["local_fragment_id"]))
        #fig.show()

    # 

if __name__ == "__main__":
    db_param = {
          #"default_host": "10.2.218.56",
          #"default_port": 27017,
          "host":"10.2.218.56",
          "port":27017,
          "username":"i24-data",
          "password":"mongodb@i24",
          #"default_username": "i24-data",
          #"readonly_user":"i24-data",
          #"default_password": "mongodb@i24",
          "database_name": "trajectories",      
          "server_id": 1,
          "session_config_id": 1,
          #"trajectory_database":"trajectories",
          #"timestamp_database":"transformed"
          }
    r = evaluate(db_param)




# 7. Print out summary
