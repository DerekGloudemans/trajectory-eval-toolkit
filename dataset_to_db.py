from i24_database_api import DBClient
from i24_configparse import parse_cfg

import os
import numpy as np
import _pickle as pickle

class WriteWrapper():
    
    def __init__(self,collection_name,server_id = -1):
    
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
    
        self.SESSION_CONFIG_ID = os.environ["TRACK_CONFIG_SECTION"]
        self.PID = os.getpid()
        self.COMPUTE_NODE_ID = server_id 
        
        self = parse_cfg("TRACK_CONFIG_SECTION",obj=self)    
    
        
        # self.dbw = DBWriter(
        #                host               = self.host, 
        #                port               = self.port, 
        #                username           = self.username, 
        #                password           = self.password,
        #                database_name      = self.db_name, 
        #                schema_file        = self.schema_file,
        #                collection_name    = collection_name,
        #                server_id          = self.COMPUTE_NODE_ID, 
        #                session_config_id  = self.SESSION_CONFIG_ID,
        #                process_id         = self.PID,
        #                process_name       = "groundtruth"
        #                )
        
        self.dbw = DBClient(**db_param,collection_name = collection_name)
            
        self.prev_len = len(self) -1
        self.prev_doc = None
    
        self.class_dict = { "sedan":0,
                    "midsize":1,
                    "van":2,
                    "pickup":3,
                    "semi":4,
                    "truck (other)":5,
                    "truck": 5,
                    "motorcycle":6,
                    "trailer":7,
                    "bus":5,
                    0:"sedan",
                    1:"midsize",
                    2:"van",
                    3:"pickup",
                    4:"semi",
                    5:"truck (other)",
                    6:"motorcycle",
                    7:"trailer"
                    }
    
    def __len__(self):
        return self.dbw.collection.count_documents({})
    
    
    def insert(self,trajectories):
        """
        Converts trajectories as dequeued from TrackState into document form and inserts with dbw
        trajectories - output from TrackState.remove()
        """
        
        if len(trajectories) == 0:
            return
        
        # cur_len = len(self)
        # if cur_len == self.prev_len:
        #     logger.warning("\n Document {} was not correctly inserted into database collection {}".format(self.prev_doc,self.raw_collection))
        # self.prev_len = cur_len
        
        for trajectory in trajectories:
            timestamps = trajectory["timestamp"]
            x = trajectory["x_position"]
            y = trajectory["y_position"]
            w = trajectory["width"]
            l = trajectory["length"]
            h = trajectory["height"]
            try:
                cls = self.class_dict[trajectory["class"]]
            except:
                cls = 1
            tid = trajectory["id"]
            
            try:
                w = w.item()
            except:
                pass
            
            try:
                l = l.item()
            except:
                pass
            
            try:
                h = h.item()
            except:
                pass
            
            xnew = []
            for it in x:
                try:
                    xnew.append(it.item())
                except:
                    xnew.append(it)
            x = xnew
            
            ynew = []
            for it in y:
                try:
                    ynew.append(it.item())
                except:
                    ynew.append(it)
            y = ynew
            
            
            # convert to document form
            doc = {}
            doc["configuration_id"]        = self.SESSION_CONFIG_ID
            doc["local_fragment_id"]       = tid
            doc["compute_node_id"]         = self.COMPUTE_NODE_ID
            doc["coarse_vehicle_class"]    = cls
            doc["fine_vehicle_class"]      = -1
            doc["timestamp"]               = timestamps
            doc["raw timestamp"]           = timestamps
            doc["first_timestamp"]         = timestamps[0]
            doc["last_timestamp"]          = timestamps[-1]
            doc["road_segment_ids"]        = [-1]
            doc["x_position"]              = x
            doc["y_position"]              = y
            doc["starting_x"]              = x[0]
            doc["ending_x"]                = x[-1]
            doc["camera_snapshots"]        = "None"
            doc["flags"]                   = ["test flag 1","test flag 2"]
            doc["length"]                  = [float(l)]
            doc["width"]                   = [float(w)]
            doc["height"]                  = [float(h)]
            doc["direction"]               = -1 if y[0] > 60 else 1
            
            
            
            
            # insert
            if len(x) > self.min_document_length:
                self.dbw.write_one_trajectory(**doc) 
                
    
            
if __name__ == "__main__":
    data_file = "./data/gt_data/linear_spacing_splines_4.cpkl"

    # 1. Create writer object
    ww = WriteWrapper("groundtruth_scene_3")
    
    # 2. Load data pickle
    with open(data_file,"rb") as f:
        [data,timestamps,splines] = pickle.load(f)
        
    # 3. Roll data up by object, replacing with corrected timestamps as necessary
    
    objects = []
    
    """ now we have a few options here. Obviously we don't want the GT trajectories to be jittery, so the best call is 
     probably to sample x and y positions from splines when available. This leaves 2 open questions:
         1. what do we do for objects that don't have splines (too short) - we'll
         implement linear interpolation to the desired timestamps in this case
         2. at what times do we sample objects - a set sampling rate is probably preferable
         and whatever times we pick, we'll have to do some interpolation on the eval end
         because each object predicted position may be at a slightly different time than the gt
         position. For now, we'll sample each position at 30 Hz over the range of times for which that object 
         has annotated positions"""
         
    
    
    for o_idx in range(len(splines)):
        print("Raveling object {}".format(o_idx))
        obj_container = {
            "timestamp":[],
            "x_position":[],
            "y_position":[]
            }
        
        #get min and max time
        min_ts = np.inf
        max_ts = -np.inf
        
        set_statics = True
        
        # get the extents of the time range for which this object is visible
        for datum in data:
            
            if set_statics:
                obj_container["length"] = datum
            
            for obj in datum.keys():
                obj_id = int(obj.split("_")[-1])
                if obj_id == o_idx:
                    
                    if set_statics:
                        obj_container["length"] = datum[obj]["l"]
                        obj_container["width"] = datum[obj]["w"]
                        obj_container["height"] = datum[obj]["h"]
                        obj_container["class"] = datum[obj]["class"]
                        obj_container["id"] = datum[obj]["id"]

                        set_statics = False
                    
                    if datum[obj]["timestamp"] < min_ts:
                        min_ts = datum[obj]["timestamp"]
                    if datum[obj]["timestamp"] > max_ts:
                        max_ts = datum[obj]["timestamp"]
        
        
                
        #sample spline every 30th of second between min and max time
        for ts in np.arange(min_ts,max_ts,1/30.0):
            if splines[o_idx][0] is not None:
                obj_container["timestamp"].append(ts)
                obj_container["x_position"].append(splines[o_idx][0](ts).item())
                obj_container["y_position"].append(splines[o_idx][1](ts).item())                                               
            else:
                # if no spline exists, find closest previous position and store that
                DONE = False
                for datum in data:
                    if not DONE:
                        for obj in datum.keys():
                            obj_id = int(obj.split("_")[-1])
                            if obj_id == o_idx:
                                if datum[obj]["timestamp"] <= ts and np.abs(ts-datum[obj]["timestamp"]) < 1/30:
                                    obj_container["timestamp"].append(datum[obj]["timestamp"])
                                    obj_container["x_position"].append(datum[obj]["x"])
                                    obj_container["y_position"].append(datum[obj]["y"])         
                                    DONE = True
                                    break
                                
        objects.append(obj_container)
        
        # 4. Write rolled object to database collection
        if len(obj_container["timestamp"]) > 0:
            ww.insert([obj_container])
        
    print("Finished inserting objects into database collection: {}".format(ww.dbw.collection.name))
    