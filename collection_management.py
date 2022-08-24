
from i24_database_api import DBClient
import warnings
warnings.filterwarnings("ignore")

class MongoCollectionManager:
    def __init__(self,host = None,port = None ,username = None ,password = None, database = None):
        self.client = DBClient(host,port,username,password,database_name = database)
        self.meta = "__METADATA__"
        
        
        if self.meta in self.client.list_collection_names():
            print("Metadata Collection Exists")
        else:
            print("WARNING: No Metadata Collection Exists")
        
    def _create_metadata_collection(self):
        self.client.db["__METADATA__"]
        meta = {"saved_collection_names":[self.meta]}
        self.client.db[self.meta].insert_one(meta)
        
    def _delete_metadata_collection(self):
        self.client.delete_collections([self.meta])
        
    def list_saved_collections(self):
        saved = self.get_saved_collections()    
        print("Saved Collections: ")
        [print(item) for item in saved]
    
    def list_all_collections(self):
        lis = self.get_all_collections()
        print("Existing Collections: ")
        [print(item) for item in lis]

    def get_all_collections(self):
        lis = self.client.list_collection_names()
        lis.sort()
        return lis

    def get_saved_collections(self):
        doc = list(self.client.db[self.meta].find())[0]
        saved = doc["saved_collection_names"]
        return saved
    
    def save_collections(self,collection_list):
        doc = list(self.client.db[self.meta].find())[0]
        saved = doc["saved_collection_names"]
        _id   = doc["_id"]
        
        for item in collection_list:
            if item not in saved and item in self.client.list_collection_names():
                saved.append(item)
                
        self.client.db[self.meta].update_one({'_id':_id},{'$set':{'saved_collection_names':saved}},upsert=True)
        self.list_saved_collections()
        
    def unsave_collections(self,collection_list):
        doc = list(self.client.db[self.meta].find())[0]
        saved = doc["saved_collection_names"]
        _id   = doc["_id"]
        
        saved_new = []
        for item in saved:
            if item not in collection_list:
                saved_new.append(item)
       
        self.client.db[self.meta].update_one({'_id':_id},{'$set':{'saved_collection_names':saved_new}},upsert=True)
        self.list_saved_collections()
        
    def cleanup(self):
        # list to remove
        all_collections = self.client.list_collection_names()
        saved = self.get_saved_collections()
        
        to_remove = []
        for item in all_collections:
            if item not in saved:
                to_remove.append(item)
        print("The following collections will be removed:")
        [print(item) for item in to_remove]

        inp = input("Do you want to continue? N/y")
        if inp == "Y" or inp == "y":
            self.client.delete_collections(to_remove)

    

def save_reconciled_descendants(db_params,parent_db,child_db):
    
    db_param["database"] = parent_db
    mcm1 = MongoCollectionManager(**db_param.copy())
    
    
    db_param["database"] = child_db
    mcm2 = MongoCollectionManager(**db_param.copy())


    parent_saved = mcm1.get_saved_collections()
    
    child_all = mcm2.get_all_collections()
    
    child_save = []
    for coll in child_all:
        try:
            base = coll.split("__")[0]
            if base in parent_saved:
                child_save.append(coll)
        except:
            pass
    mcm2.save_collections(child_save)
    

if __name__ == "__main__":
    db_name = "trajectories"
    
    db_param = {
          "host":"10.80.4.91",
          "port":27017,
          "username": "mongo-admin",
          "password": "i24-data-access",
          "database": db_name      
          }
    mcm = MongoCollectionManager(**db_param)
    mcm.list_saved_collections()
    
    if False:
        save_reconciled_descendants(db_param,"trajectories","reconciled")
        
    