import numpy as np
import json


def load_file(file_name, online = False):
    if not online:
        file_path = "../data/training_data/mixed_samples/"
        loaded_file = np.load(file_path + file_name)
        print("File " + file_name + " was loaded successfully")
    else:
        file_path = "../data/online_data/avg_df/"
        loaded_file = np.load(file_path + file_name)
        print("File " + file_name + " was loaded successfully")
    return loaded_file

def load_patient_id_list(healthy = True, male = True):
    if healthy:
        name = 'healthy_'
    else:
        name = 'pMI_'

    if male:
        name += 'male'
    else:
        name+= 'female'

    name+= '_id.json'

    with open("../data/online_data/"+name) as f:
        data = json.load(f)

    return data