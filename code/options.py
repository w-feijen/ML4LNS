# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:54:49 2023

@author: wille
"""

import random
import numpy as np
from datetime import datetime

import data_reader
import iteration_operators

import warnings
#from sklearn.exceptions import UserWarning
warnings.filterwarnings(action='ignore', category=UserWarning)



def set_options(strategy):
    global test_name, nr_of_times, nr_iterations, running_time, show_all_iterations, keep_track_of_data, seed, nhood_size, num_nhoods, ml_mode, k, oracle_mode, init_file_name

    #General test options
    test_name       = 'default'
    nr_of_times     = 10               #Nr of times an instance should be run
    nr_iterations   = 500              #Minimum of iterations in a run.
    running_time    = 1                #Minimum seconds run should take
    
    #Logging and tracking options
    show_all_iterations = False             #If false, only improving iterations are shown
    keep_track_of_data  = True              #Tracks and saves some data each iteration
    
    #Technical options
    seed        = 42
    random.seed(seed)
    np.random.seed(random.randint(0,2**32-1))
    
    #Neighborhood Selection options - General
    nhood_size  = 10                 #Nr routes in a nhood
    num_nhoods  = 10                 #How many neighborhoods must be computed in each iteration. The strategy decides which is chosen.
    ml_mode = 'highest'              #After computing prediction, what strategy is used to decide the winner. Options: 'ksample_rnd',#'ksample_prop', #'ksample_rnd','highest', #'highest', 'sample'
    k = 1
    oracle_mode = 'best'            #After computing all neighborhood improvements, what strategy is used to decide the best: 'random', 'best', 'proportional'

    iteration_operators.P_randomness = 10 if strategy == 'data_collection' else 40
    iteration_operators.update_feature_header_line = strategy == 'data_collection' #At start of data collection we need to update the header line

    #Filenames to be read - according to strategy
    if strategy in ['best', 'random', 'ML']:
        filenames = data_reader.test_filenames
        init_file_name = "..//data//solutions//init//INSTANCENAME_sol.json"
    else:
        filenames = data_reader.all_train_instance_names()
        data_reader.datafolder = data_reader.train_datafolder
        init_file_name = "..//data//solutions//init//INSTANCENAME_sol.json"
        
    return filenames
    
def find_strategy(args):
    if len(args) < 1:
        print("No strategy is given so we follow ML strategy")
        strategy = "ML"
    elif args[0] == "1":
        strategy = "best"
    elif args[0] == "2":
        strategy = "random"
    elif args[0] == "3":
        strategy = "ML"
    elif args[0] == "4":
        strategy = "data_collection"
    else:
        raise Exception("Error, given strategy: " + args[0] + " unknown. Should be one of [1,2,3,4]")
        
    global model_name_prefix, model_name_suffix, datacollection_strategy
    if strategy in ['ML']:
        # global model_name_prefix, model_name_suffix
        if len(args) < 2:
            print("No ML model specification is given so we take ML5") 
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-13-instanceslikeR1_"
            model_name_suffix = "_ML5explore10-standard_rf2_balanced.sav"
        elif args[1] == "1":
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-07-instanceslikeR1_"
            model_name_suffix = "_ML1explore10-standard_rf2_balanced.sav"
        elif args[1] == "2":
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-09-instanceslikeR1_"
            model_name_suffix = "_ML2explore10-standard_rf2_balanced.sav"
        elif args[1] == "3":
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-11-instanceslikeR1_"
            model_name_suffix = "_ML3explore10-standard_rf2_balanced.sav"
        elif args[1] == "4":
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-12-instanceslikeR1_"
            model_name_suffix = "_ML4explore10-standard_rf2_balanced.sav"
        elif args[1] == "5":
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-13-instanceslikeR1_"
            model_name_suffix = "_ML5explore10-standard_rf2_balanced.sav"
        else:
            raise Exception("Error, given ML model specification: " + args[1] + " unknown. Should be one of [1,2,3,4,5] for strategy ML")
    
    
    if strategy in ['data_collection']:
        if len(args) < 2:
            print("No ML model specification is given so we follow random strategy")
            datacollection_strategy = "random"
        elif args[1] == "0":
            datacollection_strategy = "random"
        elif args[1] == "1":
            datacollection_strategy = "ML"
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-07-instanceslikeR1_"
            model_name_suffix = "_ML1explore10-standard_rf2_balanced.sav"
        elif args[1] == "2":
            datacollection_strategy = "ML"
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-09-instanceslikeR1_"
            model_name_suffix = "_ML2explore10-standard_rf2_balanced.sav"
        elif args[1] == "3":
            datacollection_strategy = "ML"
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-11-instanceslikeR1_"
            model_name_suffix = "_ML3explore10-standard_rf2_balanced.sav"
        elif args[1] == "4":
            datacollection_strategy = "ML"
            model_name_prefix = "..//models//clf-2023-09-06-2023-09-12-instanceslikeR1_"
            model_name_suffix = "_ML4explore10-standard_rf2_balanced.sav"
        else:
            raise Exception("Error, given strategy: " + args[1] + " unknown. Should be one of [0,1,2,3,4] for strategy data_collection")
        
    return strategy

def print_options(filenames, strategy):
    
    info_string = 'start ' + datetime.today().strftime('%Y_%m_%d_%H%M') + '\n'
    info_string += f'Do test called "{test_name}", {nr_of_times} time(s) per instance\n'
    info_string += f'One run takes either {nr_iterations} iterations or {running_time} seconds \n'
    info_string += 'instances: ' + str(filenames) + '\n'
    info_string += 'tracking? ' + str(keep_track_of_data) + '\n'
    info_string += 'strategy:' + strategy + '\n'
    info_string += 'seed: ' + str(seed) + "\n"
    info_string += 'nhood_size: ' + str(nhood_size) + '\n'
    info_string += 'num_nhoods: ' + str(num_nhoods) + '\n'
    info_string += 'use_init_from_file: '+init_file_name + '\n'
    
    if strategy == ['ML']:
        info_string += 'model_name: ' + model_name_prefix + "{INSTANCENAME}" + model_name_suffix + '\n'
    if strategy == ['data_collection']:
        info_string += 'datacollection_strategy: ' + datacollection_strategy + '\n'
        if datacollection_strategy == "ML":
            info_string += 'model_name: ' + model_name_prefix + "{INSTANCENAME}" + model_name_suffix + '\n'
    
    print(info_string)

def parse_options(args):
    strategy = find_strategy(args)
    filenames = set_options(strategy)
    print_options(filenames, strategy)
    
    return filenames, strategy
    