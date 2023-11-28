# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 09:35:39 2019

@author: WFN2
"""

train_datafolder = '..//data//feijen_1000//'

test_filenames = [ 'R1_10_1.TXT',
                    'R1_10_2.TXT',
                    'R1_10_3.TXT',
                    'R1_10_4.TXT',
                    'R1_10_5.TXT',
                    'R1_10_6.TXT',
                    'R1_10_7.TXT',
                    'R1_10_8.TXT',
                    'R1_10_9.TXT',
                    'R1_10_10.TXT',
                  ]

conversion_dict = { 'r1_10_1':'100_10',
                    'r1_10_2':'75_10',
                    'r1_10_3':'50_10',
                    'r1_10_4':'25_10',
                    'r1_10_5':'100_30',
                    'r1_10_6':'75_30',
                    'r1_10_7':'50_30',
                    'r1_10_8':'25_30',
                    'r1_10_9':'100_norm60-20',
                    'r1_10_10':'100_norm120-30',
                  }


import data_structure

datafolder = '..//data//homberger_1000_customer_instances//'
accept_missing__nr_vehicles_optimal = True

def read_data(filename, restrict_nr_vehicles):
    '''Reads an instance from a filename. Returns the instance'''
    
    fw = open( filename, 'r' )
    lines = [line.rstrip('\n') for line in fw]
    

    name = lines[0]
    customers = []
    l = 1
    while l < len(lines):
        line = lines[l].split()
        if(len(line) < 1):
            l+=1
            continue
        if line[0] == "VEHICLE":
            l+=2
            line = lines[l].split()
            nr_vehicles = int(line[0])
            capacity = int(line[1])
            if not accept_missing__nr_vehicles_optimal and restrict_nr_vehicles:
                assert len(line) > 2, "Nr of vehicles in world record not given. Change instance."
                nr_vehicles = int(line[2])
        if line[0] == "CUSTOMER":
            l+=3
            break
        l += 1
    while l < len(lines):
        line = lines[l].split()
        cust = data_structure.customer( int(line[0]),
                                        int(line[1]),
                                        int(line[2]),
                                        int(line[3]),
                                        int(line[4]),
                                        int(line[5]),
                                        int(line[6]) )
        if cust.nr == 0:
            depot = cust
        else:
            customers += [cust]
        l += 1
    fw.close()
    inst = data_structure.instance(name, depot, customers, nr_vehicles, capacity, restrict_nr_vehicles)
    return inst


def get_all_instance_names(basic_instance_name, nr_of_instances, tw_lengths, fraction_tws, normal_tw_lengths_parameters):
    '''Return all the names of the training instances with the given parameters.'''

    result = []
    
    for i in range(nr_of_instances):
        for tw_length in tw_lengths:
            for fraction_tw in fraction_tws:
                instance_name = basic_instance_name + f"_{int(fraction_tw*100)}_{tw_length}_{i+1}.txt"  
                result.append(instance_name)
        for avg,std in normal_tw_lengths_parameters:
            instance_name = basic_instance_name + f"_{100}_norm{avg}-{std}_{i+1}.txt"
            result.append(instance_name)
    return result

def all_train_instance_names():
    '''Return all the names of the training instances with the specific parameters that the training instances were created with. See paper for more details '''
    
    basic_instance_filename = "r1_10_1"
    nr_of_instances = 10
    tw_lengths = [10,30]
    fraction_tws = [1.0, 0.75, 0.5, 0.25]
    normal_tw_lengths_parameters = [(60,20), (120,30)]
    instance_names = get_all_instance_names(basic_instance_filename, nr_of_instances, tw_lengths, fraction_tws, normal_tw_lengths_parameters)
    return instance_names

def ML_model_string(instance_name):
    '''Checks the name of the instance, and returns the string necessary for reading the ML model that corresponds to this instance'''
    
    name_split = instance_name.split("_")
    if len(name_split) == 3: #Standard instance, e.g. r1_10_1, or r1_10_10
        return conversion_dict[instance_name]
    elif len(name_split) == 6 and 'r1_10_1' in instance_name: #data colleciton instance, eg. r1_10_1_100_norm60-20_1
        return name_split[3] + "_" + name_split[4]
    else:
        print("ERROR: DON'T KNOW WHICH ML MODEL TO READ FOR INSTANCE", instance_name)
        assert False
        
