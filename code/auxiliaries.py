# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 13:56:54 2023

@author: wille
"""

import numpy as np
from datetime import date, datetime
import joblib

import iteration_operators


def print_result(result, filenames):
    '''Prints the result which is created in the main function for the given filenames'''

    with open(f'../testruns/test_{date.today()}.txt', 'a') as resultFile:
        
        for filename in filenames:
            msg = filename + "\t"
            
            times, best_sols, nr_iterations = result[filename]
            avg_score = np.mean([sol.total_value for sol in best_sols])
            avg_nr_vehicles = np.mean([sol.non_empty_vehicles for sol in best_sols])
            msg += f"{avg_score:.2f}\t"
            msg += f"{np.mean(times):.2f}s=("
            for runtime in times:
                msg += f"{runtime:.2f}s-"
            msg += ")\t"
            msg += f"{np.mean(nr_iterations):.2f}it=("
            for nr_iteration in nr_iterations:
                msg += f"{nr_iteration}it-"
            msg += ")\t"
            msg += f"{avg_nr_vehicles:.2f}\t("
            
            for sol in best_sols:
                msg += f'{sol.total_value:.2f}\t'
                msg += f' {sol.non_empty_vehicles:>3}\t)\t'
                
            print(msg)
            resultFile.write(msg + "\n")
            


def write_data_to_file(test_name, track_data, instance_filename):
    '''We save the track_data in a .sav file.
    '''
    
    datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    track_filename = f'../testruns/track_{test_name}_' + instance_filename.lower().replace('.txt',f'_{datetime_string}.sav')
    joblib.dump(track_data, track_filename)
    
            
def write_features_to_file(test_name, instance_name, feature_msg):
    '''The given feature_msg is written to the feature file belonging to the given instance_name'''
    
    today = date.today()
    feature_file = open(f'..//features//features-{instance_name}-{today}-{test_name}.txt', 'a')
    header_line = iteration_operators.feature_header_line
    feature_file.write(header_line)
    feature_file.write(feature_msg)
    feature_file.close()


def print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg, nr_it_no_improv):
    print("%7.2fs" %it_end_time +  " it " +  f'{it:>4}:\t' +
      f'{cur_sol.total_value:9.2f}\t' +
      f'{cur_sol.total_distance:9.2f}{sign}\t' +  
      f'{cur_sol.total_tw_violation:9.2f}' +  
      f' {cur_sol.nr_tw_violations:>3}'+
      f'\t{dist_delta:6.2f}' + f'\t{violation_delta:9.2f}'  + 
      f" ({cur_sol.non_empty_vehicles}/{cur_sol.instance.nr_vehicles})",
      "\t", msg,
      nr_it_no_improv)