import sys
import time
import data_structure
import data_reader
import iteration_operators
import joblib
import auxiliaries
from datetime import datetime
import options


def main(filenames, strategy): 
    #filenames, datafolder, num_nhoods, init_file_name):
    '''Run the instances in the given filenames with the given strategy following the given other options. 
    Return a dictionary with the results.''' 
        
    
    result = {}
    
    for instance_filename in filenames:
        

        times = []
        best_sols = []
        nr_iterations = []
        
        instance = data_reader.read_data(data_reader.datafolder + instance_filename, True)
            
        if strategy in ['ML'] or (strategy == 'data_collection' 
                                 and options.datacollection_strategy == "ML"):
            model_name = options.model_name_prefix + data_reader.ML_model_string(instance.name) + options.model_name_suffix
            print("reading model:" , model_name)
            iteration_operators.model = joblib.load(model_name)
            iteration_operators.indices_features_used_by_model = list(range(1,1 + iteration_operators.model.steps[1][1].n_features_in_))  #In this model, the iteration number is not a feature

        for i in range(options.nr_of_times): 
                
            tic = time.time()
            print('Start with iteration', i+1,'/',options.nr_of_times, 'at time', datetime.now().strftime("%H:%M:%S"), instance_filename)
            best_sol, track_data, feature_msg = solve(instance, strategy) # num_nhoods, init_file_name, strategy)
            toc = time.time()
            print("Time taken for %s: %.4f" % ( instance_filename, toc-tic ) )
            
            times += [toc-tic]
            best_sols += [best_sol]
            nr_iterations_in_run = track_data[0][-1]
            nr_iterations += [nr_iterations_in_run]
            if options.keep_track_of_data:
                auxiliaries.write_data_to_file(options.test_name, track_data, instance_filename)

            if strategy == 'data_collection':
                auxiliaries.write_features_to_file(options.test_name, instance.name, feature_msg)
            
        result[instance_filename] = [times, best_sols, nr_iterations]
        
    auxiliaries.print_result(result, filenames)
    return result
                
                
def accept(temp_sol, best_sol):
    '''Accept the temp_sol if it is better than the best_sol'''
    return temp_sol.is_better_than(best_sol)
    
def initialization(instance, init_file_name):
    '''Create an initial solution for the given instance from the given init_file_name'''
    init_sol = data_structure.solution.from_vroom_json_file(instance, init_file_name.replace("INSTANCENAME", instance.name))

    #Print info about initial solution
    msg = "init dist: %.2f" %init_sol.total_distance
    msg += " viol: %.2f" %init_sol.total_tw_violation
    msg += f" nr viol: {init_sol.nr_tw_violations}"
    msg += " sum: %.2f" %init_sol.total_value
        
    print(msg)
    
    return init_sol
    
def solve(instance, strategy): #num_nhoods, init_file_name, strategy):
    '''Solves the given instance with the given strategy, following the options given in the options file'''
    global nr_to_delete, a
    
    init_sol = initialization(instance, options.init_file_name)
    best_sol = init_sol.get_copy(False)
    cur_sol = init_sol.get_copy(False)
    
    tic = time.time()
    
    it_end_time = 0
    
    nr_it_no_improv = 0
    
    track_iterations = [0]
    track_solution_values = [init_sol.total_value]
    track_times = [0]
    track_nr_vehicles = [len(init_sol.vehicles)]
    track_improvements = []
    track_predictions = []
    track_indices = []
    
    
    if strategy == 'data_collection':
        feature_msg = ""
    else:
        feature_msg = None

    it = 0
    
    while it < options.nr_iterations or it_end_time < options.running_time:
        
        
        temp_sol = cur_sol.get_copy(False)
        temp_sol, msg, feature_string = iteration_operators.iteration_solver(temp_sol,options.nhood_size, it, strategy, options.num_nhoods)
        

        if strategy == 'data_collection':
            feature_msg += feature_string
                
        dist_delta = 0
        sign = " "
        if(temp_sol.is_better_than(best_sol)):
            best_sol = temp_sol.get_copy(False)
            sign = "*"
        accept_temp = accept(temp_sol, cur_sol )
        dist_delta = cur_sol.total_distance - temp_sol.total_distance
        violation_delta = 0
        violation_delta = cur_sol.total_tw_violation - temp_sol.total_tw_violation
            
            
        if accept_temp:
            cur_sol = temp_sol.get_copy(False)
            if dist_delta <=0:
                nr_it_no_improv +=1
            else:
                nr_it_no_improv = 0
            sign += "^" if dist_delta < 0 else ( "-" if dist_delta == 0 else "v")
        else:
            nr_it_no_improv += 1
            

        toc = time.time()
        it_end_time = toc - tic
        
        
        track_iterations.append(it+1)
        track_solution_values.append(cur_sol.total_value)
        track_times.append(it_end_time)
        track_nr_vehicles.append(cur_sol.non_empty_vehicles)

        if options.show_all_iterations or not dist_delta == 0 or not violation_delta == 0:
            if cur_sol.total_value - temp_sol.total_value >= 0:
                auxiliaries.print_iteration_info(it_end_time, it, cur_sol, sign, dist_delta, violation_delta, msg, nr_it_no_improv)
        
            
        it += 1
        
    msg = "best dist: %.2f" %best_sol.total_distance
    msg += " viol: %.2f" %best_sol.total_tw_violation
    print(msg)
     
    track_data = [track_iterations, track_times, track_solution_values, track_nr_vehicles, track_improvements, track_predictions, track_indices]
    
    return best_sol, track_data, feature_msg



if __name__ == "__main__":
    args = sys.argv[1:]
    filenames, strategy = options.parse_options(args)
    result = main(filenames, strategy) #filenames, data_reader.datafolder, num_nhoods, init_file_name)