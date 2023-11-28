import numpy as np
import math
from time import time
import vroom_operator
from datetime import datetime
import itertools
import options

global model, indices_features_used_by_model, feature_header_line, update_feature_header_line, P_randomness
model = None
indices_features_used_by_model = None

def make_prediction(features):
    '''Given a set of feature strings, 
    this method uses ML model to create prediction , 
    and: 
        - returns index of the best prediction (if ml_mode is 'highest')
        - samples a neighborhood based on probabilities (if ml_mode is 'sample')
        - samples with probabilities from the best k (if ml_mode is 'ksample_prop')
        - samples uniformly from the best k (if ml_mode is 'ksample_rnd')
    '''
    global model, indices_features_used_by_model
    
    ml_mode = options.ml_mode
    if ml_mode == 'ksample_prop' or ml_mode == 'ksample_rnd':
        k = options.k
    
    if indices_features_used_by_model is None:    
        features = np.array(features)
    else:
        features = np.array(features)[:, indices_features_used_by_model]

    predictions = model.predict_proba(features)[:,1]
    
    if ml_mode == 'sample':
        predictions_norm = predictions/sum(predictions)
        return_index = np.random.choice(list(range(len(features))), p=predictions_norm)    
    elif ml_mode == 'highest':
        return_index = np.argmax(predictions)
    elif ml_mode == 'ksample_prop':
        indices_k_best = np.argsort(predictions)[:k]
        predictions_k_best = [predictions[i] for i in indices_k_best]
        predictions_k_best = predictions_k_best/sum(predictions_k_best)
        return_index = np.random.choice(indices_k_best, p=predictions_k_best)
    elif ml_mode == 'ksample_rnd':
        indices_k_best = np.argsort(predictions)[:k]
        return_index = np.random.choice(indices_k_best)
    else:
        assert False, f'error, setting {ml_mode} unknown'
    
    return return_index, predictions


def add_max_min_avg(features, values, name):
    '''For the given list of features, return the maximum, minimum, average, standard deviation and sum. 
    Furthermore we check if the header line of the feature file needs to be updated and do so if necessary.''' 
    
    global feature_header_line, update_feature_header_line
    features += [max(values), min(values), np.average(values), np.std(values), np.sum(values)]
    
    if update_feature_header_line:
        feature_header_line += f'{name}_max\t{name}_min\t{name}_avg\t{name}_std\t{name}_sum\t'
    return features


def record_all_features(sol, anchors_vehicles, it_nr):
    '''For the given vehicles and given iteration number, calculate all features and return them. '''
    global feature_header_line, update_feature_header_line
        
    anchors = [c for c,_ in anchors_vehicles]
    vehicles_per_anchor = [v for _,v in anchors_vehicles]
    
    features = [it_nr, len(anchors)]
    
    if update_feature_header_line:       
        feature_header_line = 'date\tit_nr\tsize\t' 
        
    vehicles = [v for v in sol.vehicles if v in vehicles_per_anchor]
    deliveries = [v.get_delivery(c) for v, c in zip(vehicles_per_anchor, anchors)]
    
    
    #Features per customer
    lateness = tuple(d.lateness_value for d in deliveries)
    add_max_min_avg(features, lateness, 'lateness')
    waiting_time = tuple(d.waiting_time for d in deliveries)
    add_max_min_avg(features, waiting_time, 'wait_time')

    other_customers = [[c for c in anchors if not c in v.route] for v in vehicles_per_anchor]
    other_vehicles = [[v2 for v2 in vehicles if not v is v2] for v in vehicles_per_anchor]

    closeness = tuple(sol.get_closeness(c, False, vehicles) for c in anchors)
    add_max_min_avg(features, closeness, 'closeness')
    closeness2 = tuple(sol.get_closeness_v2(c, v, oc, False)  for v,c,oc in zip(vehicles_per_anchor, anchors, other_customers))
    add_max_min_avg(features, closeness2, 'closeness2')
    closeness2_temp = tuple(sol.get_closeness_v2(c, v, oc, True)  for v,c,oc in zip(vehicles_per_anchor, anchors,other_customers))
    add_max_min_avg(features, closeness2_temp, 'closeness2_temp')
    distance_cont = tuple(v.get_distance_contribution(c) for v, c in zip(vehicles_per_anchor, anchors))
    add_max_min_avg(features, distance_cont, 'dist_cont')
    tw_length = tuple(c.end - c.start for c in anchors)
    add_max_min_avg(features, tw_length, 'tw_length')
    distance_to_depot = tuple(sol.instance.distance(sol.instance.depot, c) for c in anchors)
    add_max_min_avg(features, distance_to_depot, 'dist_depot')
    load = tuple(c.demand for c in anchors)
    add_max_min_avg(features, load, 'demand')

    min_greedy_addition_cost_both = tuple(sol.get_min_greedy_addition_cost(c,ov,dc) for c,ov,dc in zip(anchors, other_vehicles, distance_cont))
    min_greedy_addition_cost = tuple(x for x,_ in min_greedy_addition_cost_both)
    max_gain_if_possible = tuple(y for _,y in min_greedy_addition_cost_both)
    add_max_min_avg(features, min_greedy_addition_cost, 'min_greedy_addition_cost')
    add_max_min_avg(features, max_gain_if_possible, 'max_gain_if_possible')
    
    possible_delay = tuple(d.possible_delay for d in deliveries)
    add_max_min_avg(features, possible_delay, 'possible_delay')
        
    #Features per vehicle 
    route_dist = tuple(v.distance for v in vehicles)
    add_max_min_avg(features, route_dist, 'route_dist')
    average_route_dist = tuple(v.get_average_route_distance() for v in vehicles)
    add_max_min_avg(features, average_route_dist, 'avg_rout_dist')
    empty_distance = tuple(v.get_empty_distance() for v in vehicles)
    add_max_min_avg(features, empty_distance, 'empty_dist')
    distance_worst_case_fraction = tuple(v.get_distance_worst_case_fraction() for v in vehicles)
    add_max_min_avg(features, distance_worst_case_fraction, 'dist_worst_case_frac')
    route_duration = tuple(v.get_route_duration() for v in vehicles)
    add_max_min_avg(features, route_duration, 'route_duration')
    average_route_duration = tuple(v.get_average_route_duration() for v in vehicles)
    add_max_min_avg(features, average_route_duration, 'avg_route_duration')
    idle_time = tuple(v.get_idle_time() for v in vehicles)
    add_max_min_avg(features, idle_time, 'idle_time')
    free_capacity = tuple(v.capacity - v.load for v in vehicles)
    add_max_min_avg(features, free_capacity, 'free_capacity')
    fitting_candidates = tuple(v.get_fitting_candidates(deliveries) for v in vehicles)
    add_max_min_avg(features, fitting_candidates, 'fitting_candidates')
    expected_fit = tuple(v.expected_fit(deliveries) for v in vehicles)
    add_max_min_avg(features, expected_fit, 'expected_fit')
    
    distances_between_routes_tw = tuple(v1.distance_to_route_tw(v2) for v1,v2 in itertools.product(vehicles,vehicles) if v1.nr < v2.nr)
    add_max_min_avg(features, distances_between_routes_tw, 'distances_between_routes_tw')
    
    
    if update_feature_header_line:
        feature_header_line += 'total_delta\tdist_delta\tviolation_delta\tnr_violation_delta\tnr_vehicles_delta\ttime\n'
    
    update_feature_header_line = False
    
    return features

def select_customers_routes(temp_sol, nr_to_delete):
    '''
    Randomly select an anchor route. 
    Create probabilities of the other routes based on the distance to the anchor route. 
    Randomly select neighboring routes based on these probabilities. 
    ''' 
    global P_randomness
    used_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0]
    
    #choose one route to build neighborhood around
    anchor_route = np.random.choice([v for v in used_vehicles if v.nr_tw_violations==0])
     
    vehicles_without_anchor = [v for v in used_vehicles if not v is anchor_route]
    
    #calculate distance from this route to other routes
    distances = [anchor_route.distance_to_route_tw(v) for v in vehicles_without_anchor]
    #Sort vehicles, and then create weight based on position in sorted vector
    sorted_vehicles = [x for x,_ in sorted(zip(vehicles_without_anchor, distances), key=lambda pair: pair[1])]
    p = P_randomness
    weights = [ math.pow(len(vehicles_without_anchor) - i, p) for i in range(len(vehicles_without_anchor))]
    cum_weight = sum(weights)
    probabilities = [weights[i]/cum_weight for i in range(len(vehicles_without_anchor))]
    
    vehicles = list(np.random.choice(sorted_vehicles, size=nr_to_delete, replace=False, p=probabilities))

    vehicles.append(anchor_route)
    
    customers_vehicles = [(c,v) for v in vehicles for c in v.route]
    
    return customers_vehicles, ""

def choose_ML(temp_sol, candidates, it_nr):
    '''
    Create the features for all candidate neighborhoods. 
    Make a prediction and return the vehicles in the chosen neighborhood.
    ''' 
    
    features = [record_all_features(temp_sol, candidate_neighborhood, it_nr) for candidate_neighborhood,_ in candidates]
    
    index_best, predictions = make_prediction(features)
    
    anchors_vehicles, msg_anchor = candidates[index_best]
    
    anchors = [c for c,_ in anchors_vehicles]
    vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    
    return vehicles, msg_anchor

def choose_random(temp_sol, candidates):
    '''
    Choose one random neighborhood out of the given candidates neighborhoods
    ''' 
    
    choose_index = np.random.choice(len(candidates))
    
    candidates_vehicles,_ = candidates[choose_index]
    
    anchors = [c for c,_ in candidates_vehicles]
    vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
    
    return vehicles, ''

def choose_best(temp_sol, candidates, k, setting):
    ''' 
    if k == 1: We run all neighborhoods and choose the best neighborhood and return it. 
    if k > 1: We run all neighborhoods, and sample from the best k neighborhoods, do this either:
        - randomly (setting == 'random')
        - returning the k'th best (setting = 'fixed')
        - returning a solution at random with probabilities proportional to solution values (setting='proportional')
    '''
    
    msgs = []
    solution_values = []
    
    sets_of_old_new_vehicles = []
    
    for anchors_vehicles, anchor_msg in candidates:
        
        anchors = [c for c,_ in anchors_vehicles]
        old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
        vehicles_distance = sum(v.distance for v in old_vehicles)
        
        new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
        msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg
        
        msgs.append(msg)
        
        if new_routes is None:
            solution_values.append(-np.infty)
        else:
            new_solution_value = temp_sol.total_value - vehicles_distance + sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
            solution_values.append(new_solution_value)
            
        old_vehicle_nrs = [v.nr for v in old_vehicles]
        sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
        
    
    if k == 1:
        chosen_index = solution_values.index(min(solution_values))
    else:
        indices_k_best = np.argsort(solution_values)[:k]
        
        if setting == 'random':
            chosen_index = np.random.choice(indices_k_best)
        elif setting == 'fixed':
            chosen_index = indices_k_best[k-1]
        elif setting == 'proportional':
            solution_values_best_k = [solution_values[i] for i in indices_k_best]
            sum_values = sum(solution_values_best_k)
            if sum_values == 0:
                chosen_index = np.random.choice(indices_k_best)
            else:
                p = [solution_values[i]/sum_values for i in indices_k_best]
                chosen_index = np.random.choice(indices_k_best, p=p)
    
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[chosen_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    return temp_sol, msgs[chosen_index]

def insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes):
    '''Add the new routes in the place of the old_vehicle_nrs in the temp_sol'''
    if new_routes is None:
        print('No new routes were returned by VROOM, we set improvement for this neighborhood to 0')
        return
    
    i = 0
    
    for j,v in enumerate(temp_sol.vehicles):
        if v.nr in old_vehicle_nrs:
            if i < len(new_routes):
                route = [temp_sol.instance.customers[c.nr-1] for c in new_routes[i]]
                temp_sol.vehicles[j].route = route
            else:
                temp_sol.vehicles[j].route = []
            temp_sol.vehicles[j]._needs_update=True
            i+=1
            
    temp_sol.update_cost()


def data_collection(temp_sol, candidates, it_nr, datacollection_strategy='random'):
    ''' 
    We record the features for all candidates, we run all neighborhoods in order to complete the y-values of the features
    Then we use the datacollection strategy to decide whcih neighbohrood to return.
    '''
    msgs = []
    feature_msg = ""
    all_features = []
    
    sets_of_old_new_vehicles = []
    
    
    for anchors_vehicles, anchor_msg in candidates:
        #record features
        features = record_all_features(temp_sol, anchors_vehicles, it_nr)
        if datacollection_strategy == "ML":
            all_features.append(features)
        feature_string = "\t".join([f"{feature:.3f}" for feature in features])
        
        anchors = [c for c,_ in anchors_vehicles]
        old_vehicles = [v for v in temp_sol.vehicles if len(v.route) > 0 and v.route[0] in anchors]
        vehicles_distance = sum(v.distance for v in old_vehicles)
        num_old_vehicles = len(old_vehicles)

        tic = time()
        new_routes, msg = vroom_insertion(temp_sol, old_vehicles, return_routes=True)
        toc = time()
        insert_customers_time = toc-tic
        msg = anchor_msg +"\t" + str(len(candidates)) +":\t"+ msg

        if new_routes is None:
            #Error in calculating new routes in vroom. set to 0
            dist_delta = 0
            total_delta = 0
            nr_vehicles_delta = 0
        else:
            dist_delta = vehicles_distance - sum(temp_sol.instance.calculate_route_distance(r) for r in new_routes)
            total_delta = dist_delta
            nr_vehicles_delta = len(new_routes) - num_old_vehicles

        dt_string = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        feature_string = dt_string + "\t" + feature_string
        feature_string += f"\t{total_delta:.3f}\t{dist_delta:.3f}\t{nr_vehicles_delta}\t{insert_customers_time:.3f}\n"
        feature_msg += feature_string

        msgs.append(msg)
        
        old_vehicle_nrs = [v.nr for v in old_vehicles]
        sets_of_old_new_vehicles.append((old_vehicle_nrs,new_routes))
    
    if datacollection_strategy == 'random':
        choose_index = np.random.choice(len(candidates))
    elif datacollection_strategy == 'ML':
        index_best, predictions = make_prediction(all_features)
        
        choose_index = index_best
        
    old_vehicle_nrs, new_routes = sets_of_old_new_vehicles[choose_index]
    insert_new_vehicles(temp_sol, old_vehicle_nrs, new_routes)
    
    return temp_sol, msgs[choose_index], feature_msg

def vroom_insertion(temp_sol, vehicles, debug=False, return_routes=False):
    
    '''Create a subproblem for some of the routes.
    '''
    new_routes, msg = vroom_operator.vroom_insert(vehicles, debug)
    
    if return_routes:
        return new_routes, msg
    else:
        if not new_routes is None:
            insert_new_vehicles(temp_sol, [v.nr for v in vehicles], new_routes)
        return msg

def iteration_solver(temp_sol,nr_to_delete, it_nr, strategy, pool_size):
    '''
    If the pool size is 1, only 1 neighborhood is created and this one will be destroyed and removed
    If the pool size > 1, more neighborhoods are created, and one is chosen based on the strategy:
        if strategy = 'random', a random neighborhood is chosen
        if strategy = 'best', all the neighborhoods are sent to the optimizer and the best one is taken
        if strategy = 'ML', the neighborhood with the best prediction is sent to the solver
        if strategy = 'data_collection', all the neighborhoods are sent to the optimizer, and a random one is taken
    
    '''
    
    feature_msg = None
    
    candidates_vehicles = [select_customers_routes(temp_sol, nr_to_delete) for i in range(pool_size)]
    
    if strategy == 'ML':
        vehicles, msg_anchor = choose_ML(temp_sol, candidates_vehicles, it_nr)
    elif strategy == 'random':
        vehicles, msg_anchor = choose_random(temp_sol, candidates_vehicles)
    elif strategy == 'best':
        #In case of choosing best, the solution (instead of the neighborhood) is returned, because it is calculated already anyway
        temp_sol, msg = choose_best(temp_sol, candidates_vehicles,options.k, options.oracle_mode)
    elif strategy == 'data_collection':
        temp_sol, msg, feature_msg = data_collection(temp_sol, candidates_vehicles, it_nr, options.datacollection_strategy)
    
    if strategy == 'ML' or strategy == 'random':
        msg = vroom_insertion(temp_sol, vehicles)
        msg = msg_anchor +"\t" + str(nr_to_delete) +":\t"+ msg
    
    return temp_sol, msg, feature_msg