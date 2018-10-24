#!/usr/bin/env python3
# -*- coding: utf-8 -*-
####################################################################################################################################################################################################################################
########################################################################################################### DOCUMENTATION ##########################################################################################################
####################################################################################################################################################################################################################################
"""
    This code allows reproducing the experiments from the following paper:
    "A neighborhood-preserving translation operator on graphs"
    (Bastien Pasdeloup, Vincent Gripon, Jean-Charles Vialatte, Nicolas Grelier, Dominique Pastor, Pascal Frossard)
"""
####################################################################################################################################################################################################################################
############################################################################################################## IMPORTS #############################################################################################################
####################################################################################################################################################################################################################################

import igraph
import numpy
import heapq
import shutil
import copy
import random
import os
import itertools
import matplotlib
import matplotlib.pyplot as pyplot
import time
import sys
import timeit
import subprocess

####################################################################################################################################################################################################################################
############################################################################################################# CONSTANTS ############################################################################################################
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Details for the random geometric graph creation.
"""

### CODE ###
GRAPH_ORDER = 100
RADIUS = 0.15

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Boolean to choose plotting with automatic layout.
"""

### CODE ###
AUTOMATIC_LAYOUT = False

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Nb hops to consider to initialize V1.
"""

### CODE ###
V_SRC_NEIGHBORHOOD = 1

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Nb hops to consider when adding a vertex to V2 (when looking to translate V1).
"""

### CODE ###
NB_HOPS_V2 = 1

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Seed for the random numbers generator.
    List for multiple tests.
"""

### CODE ###
RANDOM_SEED = range(40)

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Controls the complexity of the heuristics when using a greedy approach to estimate translations.
    Use float("inf") for exhaustive search.
    List for grid search.
"""

### CODE ###
HEURISTICS_COMPLEXITY = [1]

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Parameters controlling the importance of loss, EC violations and deformations in the score function.
    Lists for grid search.
"""

### CODE ###
ALPHA = [0.1, 0.5, 1.0]
BETA = [0.1, 0.5, 1.0]
GAMMA = [0.1, 0.5, 1.0]

####################################################################################################################################################################################################################################
############################################################################################################# FUNCTIONS ############################################################################################################
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Measures time taken by the function.
    Decorator to debug and optimize some of them.
    Adapted from https://www.zopyx.com/andreas-jung/contents/a-python-decorator-for-measuring-the-execution-time-of-methods.
    ---
    Arguments:
        * method: Method to run
    ---
    Returns:
        * Encapsulator for the method.
"""

### CODE ###
def timeit (method) :
    
    # Definition of global variables if needed
    global calls_per_method
    global total_time_per_method
    if "calls_per_method" not in globals() :
        calls_per_method = {}
        total_time_per_method = {}
    
    # Initilization of an entry for the method if needed
    if method.__name__ not in calls_per_method :
        calls_per_method[method.__name__] = 0
        total_time_per_method[method.__name__] = 0.0

    # Encapsulation with time measurement
    def timed (*args, **kw) :
        start = time.time()
        result = method(*args, **kw)
        stop = time.time()
        calls_per_method[method.__name__] += 1
        total_time_per_method[method.__name__] += (stop - start)
        caller = sys._getframe().f_back.f_code.co_name
        if caller in total_time_per_method :
            total_time_per_method[caller] -= (stop - start)
        return result
    return timed
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Plots an approximate translation on a graph.
    ---
    Arguments:
        * graph: Graph to plot.
        * composition: Composition up to the current location.
        * approximate_translation: Approximate translation to apply from the current location.
        * depict_objective: Boolean for exporting the objective in a different color.
        * file_name: File where to export the figure.
    ---
    Returns:
        * None.
"""

### CODE ###
@timeit
def plot_approximate_translation (graph, composition, approximate_translation, depict_objective, file_name) :
    
    # We initialize a palette of colors at first call (i.e., when we plot the objective)
    global palette
    global palette_mapping
    if depict_objective :
        palette = igraph.drawing.colors.ClusterColoringPalette(len(approximate_translation))
        initial_vertices = list(approximate_translation.keys())
        palette_mapping = {initial_vertices[i]: i for i in range(len(approximate_translation))}
    
    # Export to LaTeX file
    file_object = open(file_name, "w")
    file_object.write("\\begin{tikzpicture}\n")
    for v in range(len(graph.vs)) :
        vertex_color = "white"
        vertex_text = ""
        if v in approximate_translation :
            if composition is not None :
                initial_v = [initial_v for initial_v in composition if composition[initial_v] == v][0]
            else :
                initial_v = v
            vertex_color = "{rgb:red," + str(int(255 * palette.get(palette_mapping[initial_v])[0])) + ";green," + str(int(255 * palette.get(palette_mapping[initial_v])[1])) + ";blue," + str(int(255 * palette.get(palette_mapping[initial_v])[2])) + "}"
            if not depict_objective and approximate_translation[v] is None :
                vertex_text = "$\\bot$"
        file_object.write("    \\node (" + str(v) + ") [draw, circle, scale=0.3, fill=" + vertex_color + "] at (" + str(round(graph.vs[v]["x"], 2)) + "\\textwidth, " + str(round(graph.vs[v]["y"], 2)) + "\\textwidth) {" + vertex_text + "};\n")
    for e in graph.es :
        if e.source in approximate_translation and approximate_translation[e.source] == e.target :
            file_object.write("    \\path[-latex, green, thick] (" + str(e.source) + ") edge (" + str(e.target) + ");\n")
        elif e.target in approximate_translation and approximate_translation[e.target] == e.source :
            file_object.write("    \\path[-latex, green, thick] (" + str(e.target) + ") edge (" + str(e.source) + ");\n")
        else :
            file_object.write("    \\path[] (" + str(e.source) + ") edge (" + str(e.target) + ");\n")
    for v in approximate_translation :
        edge_id = graph.get_eid(v, approximate_translation[v], directed=False, error=False)
        if edge_id == -1 and  approximate_translation[v] is not None :
            edge_details = "-latex, blue, dashed, thick" if depict_objective else "-latex, red"
            file_object.write("    \\path[" + edge_details + "] (" + str(v) + ") edge (" + str(approximate_translation[v]) + ");\n")
    file_object.write("\\end{tikzpicture}")
    file_object.close()
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Returns the composition t1 o t2 of two approximate translations.
    ---
    Arguments:
        * t1: First approximate translation.
        * t2: First approximate translation.
    ---
    Returns:
        * Composed approximate translation.
"""

### CODE ###
@timeit
def compose (t1, t2) :
    
    # If any is None, composition is the other (to simplify calls)
    if t1 is None :
        composed_t = copy.copy(t2)
    elif t2 is None :
        composed_t = copy.copy(t1)
    
    # We compose the two approximate translations propagating Nones
    else :
        composed_t = {}
        for v in t1 :
            if t1[v] is not None :
                composed_t[v] = t2[t1[v]]
            else :
                composed_t[v] = None
    
    # Done
    return composed_t
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Generates a random geometric graph.
    ---
    Arguments:
        * graph_order: Graph order.
        * radius: Radius under which we add an edge.
        * file_name: Name of the LaTeX file where to export the graph.
    ---
    Returns:
        * An igraph object.
"""

### CODE ###
@timeit
def create_rg_graph (graph_order, radius, file_name) :
    
    # We generate 2D coordinates
    coordinates = numpy.random.uniform(low=0.0, high=1.0, size=(graph_order, 2))
    
    # We generate a graph with edges when l2 norm of coordinates is lower than given radius
    graph = igraph.Graph()
    graph.add_vertices(graph_order)
    for v1 in range(coordinates.shape[0]) :
        for v2 in range(v1 + 1, coordinates.shape[0]) :
            distance = numpy.linalg.norm(coordinates[v1, :] - coordinates[v2, :])
            if distance < radius :
                graph.add_edges([(v1, v2)])
    
    # For a clean export, we generate nicer coordinates with Graphviz
    if AUTOMATIC_LAYOUT :
        margin = 1
        file_xdot = open("tmp.dot", "w")
        file_xdot.write("graph G\n")
        file_xdot.write("{\n")
        file_xdot.write("    compound=true;\n")
        for v in range(len(graph.vs)) :
            file_xdot.write("    ")
            for c in range(margin) :
                file_xdot.write("subgraph cluster_" + str(c) + "_" + str(v) + " { ")
            file_xdot.write(str(v) + "; ")
            for c in range(margin) :
                file_xdot.write("}")
            file_xdot.write("\n")
        for e in graph.es :
            file_xdot.write("    " + str(e.source) + " -- " + str(e.target) + ";\n")
        file_xdot.write("}")
        file_xdot.close()
        raw_output = str(subprocess.check_output(["neato", "-Tdot", "tmp.dot"]))
        for row in raw_output.split("subgraph") :
            splitted_row = [word for word in row.split("\\t") if len(word) > 0]
            print(splitted_row)
            try :
                v = int(splitted_row[1])
                graph.vs[v]["x"] = float(splitted_row[3].split("\"")[1].split(",")[0])
                graph.vs[v]["y"] = float(splitted_row[3].split("\"")[1].split(",")[1])
            except :
                pass
        max_x = max(graph.vs[v]["x"] for v in range(len(graph.vs)))
        max_y = max(graph.vs[v]["y"] for v in range(len(graph.vs)))
        for v in range(len(graph.vs)) :
            graph.vs[v]["x"] = graph.vs[v]["x"] / max_x
            graph.vs[v]["y"] = graph.vs[v]["y"] / max_y
        shutil.rmtree("tmp.dot", ignore_errors=True)
    else :
        for v in range(len(graph.vs)) :
            graph.vs[v]["x"] = coordinates[v, 0]
            graph.vs[v]["y"] = coordinates[v, 1]
    
    # Export to LaTeX file
    file_object = open(file_name, "w")
    file_object.write("\\begin{tikzpicture}\n")
    for v in range(len(graph.vs)) :
        file_object.write("    \\node (" + str(v) + ") [draw, circle, scale=0.3, fill=white] at (" + str(round(graph.vs[v]["x"], 2)) + "\\textwidth, " + str(round(graph.vs[v]["y"], 2)) + "\\textwidth) {};\n")
    for e in graph.es :
        file_object.write("    \\path[] (" + str(e.source) + ") edge (" + str(e.target) + ");\n")
    file_object.write("\\end{tikzpicture}")
    file_object.close()
    
    # Done
    return graph
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Finds the list of vertices at most k-hops away from the given one.
    ---
    Arguments:
        * graph: Graph on which to define the signal.
        * vertex: Vertex on which the signal will be centered.
        * k: K-neighborhood to include in the initial signal domain.
    ---
    Returns:
        * A set of vertices.
"""

### CODE ###
@timeit
def k_hop_vertices (graph, vertex, k) :
    
    # We create a list with the given node and its neighborhood at most k-hops away
    # Not optimal way of doing it, but fine for the experiments
    vertices = set([vertex])
    for neighborhood in range(k) :
        new_neighbors = set()
        for v in vertices :
            new_neighbors.update(graph.neighbors(v))
        vertices.update(new_neighbors)
    
    # List for convenience
    vertices = list(vertices)
    return vertices

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Loss percentage of an approximate translation.
    ---
    Arguments:
        * approximate_translation: Approximate translation for which to compute the loss.
    ---
    Returns:
        * Loss percentage.
"""

### CODE ###
@timeit
def loss (approximate_translation) :
    
    # Loss
    nb_loss = len([v for v in approximate_translation if approximate_translation[v] is None])
    
    # Percentage
    ratio_loss = float(nb_loss) / float(len(approximate_translation))
    return ratio_loss

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Gives the percentage of SNP violations.
    ---
    Arguments:
        * approximate_translation: Approximate translation for which to compute the ratio.
        * graph: Graph on which the approximate translation is defined
    ---
    Returns:
        * Percentage of added or removed neighborhoods.
"""

### CODE ###
@timeit
def snp_violations (approximate_translation, graph) :
    
    # Non-lost vertices
    translation_support = [v for v in approximate_translation if approximate_translation[v] is not None]
    nb_non_lost = len(translation_support)
    
    # SNP violations
    nb_snp_violations = 0
    for i in range(nb_non_lost) :
        v1 = translation_support[i]
        for j in range(i + 1, nb_non_lost) :
            v2 = translation_support[j]
            edge_created = graph.get_eid(v1, v2, directed=False, error=False) == -1 and graph.get_eid(approximate_translation[v1], approximate_translation[v2], directed=False, error=False) != -1
            edge_removed = graph.get_eid(v1, v2, directed=False, error=False) != -1 and graph.get_eid(approximate_translation[v1], approximate_translation[v2], directed=False, error=False) == -1
            if edge_created or edge_removed :
                nb_snp_violations += 1
    
    # Percentage
    ratio_snp_violations = 0.0 if nb_non_lost <= 1 else float(nb_snp_violations) / float(nb_non_lost * (nb_non_lost - 1.0) / 2.0)
    return ratio_snp_violations

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Gives the percentage of EC violations.
    ---
    Arguments:
        * approximate_translation: Approximate translation for which to compute the ratio.
        * graph: Graph on which the approximate translation is defined
    ---
    Returns:
        * Percentage of EC violations.
"""

### CODE ###
@timeit
def ec_violations (approximate_translation, graph) :
    
    # Non-lost vertices
    translation_support = [v for v in approximate_translation if approximate_translation[v] is not None]
    nb_non_lost = len(translation_support)
    
    # EC violations
    nb_ec_violations = 0
    for v in translation_support :
        edge_exists = graph.get_eid(v, approximate_translation[v], directed=False, error=False) != -1
        if not edge_exists :
            nb_ec_violations += 1
    
    # Percentage
    ratio_ec_violations = 0.0 if nb_non_lost == 0 else float(nb_ec_violations) / float(nb_non_lost)
    return ratio_ec_violations

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Gives the average deformation caused by a transformation.
    ---
    Arguments:
        * approximate_translation: Approximate translation for which to compute the ratio.
        * graph: Graph on which the approximate translation is defined
    ---
    Returns:
        * Average difference between shortest paths.
"""

### CODE ###
@timeit
def deformation (approximate_translation, graph) :

    # We compute the shortest paths if not done once and for all
    global all_shortest_paths
    if "all_shortest_paths" not in globals() :
        all_shortest_paths = graph.shortest_paths_dijkstra()

    # Non-lost vertices
    translation_support = [v for v in approximate_translation if approximate_translation[v] is not None]
    nb_non_lost = len(translation_support)
    
    # Difference in pairwise distances
    total_deformation = 0.0
    for i1 in range(nb_non_lost) :
        v1 = translation_support[i1]
        for i2 in range(i1 + 1, nb_non_lost) :
            v2 = translation_support[i2]
            total_deformation += abs(all_shortest_paths[v1][v2] - all_shortest_paths[approximate_translation[v1]][approximate_translation[v2]])
    
    # Average
    average_deformation = 0.0 if nb_non_lost <= 1 else float(total_deformation) / float(nb_non_lost * (nb_non_lost - 1.0) / 2.0)
    return average_deformation

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Score function for approximate translations.
    ---
    Arguments:
        * alpha: Contribution of the loss.
        * beta: Contribution of the EC violations.
        * gamma: Contribution of the deformations.
    ---
    Returns:
        * Weighted score of the given approximate translation, and details.
"""

### CODE ###
@timeit
def score (approximate_translation, graph, alpha, beta, gamma) :
    
    # Useful quantities
    ratio_loss = loss(approximate_translation)
    ratio_ec_violations = ec_violations(approximate_translation, graph)
    average_deformation = deformation(approximate_translation, graph)

    # Linear combination
    result = alpha * ratio_loss + beta * ratio_ec_violations + gamma * average_deformation
    return result, ratio_loss, ratio_ec_violations, average_deformation

####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Selects a set V2 where to translate a given vertex V1.
    Here, we choose vertices that have a k-hop neighbor in V1.
    ---
    Arguments:
        * graph: Graph on which to search the vertices.
        * V1: Vertices to translate.
        * nb_hops: Number of hops to consider to be part of V2.
    ---
    Returns:
        * Set V2 of vertices at most one K away from a vertex in V1.
"""

### CODE ###
@timeit
def choose_V2 (graph, V1, nb_hops) :
    
    # We find the neighbors of vertices in V1
    vertices = set()
    for v in V1 :
        vertices.update(k_hop_vertices(graph, v, nb_hops))
    
    # List for convenience
    vertices = list(vertices)
    return vertices
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Translates a signal such that a given vertex becomes located to another given one.
    ---
    Arguments:
        * graph: Graph on which to find the composition.
        * V1: Set of initial vertices of interest that we want to translate.
        * v_src: Vertex in V1 that we want to relocate on v_tgt.
        * v_tgt: Target vertex for the translation.
        * alpha: Contribution of the loss in the score function.
        * beta: Contribution of the EC violations in the score function.
        * gamma: Contribution of the deformations in the score function.
        * k: Controls the complexity of the greedy algorithm in the minimize_score function.
    ---
    Returns:
        * Best sequence of approximate translations, and the path taken.
"""

### CODE ###
@timeit
def vsrc_to_vtgt (graph, V1, v_src, v_tgt, alpha, beta, gamma, k) :
    
    # Initialization of the algorithm
    predecessors = [None] * len(graph.vs)
    visited = [False] * len(graph.vs)
    queue = [(0.0, v_src, None, None)]
    
    # Main loop
    while len(queue) > 0 :
        
        # We extract the current location of v_src with minimum total score
        total_score_v1, v1, pred, t_from_pred = heapq.heappop(queue)
        if visited[v1] :
            continue
        visited[v1] = True
        predecessors[v1] = (pred, t_from_pred)
        
        # We stop once v_tgt is found
        if v1 == v_tgt :
            break

        # For the computation of the score, we choose V1 as the set of non-lost vertices up to here, and propagate v1 to a chosen set V2
        if t_from_pred is not None :
            V1 = [v for v in t_from_pred.values() if v is not None]
        V2 = choose_V2(graph, V1, NB_HOPS_V2)
        for v2 in list(set(V2) - set([v1])) :
            if not visited[v2] :
                
                # Here, minimize_score finds the best approximate translation t : V1 -> V2 s.t. v1 is translated to v2
                t, score_t = minimize_score(v1, v2, graph, V1, V2, alpha, beta, gamma, k)
                heapq.heappush(queue, (total_score_v1 + score_t, v2, v1, t))

    # We extract the result
    try :
        path = [v_tgt]
        approximate_translations = []
        while path[0] != v_src :
            approximate_translations = [predecessors[path[0]][1]] + approximate_translations
            path = [predecessors[path[0]][0]] + path
        return path, approximate_translations
    except :
        raise Exception("Impossible to reach target from source")
    
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Finds the best approximate translation t : V1 -> V2 U {lost} s.t. t(v1) = v2.
    ---
    Arguments:
        * v1: Vertex to translate to v2.
        * v2: Vertex that will receive v1.
        * graph: Graph on which to find the approximate translation.
        * V1: Set of vertices of interest that we want to translate.
        * V2: Possible targets for vertices in V1.
        * alpha: Contribution of the loss in the score function.
        * beta: Contribution of the EC violations in the score function.
        * gamma: Contribution of the deformations in the score function.
        * k: Controls the complexity of the greedy algorithm.
    ---
    Returns:
        * Best approximate translation (i.e., minimizing the deformation of V1) s.t. v1 is moved to v2.
"""

### CODE ###
@timeit
def minimize_score (v1, v2, graph, V1, V2, alpha, beta, gamma, k) :
    
    # We find the best association of k vertices in V1 -> V2, or None, until all elements have an image
    best_t = {v1 : v2}
    best_score = 0
    while len(best_t) != len(V1) :

        # We remember the current best
        current_best_assignment = {}
        current_best_score = float("inf")
        
        # We iterate over all pairs of permutations of K elements in V1 and in V2
        remaining_V1 = list(set(V1) - set(best_t.keys()))
        adjusted_k = min(k, len(remaining_V1))
        remaining_V2 = list(set(V2) - set(best_t.values())) + [None] * adjusted_k
        for V1_k in itertools.permutations(remaining_V1, adjusted_k) :
            for V2_k in set(itertools.permutations(remaining_V2, adjusted_k)) :
                
                # We complete the translation temporarily to compute the score
                # We do this in place to avoid copying structures
                assignment = {V1_k[i] : V2_k[i] for i in range(adjusted_k)}
                best_t.update(assignment)
                assignment_score, _, _, _ = score(best_t, graph, alpha, beta, gamma)
                for v in assignment :
                    del best_t[v]
                
                # We keep the assignment if it improves the score
                if assignment_score < current_best_score :
                    current_best_assignment = assignment
                    current_best_score = assignment_score
        
        # We apply the best assignment for the next round
        best_t.update(current_best_assignment)
        best_score = current_best_score

    # Done
    return best_t, best_score
    
####################################################################################################################################################################################################################################
############################################################################################################### MAIN ###############################################################################################################
####################################################################################################################################################################################################################################

### DOCUMENTATION ###
"""
    Entry point.
    ---
    Arguments:
        * None.
    ---
    Returns:
        * None.
"""

### CODE ###
@timeit
def main () :
        
    # We iterate over all asked seeds
    for seed in RANDOM_SEED :

        # We set the seed
        random.seed(seed)
        numpy.random.seed(seed)
        seed_directory = "seed_" + str(seed)
        shutil.rmtree(seed_directory, ignore_errors=True)
        os.makedirs(seed_directory)
        sys.stdout = open(seed_directory + os.path.sep + "Log.txt", "w")
        print("Setting seed=" + str(seed))
        
        # May fail if graph is not connected enough
        try :
            
            # We create a graph
            graph = create_rg_graph(GRAPH_ORDER, RADIUS, seed_directory + os.path.sep + "Graph.tex")
            
            # We create a localized signal, choose v_src in V1, and v_tgt outside of it
            v_src = random.randint(0, GRAPH_ORDER-1)
            V1 = k_hop_vertices(graph, v_src, V_SRC_NEIGHBORHOOD)
            possible_v_tgt = [v for v in range(len(graph.vs)) if v not in V1]
            v_tgt = possible_v_tgt[random.randint(0, len(possible_v_tgt)-1)]
            
            # We plot the objective
            objective_translation = {v : None for v in V1}
            objective_translation[v_src] = v_tgt
            plot_approximate_translation(graph, None, objective_translation, True, seed_directory + os.path.sep + "Objective.tex")
            print("Objective is to go from " + str(v_src) + " to " + str(v_tgt) + " while distorting a localized signal of " + str(len(V1)) + " elements as little as possible")

            # We prepare a figure for the Pareto optima
            pyplot.figure()

            # Grid serch over all parameters
            for k in HEURISTICS_COMPLEXITY :
                all_results = []
                for alpha in ALPHA :
                    for beta in BETA :
                        for gamma in GAMMA :

                            # We create the output directory
                            print(".... Using parameters K=" + str(k) + ", ALPHA=" + str(alpha) + ", BETA=" + str(beta) + ", GAMMA=" + str(gamma))
                            directory = seed_directory + os.path.sep + "k_" + str(k) + "_alpha_" + str(alpha) + "_beta_" + str(beta) + "_gamma_" + str(gamma)
                            shutil.rmtree(directory, ignore_errors=True)
                            os.makedirs(directory)

                            # We find the best sequence of approximate translations
                            path, approximate_translations = vsrc_to_vtgt(graph, V1, v_src, v_tgt, alpha, beta, gamma, k)

                            # Step by step result
                            composition = None
                            for i in range(len(approximate_translations)) :
                                score_t, loss_t, ec_violations_t, deformations_t = score(approximate_translations[i], graph, alpha, beta, gamma)
                                plot_approximate_translation(graph, composition, approximate_translations[i], False, directory + os.path.sep + "Step " + str(i + 1) + ".tex")
                                print("........ " + str(path[i]) + " -> " + str(path[i+1]) + " (loss " + str(round(100 * loss_t, 2)) + "%, EC violations " + str(round(100 * ec_violations_t, 2)) + "%, average deformation " + str(round(deformations_t, 2)) + ")")
                                composition = compose(composition, approximate_translations[i])
                            
                            # Overall result
                            loss_composition = loss(composition)
                            snp_violations_composition = snp_violations(composition, graph)
                            plot_approximate_translation(graph, None, composition, False, directory + os.path.sep + "Composition.tex")
                            print("........ Best path of " + str(len(path) - 1) + " steps found: " + str(path) + " (loss " + str(round(100 * loss_composition, 2)) + "%, SNP violations of non-lost part " + str(round(100 * snp_violations_composition, 2)) + "%)")
                            
                            # We save the result for later to plot Pareto optima
                            all_results.append((loss_composition, snp_violations_composition, k, alpha, beta, gamma))
            
                # We Extract the Pareto optima within results for this value of K
                pareto_optima = []
                for result_1 in all_results :
                    if result_1 not in pareto_optima :
                        result_1_dominated = False
                        for result_2 in all_results :
                            same_result = (result_1[0] == result_2[0] and result_1[1] == result_2[1])
                            if not same_result and result_1[0] >= result_2[0] and result_1[1] >= result_2[1] :
                                result_1_dominated = True
                                break
                        if not result_1_dominated :
                            pareto_optima.append(result_1)
                pareto_optima.sort()
                
                # We print the Pareto optima
                for optimum in pareto_optima :
                    print("Pareto optimum (" + str(optimum[0]) + ", ", str(optimum[1]) + ") found for K=" + str(optimum[2]) + ", ALPHA=" + str(optimum[3]) + ", BETA=" + str(optimum[4]) + ", GAMMA=" + str(optimum[5]))
                
                # We plot the Pareto optima with different colors for the values of K
                pyplot.plot([pareto_optima[i][0] for i in range(len(pareto_optima))], [pareto_optima[i][1] for i in range(len(pareto_optima))], '-*', label="K = " + str(k))
            pyplot.title("Pareto optima")
            pyplot.xlabel("Total loss")
            pyplot.ylabel("Total SNP violations of non-lost part")
            pyplot.xlim((0.0, 1.0))
            pyplot.ylim((0.0, 1.0))
            pyplot.legend(loc="best")
            pyplot.savefig(seed_directory + os.path.sep + "Results.png")
        
        # Seed is ignored in case of non-reachable solution
        except :
            print("Target cannot be reached")
            shutil.rmtree(seed_directory, ignore_errors=True)
        
####################################################################################################################################################################################################################################

# Go
main()

# Time measurement details
global calls_per_method
global total_time_per_method
time_per_call = {key: (total_time_per_method[key] / float(calls_per_method[key])) for key in calls_per_method}
pyplot.figure()
matplotlib.rcParams.update({"font.size": 7})
pyplot.subplot(1, 3, 1)
pyplot.bar(calls_per_method.keys(), calls_per_method.values())
pyplot.title("Nb calls")
pyplot.ylabel("Time")
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.subplot(1, 3, 2)
pyplot.bar(total_time_per_method.keys(), total_time_per_method.values())
pyplot.title("Total time")
pyplot.ylabel("Time")
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.subplot(1, 3, 3)
pyplot.bar(time_per_call.keys(), time_per_call.values())
pyplot.title("Time / call")
pyplot.ylabel("Time")
pyplot.xticks(rotation=90)
pyplot.tight_layout()
pyplot.savefig("Timing.png")

####################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################
