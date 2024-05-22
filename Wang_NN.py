#!/usr/bin/python3 

import sys, random, math
import time
import numpy as np
import tsplib95


############################## PRINT FUNCTIONS ###############################

def print_int_matrix( matrix, matrix_name="" ):
    if matrix_name != "":
        print( "\n\n\033[1;93m", 30 * "#", matrix_name, 30 * "#", "\033[0m" )

    for row in matrix:
        print()
        for elem in row:
            print( "\033[1;92m", f"{elem:3}", "\033[0m", end=" " )
        print("\n")


def print_matrix( matrix, matrix_name="" ):
    if matrix_name != "":
        print( "\n\n\033[1;93m", 30 * "#", matrix_name, 30 * "#", "\033[0m" )

    for row in matrix:
        print()
        for elem in row:
            print( "\033[1;92m", "{:.3f}\033[0m".format(elem), end=" " )
        print("\n")


def print_array( array, array_name="" ):
    if array_name != "":
        print( "\n\n\033[1;93m", 30 * "#", array_name, 30 * "#", "\033[0m" )

    print()
    for elem in array:
        print( "\033[1;92m", "{:.5f}\033[0m".format(elem), end=" " )
    print("\n")


def print_partial_array( array, array_name="", processed_until=0 ):
    if array_name != "":
        print( "\n\n\033[1;93m", 30 * "#", array_name, 30 * "#", "\033[0m" )

    print()
    for i in range( len (array) ):
        if i <= processed_until:
            print( "\033[1;91m", "{:.5f}\033[0m".format( array[ i ] ), end=" " )
        else:
            print( "\033[1;92m", "{:.5f}\033[0m".format( array[ i ] ), end=" " )

############################  UTILITY FUNCTIONS ##############################

def is_requirement_met( matrix_solution, sums_of_rows, sums_of_cols ):
    passed_elements = 0
    for row in range( len( matrix_solution ) ):
        for col in range( len( matrix_solution ) ):    
            # Do not calculate the element from the first column as its
            # x_{ij} has been artifically increased as the first column
            # of the C_{ij} has been multiplied by parameter p
            if col == 0:
                continue

            # If for any element the requirement is not met
            # Then need to continue iterations
            if( sums_of_rows[row] + sums_of_cols[col] - 2 > epsilon ):
                #print( "Passed elements count: ", passed_elements )

                #######################TEMP#####################################
                if iteration % 20 == 0:
                    overall_passed = 0
                    for row in range( len( matrix_solution ) ):
                        for col in range( len( matrix_solution ) ):    
                            if col == 0:
                                continue

                            if( sums_of_rows[row] + sums_of_cols[col] - 2 > epsilon ):
                                overall_passed += 1

                    #print( "Overall passed eleemnts:", overall_passed )
                #######################TEMP_END#################################

                return False 

            passed_elements += 1
                
    # If the function didn't return to this point,
    # Then for all the  elements the requirement is met
    return True


def calculate_partial_sums( matrix ):
    matrix_size = len( matrix )

    vertical_partial_sums = np.zeros( ( matrix_size, matrix_size ), dtype=float )
    horizontal_partial_sums = np.zeros( ( matrix_size, matrix_size ), dtype=float )

    for row in range( matrix_size ):
        for col in range( matrix_size ):
            cur_elem = matrix[ row, col ]
            if row != 0 and col != 0:
                # Each next element of the vertical partial sums is calculated by
                # adding the previous partial sum on the column to the current element
                # on from matrix
                prev_row_elem = vertical_partial_sums[ row - 1, col ]
                vertical_partial_sums[ row, col ] = prev_row_elem + cur_elem

                # Each next element of the horizontal partial sums is calculated by
                # adding the previous partial sum on the row to the current element
                # on from matrix
                prev_col_elem = horizontal_partial_sums[ row, col - 1 ]
                horizontal_partial_sums[ row, col ] = prev_col_elem + cur_elem
            elif row == 0 and col == 0:
                # As this is the first element there are no previous calculations
                vertical_partial_sums[ row, col ] = cur_elem
                horizontal_partial_sums[ row, col ] = cur_elem
            elif row == 0:
                # As this is the first element of the row there are no previous
                # vertical calculations
                vertical_partial_sums[ row, col ] = cur_elem

                # Each next element of the horizontal partial sums is calculated by
                # adding the previous partial sum on the row to the current element
                # on from matrix
                prev_col_elem = horizontal_partial_sums[ row, col - 1 ]
                horizontal_partial_sums[ row, col ] = prev_col_elem + cur_elem
            else:
                # This is the case when col == 0

                # Each next element of the vertical partial sums is calculated by
                # adding the previous partial sum on the column to the current element
                # on from matrix
                prev_row_elem = vertical_partial_sums[ row - 1, col ]
                vertical_partial_sums[ row, col ] = prev_row_elem + cur_elem

                # As this is the first element of the column there are no previous
                # horizontal calculations
                horizontal_partial_sums[ row, col ] = cur_elem

    return horizontal_partial_sums, vertical_partial_sums
                
    

def calculate_sums_of_each_row( matrix ):
    partial_sum_for_elems = []
    sums_of_rows = []
    for row in matrix:
        sum = 0
        for elem in row:
            sum += elem
        sums_of_rows.append( sum )

    return sums_of_rows
    
def calculate_sums_of_each_column( matrix ):
    sums_of_cols = []
    for col in range( len( matrix ) ):
        sum = 0
        for row in range( len( matrix ) ):
            sum += matrix_solution[ row, col ]
        sums_of_cols.append( sum )

    return sums_of_cols

def get_horizontal_partial_sums( row, col ):
    # This is a utility function to get the horizontal partial sum of already
    # updated x_{ij}s in the row: H' and horizontal partial sum of the elements
    # from the previous iteration: H
    horizontal_new_step = 0
    horizontal_old_step = 0

    # As in the matrix all the elements preceding the elem[row, col] are 
    # updated, then the H' = horizontal_partial_sums[ row, col - 1 ]
    # For the first element in the column, there are no updated
    # partial sums
    if col != 0:
        horizontal_new_step = horizontal_partial_sums[ row, col - 1 ]
    
    # The value of the H should be calculated by summing up all the 
    # elements after the col element, which can be done using already existing
    # partial sums. The sum of the elements after [col]th element equals to
    # sum_of_elems_in_the_row - partial_sum_until_the_cur_elem
    # Adding the x_{row, col} to the obtained result will get us the H
    row_sum = horizontal_partial_sums[ row, num_of_towns - 1 ]
    part_sum_cur_elem = horizontal_partial_sums[ row, col ]
    cur_elem = matrix_solution[ row, col ]
    
    horizontal_old_step = row_sum - part_sum_cur_elem + cur_elem

    return horizontal_old_step, horizontal_new_step


def get_vertical_partial_sums( row, col ):
    # This is a utility function to get the vertical partial sum of already
    # updated x_{ij}s in the row: V' and vertical partial sum of the elements
    # from the previous iteration: V
    vertical_new_step = 0
    vertical_old_step = 0

    # As in the matrix all the elements preceding the elem[row, col] are 
    # updated, then the V' = vertical_partial_sums[ row - 1, col ]
    # For the first element in the row, there are no updated
    # partial sums
    if row != 0:
        vertical_new_step = vertical_partial_sums[ row, col - 1 ]
    
    # The value of the V should be calculated by summing up all the 
    # elements after the row element, which can be done using already existing
    # partial sums. The sum of the elements after [row]th element equals to
    # sum_of_elems_in_the_col - partial_sum_until_the_cur_elem
    # Adding the x_{row, col} to the obtained result will get us the V
    col_sum = vertical_partial_sums[ num_of_towns - 1, col ]
    part_sum_cur_elem = vertical_partial_sums[ row, col ]
    cur_elem = matrix_solution[ row, col ]
    
    vertical_old_step = col_sum - part_sum_cur_elem + cur_elem

    return vertical_old_step, vertical_new_step

def calculate_path_length( matrix_weights, arr_solutions ):
    row = 0
    path_length = 0
    num_of_towns = len( matrix_weights )
    for i in range( num_of_towns ):
        next_town_index = arr_solutions[i % num_of_towns]
        print( row, "->", next_town_index, matrix_weights[row][next_town_index] )
        path_length += matrix_weights[row][next_town_index]
        row = next_town_index

    return path_length
    
################################ MAIN PROGRAM ################################
    
for hugachaga in range( 20 ):
    print( "\n\n\n\n", 30 * "#", hugachaga, 30 * "#" )
    file = open('example.txt', 'a')
    
    # Load the TSP problem
    tsp_problem = tsplib95.load( "gr17.tsp" )
    #tsp_problem = tsplib95.load( "gr666.tsp" )
    #tsp_problem = tsplib95.load( "gr229.tsp" )
    #tsp_problem = tsplib95.load( "gr96.tsp" )
    #tsp_problem = tsplib95.load( "d18512.tsp" )
    num_of_towns = len( list( tsp_problem.get_nodes() ) )
    
    # Create an empty matrix of weights
    matrix_weights = np.zeros( (num_of_towns, num_of_towns), dtype=int )
    
    # Initialize the weight matrix from the loaded tsp_problem
    for i in range( num_of_towns ):
        for j in range( num_of_towns ):
            edge = i + 1, j + 1
            #edge = i, j 
            matrix_weights[i,j] = tsp_problem.get_weight( *edge )

    # Generate a random solution matrix
    matrix_solution = np.random.rand( num_of_towns, num_of_towns )
    #print_matrix( matrix_solution, "Initial soluition matrix" )
    
    # Print the weights matrix
    #print_int_matrix( matrix_weights, "Weights (Distances)" )
    
    # The accuracy of meeting the requirement
    epsilon = 1 
    beta = 0.1
    deltaT = 0.001
    tau = 1000
    eta = 10
    lambdaa = 1
    p = 1000000
    
    matrix_weights[ :, 0 ] = p * matrix_weights[ :, 0 ]

    print_int_matrix( matrix_weights, "Weights (Distances)" )
    
    horizontal_partial_sums, vertical_partial_sums = \
                calculate_partial_sums( matrix_solution )
    
    # Initialize the matrix of u_{ij}s based on x_{ij}s
    matrix_neuron = np.zeros( (num_of_towns, num_of_towns), dtype=float )
    for row in range( num_of_towns ):
        for col in range ( num_of_towns ):
            x = matrix_solution[ row, col ]
            #print( "x: ", x )
            matrix_neuron[ row, col ] = -1 / beta * np.log( ( 1 - x ) / x )
    
    print_matrix( matrix_solution, "Initial Matrix of Solutions" )
    print_matrix( matrix_neuron, "Initial Matrix of Neurons" )

    start_time = time.time()

    iteration = -1

    while True:
        #print( "Iteration: ", iteration )
        iteration += 1
    
        # The latest row and column of the partial sums
        # are the full sums of the rows and columns
        sums_of_rows = horizontal_partial_sums[:, -1]
        sums_of_cols = vertical_partial_sums[-1, :]
        
        #print_array( sums_of_rows, "Sums of the rows" ) 
        #print_array( sums_of_cols, "Sums of the columns" ) 
    
        # Check if the requirements are met
        b_solution_is_valid = is_requirement_met( matrix_solution, sums_of_rows, sums_of_cols )
        #print( "b_solution_is_valid: ", b_solution_is_valid )
    
        if b_solution_is_valid == True:
            print( "\n\n\n" )
            #print_matrix( matrix_solution, "Winner Matrix" )
            break
        
        # Do one iteration
        for row in range( num_of_towns ):
            for col in range ( num_of_towns ):
                u = matrix_neuron[ row, col ];
                
                #print( "\n\n\n\033[1;96m", 20 * "#", "[", row, "][", col, "]", 20 * "#", "\033[0m", end=" " )
    
                # Calculate the: H and H' values
                #    * horizontal_old_step, which includes values calculated for (T) iteration
                #    * horizontal_new_step, which includes values calculated for (T + 1) iteration
                horizontal_old_step, horizontal_new_step = \
                                get_horizontal_partial_sums( row, col )
        
    
                # Calculate the: V and V' values
                #    * vertical_old_step, which includes values calculated for (T) iteration
                #    * vertical_new_step, which includes values calculated for (T + 1) iteration
                vertical_old_step, vertical_new_step = \
                                get_vertical_partial_sums( row, col )
    
                # Calculate the S' based on H, H', V and V'
                hybrid_sum = horizontal_old_step + horizontal_new_step + \
                             vertical_old_step + vertical_new_step
        
                hybrid_sum_coef = eta * ( hybrid_sum - 2 )
                weight_coef = lambdaa * matrix_weights[ row, col ] *  np.exp( -iteration / tau )
                u_new = u - deltaT * ( hybrid_sum_coef + weight_coef )
    
                if False and ( col == 0 or col == 6 ):
                    print( "\n\n\ncol: ", col )
                    print( "hybrid_sum_coef: ", hybrid_sum_coef )
                    print( "weight_coef: ", weight_coef )
                    print( "u: ", u, "\t->", u_new )
    
                # Update the value of neuron
                matrix_neuron[ row, col ] = u_new
        
                # Pass through activation function to calculate new solution variable
                x_new = 1 / ( 1 + np.exp( -beta * u_new ) )
    
                # Update the value in solution matrix
                matrix_solution[ row, col ] = x_new
        
                # Update the H'_{i,j} element
                part_sum_prev = 0
                if col != 0:
                    part_sum_prev = horizontal_partial_sums[ row, col - 1 ]
        
                horizontal_partial_sums[ row, col ] = part_sum_prev + x_new
                
                # Update the V'_{i,j} element
                part_sum_prev = 0
                if row != 0:
                    part_sum_prev = vertical_partial_sums[ row - 1, col ]
        
                vertical_partial_sums[ row, col ] = part_sum_prev + x_new
    
                #if iteration % 5000 == 0 and row == 0 and col == 0:
                #    print_matrix( vertical_partial_sums, "Vertical partial sums" )
                #    print_matrix( horizontal_partial_sums, "Horizontal partial sums" )
                #array_title = "UPDATED Horizontal Partial Sums of " + str( row ) + "th column"
                #print_partial_array( horizontal_partial_sums[ row, : ], "UPDATED Horizontal Partial Sums", col ) 
    
                #array_title = "UPDATED Vertical Partial Sums of " + str( col ) + "th column"
                #print_partial_array( vertical_partial_sums[ :, col ], array_title , row ) 
    
    
    matrix_solution[0,0] = 0
    print_matrix( matrix_solution, "Final solution matrix" )
    # Winner takes all
    
    b_need_to_stop = False
    i = 0
    
    count = 0
    #print( "0 ->", end="")
    matrix_weights[ :, 0] =  matrix_weights[ :, 0] / p
    
    path_list = [0]
    path_length = 0
    while True:
        count += 1
        column_of_max = np.argmax( matrix_solution[i] )   
        matrix_solution[ i, : ] = 0 
        matrix_solution[ :, column_of_max ] = 0 
        matrix_solution[ i, column_of_max ] = 1
        
        
        path_length += matrix_weights[ i, column_of_max ] 
        i = column_of_max
        print( i, "->", end=" ")
        path_list.append( i )
    
        if i == 0:
            break
    
    print( "\n\n\n" )

    end_time = time.time()
    print( count, "steps" )
    print( "execution time: ", end_time - start_time )
    print( "iteration count: ", iteration )
    #print_matrix( matrix_solution, "Kabooooooooooooooooooom" )
    print( "\n\n", calculate_path_length( matrix_weights, path_list[1:] ) )
    sys.exit()
    string = "\n\n\n$$$$$$" + str(path_length) + "for" + str(iteration) + "steps"
    file.write( string )
    for item in path_list:
        file.write(str(item) + "  ")

    file.close()
