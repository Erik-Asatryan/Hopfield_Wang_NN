#!/usr/bin/python3 

import sys, random, math
import numpy as np
import tsplib95
import time
import copy


############################## PRINT FUNCTIONS ###############################

def print_int_matrix( matrix, matrix_name="" ):
    if matrix_name != "":
        print( "\n\n\033[1;93m", 30 * "#", matrix_name, 30 * "#", "\033[0m" )

    for row in matrix:
        print()
        for elem in row:
            print( "\033[1;92m", f"{elem:6}", "\033[0m", end=" " )
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

def is_requirement_met( row_sums, col_sums, all_sum ):
    if all_sum != len(row_sums):
        return False

    for i in range( len(row_sums) ):
        if( row_sums[i] != 1 or col_sums[i] != 1 ):
            return False

    # If the function didn't return to this point,
    # Then for all the elements the requirement is met
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
    
# Load the TSP problem
tsp_problem = tsplib95.load( "gr17.tsp" )
num_of_towns = len( list( tsp_problem.get_nodes() ) )

# Create an empty matrix of weights
matrix_weights = np.zeros( (num_of_towns, num_of_towns), dtype=float )

# Initialize the weight matrix from the loaded tsp_problem
for i in range( num_of_towns ):
    for j in range( num_of_towns ):
        edge = i , j
        matrix_weights[i,j] = tsp_problem.get_weight( *edge )

arr_solutions = [ 2, 11, 13, 6, 8, 9, 16, 14, 7, 1, 4, 5, 12, 3, 10, 15, 0 ]
print( calculate_path_length( matrix_weights, arr_solutions ) )
sys.exit()

max_value = np.max(matrix_weights)
for i in range( num_of_towns ):
    for j in range( num_of_towns ):
        matrix_weights[i,j] /= max_value

print_matrix( matrix_weights, "Matrix of weights" )

u_0 = 0.02
n = num_of_towns

u_init = u_0 / 2 * np.log( n - 1 )

# Initialize matrix of neurons
matrix_neurons_new = np.zeros( (num_of_towns, num_of_towns), dtype=float )

for row in range( num_of_towns ):
    for col in range( num_of_towns ):
        bias = np.random.uniform(-0.1, 0.1) * u_0
        matrix_neurons_new[ row, col ] = u_init + bias

print_matrix( matrix_neurons_new, "Matrix of neurons" )

# Initialize matrix of solutions
matrix_solutions_new = np.zeros( (num_of_towns, num_of_towns), dtype=float )

for row in range( num_of_towns ):
    for col in range( num_of_towns ):
        u = matrix_neurons_new[ row, col ]
        matrix_solutions_new[ row, col ] = 1 / 2 * ( 1 + np.tanh( u / u_0 ) )

#for i in range( num_of_towns ):
#    matrix_neurons_new[i,0] = 0.2 
#    matrix_solutions_new[i,0] = 0

A = 500
B = 500
C = 200
D = 500
deltaT = 0.00001
tau = 1

iteration = -1

stop_count = 0

while True:
    iteration += 1
    matrix_solutions = matrix_solutions_new
    matrix_neurons = matrix_neurons_new

    row_sums = np.zeros( (num_of_towns), dtype=float )
    col_sums = np.zeros( (num_of_towns), dtype=float )
    all_sum = 0

    for row in range( num_of_towns ):
        for col in range( num_of_towns ):
            if matrix_solutions[row,col] > 0.3:
                matrix_solutions[row,col] = 1
            else:
                matrix_solutions[row,col] = 0

    for row in range( num_of_towns ):
        for col in range( num_of_towns ):
            row_sums[row] += matrix_solutions[ row, col ]
            col_sums[col] += matrix_solutions[ row, col ]
            all_sum += matrix_solutions[ row, col ]

    if( is_requirement_met( row_sums, col_sums, all_sum ) ):
        print( "Solution found after", iteration, "iterations" )
        print_matrix( matrix_neurons_new, "Final matrix of neurons" )
        print_matrix( matrix_solutions_new, "Final matrix of solution" )
        break

    for row in range( num_of_towns ):
        for col in range( num_of_towns ):
            col_sum = col_sums[col] - matrix_solutions[ row, col ]
            row_sum = row_sums[row] - matrix_solutions[ row, col ]

            dont_know_sum = 0
            for y in range( num_of_towns ):
                if y == row:
                    continue
                dont_know_sum += matrix_weights[ row, y ] * matrix_solutions[ y, col] * ( matrix_solutions[ y, ( col + 1 ) % num_of_towns ] + \
                                                              matrix_solutions[ y, col - 1 ] )

            sigma = -A * col_sum - B * row_sum - C * ( all_sum - num_of_towns ) - D * dont_know_sum

            u = matrix_neurons[ row, col ]
            if row == col:
                u -= 10000


            u_new = u + deltaT * ( -u / tau + sigma )

            solution_new = 1 / 2 * ( 1 + np.tanh( u_new / u_0 ) )
            
            matrix_neurons_new[ row, col ] = u_new 
            matrix_solutions_new[ row, col ] = solution_new 

            ##if ( False and iteration == 1 or iteration % 5 == 0 ) and row == col and row == num_of_towns - 1:
            ##    print( iteration )
            ##    print( "\n\n\033[1;93m", iteration, "\033[0m" )
            ##    print_matrix( matrix_neurons_new, "Matrix of neurons" )
            ##    print_matrix( matrix_solutions_new, "Matrix of solution" )

def find_index_of_first_non_zero( arr, n ):
    for i in range(n):
        if arr[i] != 0:
            return i

    return -1

row_index = 0
for i in range(num_of_towns):
    print( row_index, end=" -> ")
    row_index = find_index_of_first_non_zero( matrix_solutions_new[row_index], num_of_towns )
