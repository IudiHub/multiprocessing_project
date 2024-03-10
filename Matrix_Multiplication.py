#!/usr/bin/env python
import multiprocessing as mp
import numpy as np
import sys

## FUNCTIONS DEFINITION

# 1) check_input_values function checks whether the command line arguments that are provided by the user are valid and within the expected format
#    The function returns all the validated input arguments
def check_input_values():
    # read command line arguments  
    if len(sys.argv) < 5:
        raise Exception("You are missing at least one input argument: \nUsage: python matrix_multiplication_test.py <matrices_size> <scalar> <num_of_processes> <condition_unmet_to_test>")

    non_valid_inputs = []
    try:
        N = int(sys.argv[1])
    except ValueError:
        non_valid_inputs.append('<matrices_size>')
        
    try:
        c = float(sys.argv[2])
    except ValueError:
        non_valid_inputs.append('<scalar>')
        
    try:
        num_processes = int(sys.argv[3])
    except ValueError:
        non_valid_inputs.append('<num_of_processes>')
    
    condition_unmet_to_test = True 
    if sys.argv[4].lower() == 'false':
        condition_unmet_to_test = False
    else: 
        if sys.argv[4].lower() != 'true':
            non_valid_inputs.append('<condition_unmet_to_test>')
            
    if non_valid_inputs: 
        non_valid_inputs_length_g_1 = len(non_valid_inputs) > 1
        print(', '.join(non_valid_inputs), "arguments are not valid inputs" if non_valid_inputs_length_g_1 else "argument is not a valid input")
        print("Try to fix the above non valid input"+("s" if non_valid_inputs_length_g_1 else "") +" and relaunch the script")
        sys.exit(1)
    
    return N, c, num_processes, condition_unmet_to_test
 
# 2) create_matrices function generates an array of 10 random square matrices of size NxN and a second array obtained by element-wise
# multiplication with a scalar factor c
# The function returns the two arrays of matrices
def create_matrices(N, c):
    matrices_A = np.array([np.random.rand(N, N) for _ in range(10)])
    matrices_B = matrices_A * c
    return matrices_A, matrices_B
    
# 3) test_matrices_product_condition_single_process function is designed to test whether AiBi=BiAi using the NumPy library, iterating over the 10 matrices
#    If the condition is not met for any of the iterations, the function will raise an assertion error
def test_matrices_product_condition_single_process(matrices_A, matrices_B):  
    for A, B in zip(matrices_A, matrices_B):
        assert np.allclose(np.dot(A, B), np.dot(B, A))

# 4) create_chunk_coordinates function is used to generate chunk coordinates that are used to break down each matrix multiplication into smaller chunks
#    to distribute the work among processes
#    The function returns a list of tuples containing the coordinates of each chunk
def create_chunk_coordinates(N, chunk_size):
    chunk_coordinates = []
    for i in range(0, N, chunk_size):
        # calculate the end row index of the current chunk: its maximum value can be N
        i_end = i + chunk_size if i + chunk_size <= N else N
        for j in range(0, N, chunk_size):
            # calculate the end column index of the current chunk: its maximum value can be N
            j_end = j + chunk_size if j + chunk_size <= N else N
            # coordinates that define each chunk in the 2 matrices that have to be multiplied
            chunk_coordinates.append((i, i_end, j, j_end))
    return chunk_coordinates

# 5) create_worker_process_args function is used to create the arguments that are passed to the worker processes
#    The function returns two lists of tuples containing the arguments for AB and BA multiplications
def create_worker_process_args(num_processes, matrix_pairs, chunk_coordinates, condition_unmet_to_test):
    results = [] 
    num_matrix_pairs = len(matrix_pairs)
    # distribute the matrix multiplication tasks among the available processes (num_processes)
    for process_index in range(num_processes):
        for i in range(process_index, num_matrix_pairs, num_processes):
            if process_index >= num_matrix_pairs:
                break
            A, B = matrix_pairs[i]
            for i_start, i_end, j_start, j_end in chunk_coordinates:
                results.append((i_start, i_end, j_start, j_end, A, B, i, condition_unmet_to_test))
    return results

# 6) worker_process is a worker function to perform matrix multiplication on submatrices.
#    The function returns a tuple containing the result of the matrix multiplication and the index of the matrices pair
#    NOTE-1: the size of the sumbmatrices (i.e., the indices i_start, i_end, j_start, j_end) depends on the number of processes you are considering to perform multiprocessing
#    NOTE-2: if the user specifies to test the condition AB = BA only when it is not met, the worker process will modify the value of the element in the first row and the first column of the BA matrix chunk
def worker_process(args):
    i_start, i_end, j_start, j_end, A, B, i, condition_unmet_to_test = args
    AB_chunk = np.dot(A[i_start:i_end], B[:, j_start:j_end])
    BA_chunk = np.dot(B[i_start:i_end], A[:, j_start:j_end])
    if condition_unmet_to_test and i % 2 == 0:
            BA_chunk[0][0] = 0
    return np.allclose(AB_chunk, BA_chunk), i

if __name__ == '__main__':
    # validate input arguments
    N, c, num_processes, condition_unmet_to_test = check_input_values()
    
    # create the 10 matrices A and B using the create_matrices function
    matrices_A, matrices_B = create_matrices(N, c)
    
    ## SINGLE PROCESS IMPLEMENTATION
    
    test_matrices_product_condition_single_process(matrices_A, matrices_B)
    
       
    print("Version without multiprocessing:")
    print("The condition AB = BA is met")
    
    ## MULTIPROCESSING IMPLEMENTATION
    
    # create a pool of worker processes 
    # This pool will be used to distribute and execute tasks in parallel across multiple processes 
    # NOTE: num_processes (i.e., the number of worker processes in the pool) is defined by the user
    pool = mp.Pool(processes=num_processes)
    
    # calculate chunk size that is used divide the matrices into 
    # smaller chunks for parallel processing:
    # N // num_processes is an integer division that returns the quotient of the division without the remainder 
    # the max() function is used to ensure that the chunk_size is at least 1
    chunk_size = max(1, N // num_processes)

    # create pair of matrices for processing:
    # this pairing ensures that the correct matrices are used together when distributing work among different worker processes
    matrix_pairs = list(zip(matrices_A, matrices_B))
    num_matrix_pairs = len(matrix_pairs)

    # generate chunk coordinates: you need to break down each matrix multiplication into smaller chunks 
    # to distribute the work among processes
    chunk_coordinates = create_chunk_coordinates(N, chunk_size)  # chunk_coordinates is a list of tuples

    # create arguments for worker processes containing the coordinates of each chunk and the matrices to be multiplied
    worker_process_tuples = create_worker_process_args(num_processes, matrix_pairs, chunk_coordinates, condition_unmet_to_test)

    # distribute tasks among worker processes in the pool and collect the results
    chunk_results = pool.map(worker_process, worker_process_tuples)

    # ensure that the main program waits for all tasks to finish processing before moving on, preventing resource conflicts
    # and unexpected behaviors
    pool.close()
    pool.join()    
    
    print("\nMultiprocessing version:")    
    false_indices = set(i for is_false, i in chunk_results if not is_false)
    if len(false_indices) == 0:
        print("The condition AB = BA is met")
    else: 
        print("The condition AB = BA is not met for matrices with indices:", sorted(false_indices))


    
