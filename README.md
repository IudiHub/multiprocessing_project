# Matrix Multiplication Test Documentation

This document provides an overview and usage guide for the Matrix Multiplication Test script. The script is designed to test the validity of matrix multiplication between two sets of matrices and to compare the results of matrix multiplication `AB` and `BA`. The script includes both single process and multiprocessing implementations to potentially accelerate the computation.

## Table of Contents

- [Introduction](#introduction)
- [Function Definitions](#function-definitions)
- [Usage](#usage)
  - [Command Line Arguments](#command-line-arguments)
- [Single Process Implementation](#single-process-implementation)
- [Multiprocessing Implementation](#multiprocessing-implementation)
- [Results](#results)

## Introduction

The Matrix Multiplication Test script is developed to determine whether the matrix multiplication condition `AB = BA` holds true for given sets of matrices `A` and `B`. `A` are square matrices of size NxN and `B` are obtained through the equation: `B = cA`, where `c` is a scalar. The script provides two implementations: a single-process version and a multi-process version using the `multiprocessing` library.

## Function Definitions

1. **`check_input_values()`**:
   checks the validity of command line arguments provided by the user. Ensures that the required input arguments are correctly formatted. The function returns all the validated input arguments.

2. **`create_matrices(N, c)`**:
   generates an array of 10 random square matrices of size NxN and a second array obtained by element-wise multiplication with a scalar factor `c`. The function returns the two arrays of matrices.
   
3. **`test_matrices_product_condition_single_process(matrices_A, matrices_B)`**:
   this function tests the condition `AiBi = BiAi` holds true for each pair of matrices (i=1, 2,...,10). It raises an assertion error if the condition is not met.

4. **`create_chunk_coordinates(N, chunk_size)`**:
   is used to generate chunk coordinates that are used to break down each matrix multiplication into smaller chunks to distribute the work among processes. The function returns a list of tuples containing the coordinates of each chunk.
   
5. **`create_worker_process_args(num_processes, matrix_pairs, chunk_coordinates, condition_unmet_to_test)`**:
  creates arguments that are passed to worker processes, specifying which chunks of matrices they will operate on. It returns a list of tuples: each tuple includes information about matrix pairs indices, chunk coordinates, matrices to be multiplied and the argument that controls if you want to test when the hypothesis is not met.
  
6. **`worker_process(args)`**: 
   performs matrix multiplication on submatrices and returns whether the condition is met for the specified submatrix.
    
   **NOTE 1:** the size of the sumbmatrices (i.e., the indices i_start, i_end, j_start, j_end) depends on the number of processes you are considering to perform multiprocessing.
   
   **NOTE 2:** if the user specifies to test the condition AB = BA when it is not met (i.e., `condition_unmet_to_test` is set to `true` in the command line), the worker process will modify the value of the element in the first row and the first column of the BA matrix chunk.

## Usage

### Command Line Arguments

The script is intended to be run from the command line. The following command line arguments must be provided:

1. `matrices_size`: the size `N` of the square matrices.
2. `scalar`: a scalar factor `c` for element-wise multiplication (`B = cA`).
3. `num_of_processes`: the number of worker processes for parallel computation.
4. `condition_unmet_to_test`: when set to `'true'`, this parameter enables testing a scenario where the matrix multiplication condition `AB = BA` is intentionally not met for even-indexed matrices. This helps to assess the behavior of the script when the standard product equality condition is deliberately not satisfied. When set to `'false'`, the script tests the standard condition where the matrix multiplication equality holds true for all matrix pairs.


Example usage: `python script_name.py 100 2.0 4 true`

## Single Process Implementation

The script starts by validating input arguments and generating sets of matrices `A` and `B` using the `create_matrices` function. It then tests the matrix multiplication condition `AB = BA` using the `test_matrices_product_condition_single_process` function for each matrix pair.

## Multiprocessing Implementation

The script employs a multiprocessing approach to speedup matrix multiplication. It creates a pool of worker processes and divides the matrix multiplication tasks into chunks. The tasks are distributed among the processes using the `pool.map()` function. 

### How Multiprocessing works

- A pool of worker processes is created using the `multiprocessing.Pool` class, based on the user defined number of processes.
- The matrices are split into smaller sections to enable multiprocessing. The script calculates the appropriate chunk size.
- The matrix pairs and chunk coordinates are prepared, and arguments for worker processes are generated.
- Multiprocessing occurs as worker processes simultaneously execute matrix multiplication tasks using the `worker_process` function.
Results from the processes are collected, and the matrix multiplication condition is evaluated.
- The script closes and joins the worker processes.
- The output indicates whether the condition is met for the entire set of matrices.


## Results

The output of the Matrix Multiplication Test script provides valuable insights into the validity of the matrix multiplication condition for the given set of matrices. Depending on the implementation chosen, the script presents the results in the following manner:

### Single Process Implementation

In the version without multiprocessing, the script iterates over each matrix pair and checks whether the matrix multiplication condition AB = BA holds true. If the condition is met for all iterations, the script displays the message:
```
Version without multiprocessing:
The condition AB = BA is met
```

This outcome indicates that the matrices exhibit the commutative property of matrix    multiplication, satisfying the equation AB = BA for all tested cases.

### Multiprocessing Implementation

The output in this case is interpreted as follows:

- If the output indicates that the condition is met for all matrix pairs and chunks, the script displays the message:
```
Multiprocessing version:
The condition AB = BA is met
```
This result indicates that the matrix multiplication condition AB = BA is satisfied across all tested combinations.

- If the output indicates that the condition is not met for specific matrix pairs and chunks, the script lists the indices of the matrices that deviate from the expected behavior. For example:

```
Multiprocessing version:
The condition AB = BA is not met for matrices with indices: [2, 4, 6, 8]
```
In this case, the script identifies the matrix pairs with indices 2, 4, 6, and 8 that do not satisfy the matrix multiplication condition AB = BA.

---
For additional information and usage instructions, please refer to the provided code documentation and comments.

**Author**: [Cristina Iudica]
**Last Update**: [27th August 2023]
