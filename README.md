📊 PageRank Algorithm - Numerical Linear Algebra Project
📝 Project Overview

This project implements the PageRank algorithm, the fundamental algorithm used by search engines to rank web pages. It explores the connection between Graph Theory, Markov Chains, and Numerical Linear Algebra.

The goal is to compute the importance of nodes in a network by finding the stationary distribution of a random walk.
🚀 Key Features

    Matrix Computation: Efficient handling of large-scale transition matrices.

    Power Method Implementation: Iterative approach to find the dominant eigenvector (the PageRank vector).

    Handling Dead Ends: Implementation of teleportation and damping factors (

            
    α
    α

          

    ) to solve the "spider trap" and "dead end" problems.

    Data Visualization: Graphs showing how rankings evolve over iterations.

🛠 Tech Stack

    Language: Python

    Libraries: NumPy (Matrix operations), SciPy (Sparse calculations), Matplotlib (Visualization).

🧮 Mathematical Background

The project focuses on solving the equation:

        
v=αMv+(1−α)1ne
v=αMv+(1−α)n1​e

      


Where:

            
    M
    M

          

    is the transition matrix.

            
    α
    α

          

    is the damping factor (usually 0.85).

            
    v
    v

          

    is the PageRank vector.