# SAT vs ILP

![nqueens benchmark example](docs/img/nqueens.svg)

General purpose mixed integer linear programming solvers (e.g., gurobi and CPLEX) are frequently used to solve satisfiability problems, i.e., constraint problems with no optimization objective. This repository contains experiments comparing common ILP solvers against common SAT solvers *for SAT problems* to determine how much faster these tailored solvers are. The goals are
* benchmark these tools against *common*, and next *realistic*, sat benchmarks
* implement and test sat backends for the Python PuLP modeler. These implementations should be general purpose enough to be useful contributions to the community.

## Benchmarks

First, we look at common benchmarks given for PuLP, being
* $n$-Queens: $n$-queens problem with $n$ queens on an $n \times n$ chessboard.
* Sudoku: Sudoku problem with $n$ rows, $n$ columns, and $n$ cells.

## Solvers

* OR-Tools CP-SAT: [Google's Open-Source Constraint Programming Solver](https://developers.google.com/optimization/cp/cp_solver)
* Z3: [Microsoft's Open-Source SMT Solver](https://github.com/Z3Prover/z3)
* Why3 (TODO): [Interface to many SMT solvers](https://why3.lri.fr/)