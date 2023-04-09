"""Benchmark for nqueens problem.

This benchmark runs the nqueens problem with different solvers and different
board sizes. The board sizes are chosen logarithmically, so that the problem
becomes more difficult as the board size increases.
"""

import pulp
import numpy as np
from benchutils.cli import *


def build_nqueens_model(n):
    queens = pulp.LpProblem("queens", pulp.LpMinimize)

    x = [
        [pulp.LpVariable("x({},{})".format(i, j), 0, 1, "Binary") for j in range(n)]
        for i in range(n)
    ]

    # one per row
    for i in range(n):
        queens += pulp.lpSum(x[i][j] for j in range(n)) == 1, "row({})".format(i)

    # one per column
    for j in range(n):
        queens += pulp.lpSum(x[i][j] for i in range(n)) == 1, "col({})".format(j)

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens += pulp.lpSum(
            x[i][j] for i in range(n) for j in range(n) if i - j == k
        ) <= 1, "diag1({})".format(p)

    for p, k in enumerate(range(1, n + n - 2)):
        queens += pulp.lpSum(
            x[i][j] for i in range(n) for j in range(n) if i + j == k
        ) <= 1, "diag2({})".format(p)

    return queens


if __name__ == "__main__":
    app = BenchmarkCliApp(
        ModelBenchmark(
            "nqueens",
            ModelParameters(
                ["board_size"],
                [int],
                [np.logspace(1, np.log10(100), 10, dtype=int).tolist()],
            ),
            build_nqueens_model,
        )
    )

    app.run()
