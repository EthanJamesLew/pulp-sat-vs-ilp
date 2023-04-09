import argparse
import pulp
from pydantic import BaseModel
import numpy as np


class ExperimentResult(BaseModel):
    solver: str
    model: str
    solve_time: float
    model_status: int


def build_nqueens_model(n):
    queens = pulp.LpProblem('queens', pulp.LpMinimize)

    x = [[pulp.LpVariable('x({},{})'.format(i, j), 0, 1, 'Binary')
          for j in range(n)] for i in range(n)]

    # one per row
    for i in range(n):
        queens += pulp.lpSum(x[i][j] for j in range(n)) == 1, 'row({})'.format(i)

    # one per column
    for j in range(n):
        queens += pulp.lpSum(x[i][j] for i in range(n)) == 1, 'col({})'.format(j)

    # diagonal \
    for p, k in enumerate(range(2 - n, n - 2 + 1)):
        queens += pulp.lpSum(x[i][j] for i in range(n) for j in range(n)
                        if i - j == k) <= 1, 'diag1({})'.format(p)
    
    for p, k in enumerate(range(1, n + n - 2)):
        queens += pulp.lpSum(x[i][j] for i in range(n) for j in range(n)
                        if i + j == k) <= 1, 'diag2({})'.format(p)

    return queens


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--max', type=int, default=100)
    parser.add_argument('-n', '--n_experiments', type=int, default=10)
    parser.add_argument('-t', '--timeout', type=int, default=1000)
    parser.add_argument('-o', '--output_dir', type=str, default='.')
    return parser.parse_args()


def run_experiment(n, solver_name, timeout) -> ExperimentResult:
    solver = pulp.getSolver(solver_name)
    model = build_nqueens_model(n)
    status = model.solve(solver)
    return ExperimentResult(
        solver=solver_name,
        model='nqueens',
        solve_time=model.solutionTime,
        model_status=status
    )

def main(n_max: int, n_experiments: int, timeout: int, output_dir: str):
    board_sizes = np.logspace(1, np.log10(n_max), n_experiments, dtype=int)

    for solver_name in ['PULP_CBC_CMD', 'CPSAT_PY']:
        for n in board_sizes:
            result = run_experiment(n, 'PULP_CBC_CMD', timeout) 
            with open(f"nqueens-{solver_name}-{n}.json", "w") as fp:
                fp.write(result.json())


if __name__ == '__main__':
    args = get_args()
    main(args.max, args.n_experiments, args.timeout, args.output_dir)