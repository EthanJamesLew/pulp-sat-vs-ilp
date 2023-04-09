"""Schema Models for Benchmarks"""
from pydantic import BaseModel
from typing import Any, Dict


class SolverResult(BaseModel):
    """fields collected by running a PuLP solver"""

    solver: str
    solve_time: float
    model_status: int


class ExperimentResult(BaseModel):
    """fields collected by running a benchmark"""

    model: str
    solver_result: SolverResult
    parameter_values: Dict[str, Any]
