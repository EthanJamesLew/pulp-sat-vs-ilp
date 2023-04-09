"""Command line interface app for benchmarking"""
import itertools
from pathlib import Path
from typing import Callable, Any
import pulp
import benchutils.model


class ModelParameters:
    """A set of parameters for a benchmark model"""

    def __init__(
        self,
        parameter_names: list[str],
        parameters_cls: list[Any],
        parameters_slices: list[list[Any]],
    ):
        assert len(parameter_names) == len(parameters_cls)
        assert len(parameters_slices) == len(parameters_cls)
        self.parameter_names = parameter_names
        self.parameters_cls = parameters_cls
        self.parameters_slices = parameters_slices

    def validate(self, parameters: list[Any]):
        assert len(parameters) == len(self.parameter_names)
        for p, pname, pcls in zip(
            parameters, self.parameter_names, self.parameters_cls
        ):
            assert isinstance(
                p, pcls
            ), f"parameter '{pname}' value {p} is not of type {pcls} (type is {type(p)})"

    def __iter__(self):
        for parameters in itertools.product(*self.parameters_slices):
            yield parameters

    def __repr__(self) -> str:
        return f"""ModelParameters(parameter_names={self.parameter_names}, parameters_cls={self.parameters_cls})"""


class ModelBenchmark:
    """A benchmark for a PuLP model"""

    def __init__(
        self,
        name: str,
        model_parameters: ModelParameters,
        model_generator: Callable[[Any], pulp.LpProblem],
    ):
        self.model_parameters = model_parameters
        self.model_generator = model_generator
        self.name = name

    def generate_model(self, parameters) -> pulp.LpProblem:
        self.model_parameters.validate(parameters)
        return self.model_generator(*parameters)

    def __repr__(self) -> str:
        return f"""ModelBenchmark(name={self.name}, model_parameters={self.model_parameters})"""


class BenchmarkCliApp:
    @property
    def description(self) -> ModelBenchmark:
        lines = [f"Benchmarking App for {self.benchmark.name}"]
        lines.append("Parameters: [")
        for pname, pcls in zip(
            self.benchmark.model_parameters.parameter_names,
            self.benchmark.model_parameters.parameters_cls,
        ):
            lines.append(f"  {pname} ({pcls}),")
        lines.append("]")
        return "\n".join(lines)

    def __init__(self, benchmark: ModelBenchmark):
        self.benchmark = benchmark

    def make_args(self):
        import argparse

        parser = argparse.ArgumentParser(description=self.description)
        parser.add_argument("-t", "--timeout", type=int, default=1000)
        parser.add_argument("-o", "--output_dir", type=str, default=".")
        parser.add_argument("-s", "--solver", type=str, default="PULP_CBC_CMD,CPSAT_PY")
        return parser.parse_args()

    def run(self):
        args = self.make_args()
        output_path = Path(args.output_dir)

        solver_names = args.solver.split(",")
        for idx, (solver_name, mparameters) in enumerate(
            itertools.product(solver_names, self.benchmark.model_parameters)
        ):
            model = self.benchmark.generate_model(mparameters)
            solver = pulp.getSolver(solver_name)
            status = model.solve(solver)
            solver_result = benchutils.model.SolverResult(
                solver=solver_name, solve_time=model.solutionTime, model_status=status
            )

            exp_result = benchutils.model.ExperimentResult(
                parameter_values={
                    pname: pvalue
                    for pname, pvalue in zip(
                        self.benchmark.model_parameters.parameter_names, mparameters
                    )
                },
                model=self.benchmark.name,
                solver_result=solver_result,
            )

            # write the results to disk
            with open(
                output_path / f"{self.benchmark.name}-{solver_name}-{idx}.json", "w"
            ) as fp:
                fp.write(exp_result.json())
