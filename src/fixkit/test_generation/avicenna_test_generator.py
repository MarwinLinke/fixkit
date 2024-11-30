import os
from pathlib import Path
from typing import List, Optional, Callable

from fixkit.constants import DEFAULT_WORK_DIR
from fixkit.test_generation.test_generator import TestGenerator
from fixkit.logger import LOGGER

from avicenna.core import Grammar
from avicenna.data import OracleResult
from avicenna import Avicenna
from avicenna.runner.report import SingleFailureReport
from avicenna.diagnostic import Candidate

from isla.solver import ISLaSolver, _DEFAULTS
from isla.language import ISLaUnparser, Formula, parse_isla



class AvicennaTestGenerator(TestGenerator):

    def __init__(
        self,
        oracle: Callable,
        grammar: Grammar,
        initial_inputs: List[str],
        max_iterations: int,
        out: Optional[os.PathLike] = None,
        saving_method: Optional[str] = None,
        save_automatically: Optional[bool] = True,
        identifier: Optional[str] = None,
    ):
        """
        Initialize the test generator
        :param os.PathLike src: The source directory of the project.
        :param Callable oracle: The oracle used for labeling inputs.
        :param Grammar grammar: The grammar used in Avicenna.
        :param List[str] initial_inputs: The initial inputs required to run Avicenna, at least one passing and one failing one.
        :param int max_iterations: The number of iterations.
        :param Optional[os.PathLike] out: The path location for saving labeled inputs.
        :param Optional[str] saving_method: Use "json" to save inputs inside json files or "files" separate text files for each input.
        :param Optional[bool] save_automatically:  If true, test cases are automatically saved after running. Alternatively, use save_test_cases() with a given path. 
        :param Optional[str] identifier: Is used for saving and loading formulas generated through avicenna.
        """
        super().__init__(
            out=Path(out or DEFAULT_WORK_DIR, "avicenna"),
            saving_method=saving_method,
            save_automatically=save_automatically
            )

        self.oracle = oracle
        self.grammar = grammar
        self.initial_inputs = initial_inputs
        self.max_iterations = max_iterations
        self.identifier = identifier or "formula"

        self.avicenna = Avicenna(
            grammar = self.grammar, 
            oracle = self.oracle, 
            initial_inputs = self.initial_inputs,
            max_iterations = self.max_iterations,
            report = SingleFailureReport()
            )
        
        self.failing = []    
        self.passing = []
        self.diagnoses = None

    def _save_formula(self) -> str:
        """
        Saves formula after running avicenna. Can be found under self.out / "formulas" / self.identifier.
        """
        dir = Path(self.out) / "formulas"
        dir.mkdir(parents=True, exist_ok=True)
        file_path = dir / self.identifier

        formula = self.diagnoses[0].formula     
        formula_string = ISLaUnparser(formula).unparse()

        with file_path.open("w") as f:
            f.write(formula_string)
        
        return file_path

    def load_formula(self, identifier: str) -> str:
        """
        Loads formula from self.out / "formulas" / self.identifier.
        """
        dir = Path(self.out) / "formulas"
        file_path = dir / identifier
        if not file_path.exists():
            LOGGER.info(f"No cached formula found at {dir}")
            return None

        with file_path.open("r") as f:
            formula = f.read()

        return formula
    
    def run(self, save_inputs: bool = True):
        """
        Executes Avicenna with given parameters and saves results in out directory.
        """
        self.diagnoses: List[Candidate] = self.avicenna.explain()

        failing = self.avicenna.report.get_all_failing_inputs()
        passing = self.avicenna.report.get_all_passing_inputs()

        file_path = self._save_formula()
        
        LOGGER.info(f"Saved formula under {file_path}.")
        LOGGER.info(f"Avicenna generated {len(failing)} failing and {len(passing)} passing inputs.")

        if save_inputs:
            self.failing = failing
            self.passing = passing
            self._save_inputs()

    def solve_formula(
        self, 
        max_iterations: int, 
        negate_formula: bool = False,
        formula: Formula = None,
        only_unique_inputs: bool = False,
        optimized_queries: bool = False
    ):
        """
        Solves formula for more inputs. If no formula is specified, takes diagnosis from last run of this TestGenerator.
        """
        if formula:
            failure_formula = parse_isla(formula, self.grammar, _DEFAULTS.structural_predicates, _DEFAULTS.semantic_predicates)
        elif self.diagnoses:
            failure_formula = self.diagnoses[0].formula      
        else:
            LOGGER.info("No diagnosis or formula was found.")
            return

        passing: List[str] = []
        failing: List[str] = []
        undefined: List[str] = []

        solver = ISLaSolver(
            grammar = self.grammar,
            formula = -failure_formula if negate_formula else failure_formula,
            enable_optimized_z3_queries = optimized_queries)       
        
        i = 0
        isla_restart = 0
        fail_safe = 0
        while i < max_iterations and isla_restart < 100:
            try:      
                inp = solver.solve()            
                oracle_result, _ = self.oracle(inp)

                if fail_safe >= 50:
                    fail_safe = 0
                    raise StopIteration

                if (only_unique_inputs and (str(inp) in passing or str(inp) in failing)):
                    fail_safe += 1
                    continue

                if oracle_result == OracleResult.PASSING:
                    passing.append(str(inp))
                elif oracle_result == OracleResult.FAILING:
                    failing.append(str(inp))
                else:
                    undefined.append(str(inp))
                    fail_safe += 1
                    continue

                i += 1
                if i % 10 == 0:
                    LOGGER.info(f"ISLaSolver generated {len(failing)} failing and {len(passing)} passing inputs so far.")

            except StopIteration:

                isla_restart += 1 
                if isla_restart % 10 == 0:
                    LOGGER.info(f"ISLaSolver was restarted {isla_restart} times (max 100). Generated {len(failing)} failing and {len(passing)} passing inputs so far.")

                solver = ISLaSolver(
                    grammar = self.grammar,
                    formula = -failure_formula if negate_formula else failure_formula,
                    enable_optimized_z3_queries = optimized_queries)       
                continue

        if only_unique_inputs:
            passing = list(set(passing))
            failing = list(set(failing))

        self.passing.extend(passing)
        self.failing.extend(failing)

        unique = " unique" if only_unique_inputs else ""
        LOGGER.info(f"ISLaSolver generated {len(failing)}{unique} failing and {len(passing)}{unique} passing inputs.")
        if undefined:
            LOGGER.info(f"ISLaSolver generated {len(undefined)} undefined inputs.")

        self._save_inputs()