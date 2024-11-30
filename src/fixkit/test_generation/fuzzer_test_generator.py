import os
from pathlib import Path
from typing import List, Optional, Callable

from fixkit.test_generation.test_generator import TestGenerator
from fixkit.constants import DEFAULT_WORK_DIR
from fixkit.logger import LOGGER

from avicenna.core import Grammar
from avicenna.data import OracleResult
from isla.fuzzer import GrammarFuzzer

class GrammarFuzzerTestGenerator(TestGenerator):

    def __init__(
        self,
        oracle: Callable,
        grammar: Grammar,
        num_failing: int,
        num_passing: int,
        max_iterations: int = 20000,
        out: Optional[os.PathLike] = None,
        saving_method: Optional[str] = None,
        save_automatically: Optional[bool] = True,
    ):
        """
        Initialize the test generator
        :param Callable oracle: The oracle used for labeling inputs.
        :param Grammar grammar: The grammar used in the grammar fuzzer.
        :param int num_failing: The number of failing test cases the fuzzer aims to generate.
        :param int num_passing: The number of passing test cases the fuzzer aims to generate.
        :param int generation_limit: The max number of iterations the fuzzer will perform. Use it as a fail-safe.
        :param Optional[os.PathLike] out: The path location for saving labeled inputs.
        :param Optional[str] saving_method: Use "json" to save inputs inside json files or "files" separate text files for each input.
        :param Optional[bool] save_automatically: If true, test cases are automatically saved after running. Alternatively, use save_test_cases() with a given path. 
        """

        super().__init__(
            out=Path(out or DEFAULT_WORK_DIR, "grammar_fuzzer"),
            saving_method=saving_method,
            save_automatically=save_automatically
            )

        self.oracle = oracle
        self.grammar = grammar
        self.num_failing = num_failing
        self.num_passing = num_passing
        self.max_iterations = max_iterations

        self.failing = []    
        self.passing = []
    
    def run(self):
        """
        Execute GrammarFuzzer with parameter and save results in out directory.
        """
        passing_count = 0
        failing_count = 0

        failing_inputs: List[str] = []
        passing_inputs: List[str] = []

        fuzzer = GrammarFuzzer(self.grammar)
        iteration = 0

        while iteration < self.max_iterations:

            inp = fuzzer.fuzz()
            oracle_result, _ = self.oracle(inp)
            iteration += 1

            if iteration % 10 == 0:
                LOGGER.info(f"Found {len(failing_inputs)} failing and {len(passing_inputs)} passing inputs in {iteration} iterations")

            if oracle_result == OracleResult.FAILING and inp not in failing_inputs:
                if failing_count >= self.num_failing:
                    continue

                failing_inputs.append(inp)
                failing_count += 1

            elif oracle_result == OracleResult.PASSING and inp not in passing_inputs:
                if passing_count >= self.num_passing:
                    continue

                passing_inputs.append(inp)
                passing_count += 1

            if failing_count >= self.num_failing and passing_count >= self.num_passing:
                break

        passing_inputs = list(set(passing_inputs))
        failing_inputs = list(set(failing_inputs))

        LOGGER.info(f"Grammar fuzzer found {len(failing_inputs)} failing and {len(passing_inputs)} passing inputs in {iteration} iterations.")

        self.passing = passing_inputs
        self.failing = failing_inputs

        self._save_inputs()