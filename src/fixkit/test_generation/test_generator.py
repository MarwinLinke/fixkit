import abc
from abc import ABC
import os
import json
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable
from fixkit.constants import DEFAULT_WORK_DIR
from avicenna.core import Input, Grammar
from avicenna.data import OracleResult
from avicenna import Avicenna
from avicenna.runner.report import SingleFailureReport
from avicenna.diagnostic import Candidate
from isla.language import Formula
from isla.solver import ISLaSolver
import logging
import shutil

LOGGER = logging.getLogger("fixkit")
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s :: %(levelname)-8s :: %(message)s",
)


class TestGenerator(ABC):

    def __init__(
        self,
        out: Optional[os.PathLike] = None,
        saving_method: Optional[str] = None,
        overwrite: Optional[bool] = False
    ):
        """
        Initialize the test generator
        :param Optional[os.PathLike] out: The path location for saving labeled inputs.
        :param Optional[str] saving_method: Saves passing and failing tests in json files or separate text files.
        """
    
        self.out = Path(out or DEFAULT_WORK_DIR)

        self.saving_method = saving_method or "files" 
        if self.saving_method not in ["json", "files"]:
            raise ValueError('Invalid argument. Use either "json" or "files".')
        
        self.overwrite = overwrite
        
        self.failing = None
        self.passing = None
    
    @abc.abstractmethod
    def run(self):
        """
        Abstract method for running the test generator.
        """
        pass

    def _save_inputs(self):
        if self.saving_method == "json":
            self._save_as_json()
        elif self.saving_method == "files":
            self._save_as_files()

        LOGGER.info(f"Saved {len(self.failing) + len(self.passing)} test cases under {self.out}")


    def _save_as_files(self):
        """
        Saves each input from self.passing and self.failing as separate text files.
        Files are saved in the output directory as:
        - passing_test_X.txt
        - failing_test_X.txt
        """

        dir = self.out
        if self.overwrite:
            shutil.rmtree(dir)
        dir.mkdir(parents=True, exist_ok=True)

        for idx, test in enumerate(self.passing):
            passing_file_path = dir / f"passing_test_{idx}"
            with passing_file_path.open("w") as f:
                f.write(str(test))

        for idx, test in enumerate(self.failing):
            failing_file_path = dir / f"failing_test_{idx}"
            with failing_file_path.open("w") as f:
                f.write(str(test))


    def _save_as_json(self):
        """
        Saves inputs in json files.
        """

        dir = self.out
        if self.overwrite and dir.exists():
            shutil.rmtree(dir)
        dir.mkdir(parents=True, exist_ok=True)
        
        filepath_failing = os.path.join(dir, "failing_tests.json")
        filepath_passing = os.path.join(dir, "passing_tests.json")

        failing_data = {
            "length": len(self.failing),
            "inputs": [str(input) for input in self.failing]      
        }

        passing_data = {
            "length": len(self.passing),
            "inputs": [str(input) for input in self.passing]
        }

        with open(filepath_failing, 'w') as f:
            json.dump(failing_data, f)
        
        with open(filepath_passing, 'w') as f:
            json.dump(passing_data, f)

    def get_failing_tests(self) -> List[str]:
        """
        Returns failing tests.
        """
        return self.failing
    
    def get_passing_tests(self) -> List[str]:
        """
        Returns passing tests.
        """
        return self.passing
    
    @staticmethod
    def load_failing_tests(path: os.PathLike) -> List[str]:
        """
        Retrieves failing tests from specified directory.
        Only works with json saving method.
        """
        filepath_failing = os.path.join(Path(path), "failing_tests.json")

        if os.path.exists(filepath_failing):
            with open(filepath_failing, 'r') as f:
                failing_data = json.load(f)
                return failing_data.get("inputs", [])
        else:
            return []
    
    @staticmethod
    def load_passing_tests(path: os.PathLike) -> List[str]:
        """
        Retrieves passing tests from specified directory.
        Only works with json saving method.
        """
        filepath_passing = os.path.join(Path(path), "passing_tests.json")

        if os.path.exists(filepath_passing):
            with open(filepath_passing, 'r') as f:
                passing_data = json.load(f)
                return passing_data.get("inputs", [])
        else:
            return []
    

    @staticmethod
    def load_failing_test_paths(path: os.PathLike) -> List[os.PathLike]:
        """
        Retrieves failing test paths from specified directory.
        Only works with text files saving method.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(file) for file in filepath.glob("failing_test_*")]
        else:
            return []
        

    @staticmethod
    def load_passing_test_paths(path: os.PathLike) -> List[os.PathLike]:
        """
        Retrieves passing test paths from specified directory.
        Only works with text files saving method.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(file) for file in filepath.glob("passing_test_*")]
        else:
            return []

class AvicennaTestGenerator(TestGenerator):

    def __init__(
        self,
        oracle: Callable,
        grammar: Grammar,
        initial_inputs: List[str],
        max_iterations: int,
        out: Optional[os.PathLike] = None,
        saving_method: Optional[str] = None,
        overwrite: Optional[bool] = False
    ):
        """
        Initialize the test generator
        :param os.PathLike src: The source directory of the project.
        :param Callable oracle: The oracle used for labeling inputs.
        :param Grammar grammar: The grammar used in Avicenna.
        :param List[str] initial_inputs: The initial inputs required to run Avicenna, at least one passing and one failing one.
        :param int iterations: The number of iterations.
        :param Optional[os.PathLike] out: The path location for saving labeled inputs.
        """

        super().__init__(
            out=Path(out or DEFAULT_WORK_DIR, "avicenna_test_cases"),
            saving_method=saving_method,
            overwrite=overwrite
            )

        self.oracle = oracle
        self.grammar = grammar
        self.initial_inputs = initial_inputs
        self.max_iterations = max_iterations

        self.avicenna = Avicenna(
            grammar = self.grammar, 
            oracle = self.oracle, 
            initial_inputs = self.initial_inputs,
            max_iterations = self.max_iterations,
            #enable_logging = True,
            report = SingleFailureReport()
            )
        
        self.diagnoses = None

    def run(self):
        """
        Execute Avicenna with parameter and store results in out directory.
        """

        self.diagnoses: List[Candidate] = self.avicenna.explain()

        self.failing = self.avicenna.report.get_all_failing_inputs()
        self.passing = self.avicenna.report.get_all_passing_inputs()

        LOGGER.info(f"Avicenna generated {len(self.failing)} failing and {len(self.passing)} passing inputs.")

        self._save_inputs()


    def generate_more_inputs(self, max_iterations: int, optimized_queries: bool = False):
        """
        Solves diagnosis for more inputs if a valid Diagnosis exists.
        """
        logger = logging.getLogger(__name__)
        
        if not self.diagnoses:
            logger.info("No diagnosis was found.")
            return

        passing: List[str] = []
        failing: List[str] = []
        undefined: List[str] = []

        failure_dianosis = self.diagnoses.pop(0)

        solver = ISLaSolver(
            grammar = self.grammar,
            formula = failure_dianosis.formula,
            enable_optimized_z3_queries = optimized_queries)
        
          
        for _ in range(max_iterations):
            try:      
                inp = solver.solve()            
                oracle_result, _ = self.oracle(inp)

                if oracle_result == OracleResult.PASSING:
                    passing.append(str(inp))
                elif oracle_result == OracleResult.FAILING:
                    failing.append(str(inp))
                else:
                    undefined.append(str(inp))
            except StopIteration:
                
                #solver = ISLaSolver(
                #    grammar = self.grammar,
                #    formula = self.diagnosis[0],
                #    enable_optimized_z3_queries = False)
                #    continue

                break

        self.passing.extend(passing)
        self.failing.extend(failing)

        LOGGER.info(f"ISLaSolver generated {len(failing)} more failing and {len(passing)} passing inputs.")
        if undefined:
            LOGGER.info(f"ISLaSolver generated {len(undefined)} undefined inputs.")

        self._save_inputs()



        
