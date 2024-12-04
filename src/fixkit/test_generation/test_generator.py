import abc
from abc import ABC
import os
import json
from pathlib import Path
from typing import List, Optional
from fixkit.constants import DEFAULT_WORK_DIR
import shutil
from fixkit.logger import LOGGER
import random
import numpy as np

class TestGenerator(ABC):

    def __init__(
        self,
        seed: int = 0,
        out: Optional[os.PathLike] = None,
        saving_method: Optional[str] = None,
        save_automatically: Optional[bool] = True
    ):
        """
        Initialize the test generator
        :param Optional[os.PathLike] out: The path location for saving labeled inputs.
        :param Optional[str] saving_method: Saves passing and failing test cases in json files or separate text files.
        :param Optional[bool] save_automatically: If true, test cases are automatically saved after running. Alternatively, use save_test_cases() with a given path. 
        """

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.out = Path(out or DEFAULT_WORK_DIR)

        self.saving_method = saving_method or "files" 
        if self.saving_method not in ["json", "files"]:
            raise ValueError('Invalid argument. Use either "json" or "files".')

        self.save_automatically = save_automatically

        self.failing = None
        self.passing = None
        self.saving_path = Path(self.out, "test_cases")
    
    @abc.abstractmethod
    def run(self):
        """
        Abstract method for running the test generator.
        """
        pass

    def _save_inputs(self):
        if not self.save_automatically:
            return

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

        dir = self.out / "test_cases"
        if dir.exists():
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

        dir = self.out / "test_cases"
        if dir.exists():
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


    def save_test_cases(self, path: os.PathLike):

        dir = Path(path)
        dir.mkdir(parents=True, exist_ok=True)

        for idx, test in enumerate(self.passing):
            passing_file_path = dir / f"passing_test_{idx}"
            with passing_file_path.open("w") as f:
                f.write(str(test))

        for idx, test in enumerate(self.failing):
            failing_file_path = dir / f"failing_test_{idx}"
            with failing_file_path.open("w") as f:
                f.write(str(test))

        LOGGER.info(f"Saved {len(self.failing) + len(self.passing)} test cases under {self.out}")

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
    def load_failing_test_paths(path: os.PathLike, num_tests: int) -> List[os.PathLike]:
        """
        Retrieves first X number of failing test paths from specified directory.
        Only works with text files saving method.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(os.path.join(path, f"failing_test_{i}")) 
            for i in range(num_tests)]
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
        
    @staticmethod
    def load_passing_test_paths(path: os.PathLike, num_tests: int) -> List[os.PathLike]:
        """
        Retrieves first X number of passing test paths from specified directory.
        Only works with text files saving method.
        """
        filepath = Path(path)

        if filepath.exists():
            return [os.path.abspath(os.path.join(path, f"passing_test_{i}")) 
            for i in range(num_tests)]
        else:
            return []