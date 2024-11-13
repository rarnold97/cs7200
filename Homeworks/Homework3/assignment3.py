"""


SUBPROBLEMS:
------------
Wanting to prove:
    s3[0:i+j-1] is an interleaving of s1[0:i-1] and s1[0:j-1]
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import List, Tuple, TypeAlias
from pathlib import Path

IntMatrix: TypeAlias = List[List[int]]

class InputFileSchema(enum.IntEnum):
    STRING1 = 0
    STRING2 = 1
    STRING3 = 2
    LENGTH = 3

@dataclass
class InputData:
    string1: str
    string2: str
    string3: str

    @classmethod
    def from_input_file(cls, filename: Path) -> InputData:
        """
        Reads in input data as prescrtibed by assignment.
        Will assert if the input data format is not matched.

        Parameters
        ----------
        filename : Path
            path to input data file.

        Returns
        -------
        InputData
            returns input parameters to the string interleaving problem.
        """
        assert filename.exists()
        with open(filename, 'r', encoding = 'utf-8') as file:
            data = file.readlines()
            assert len(data) == InputFileSchema.LENGTH
            string1: str = data[InputFileSchema.STRING1]
            string2: str = data[InputFileSchema.STRING2]
            string3: str = data[InputFileSchema.STRING3]
            return cls(string1, string2, string3)

@dataclass
class InterleaveResults:
    interleave_exists: bool = False
    interleave_count: int = 0
    s1_substrings: List[str] = field(default_factory = list)
    s2_substrings: List[str] = field(default_factory = list)

    def to_ascii_file(self, filename: Path):
        with open(filename, 'w', encoding = 'utf-8') as output_file:
            s1_substrings_str: str = ','.join(self.s1_substrings)
            s2_substrings_str: str = ','.join(self.s2_substrings)
            output_str = f"""Interleaving exists: {self.interleave_exists}, Count of interleavings: {self.interleave_count}
s1 substrings: {s1_substrings_str}
s2 substrings: {s2_substrings_str}
            """
            output_file.write(output_str)


def is_string_interleaved(string1: str, string2: str, string3: str) -> InterleaveResults:
    # aliases from input data
    if len(string1) + len(string2) != len(string3):
        return InterleaveResults(interleave_exists = False)

    results = InterleaveResults()
    n: int = len(string1) # ROWS
    m: int = len(string2) # COLUMNS
    interleave_matrix: IntMatrix = [[False] * (n+1) for _ in range(m+1)]
    interleave_matrix[m+1][n+1] = True

    # using a bottom-up approach, we will start at the bottom corner of the table and work our way back.
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i < n and string1[i] == string3[i + j] and interleave_matrix[i + 1][j]:
                interleave_matrix[i][j] = True
            elif j < m and string2[j] == string3[i + j] and interleave_matrix[i][j + 1]:
                interleave_matrix[i][j] = True

    results.interleave_exists = interleave_matrix[0][0]

    # we are interested in enumerating all the different substring combinations.
    # therefore, we will traverse the table and make use of backtracking.

    return results