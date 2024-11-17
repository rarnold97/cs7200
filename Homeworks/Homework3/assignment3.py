"""
------------
Wanting to prove:
    s3[0:i+j-1] is an interleaving of s1[0:i-1] and s1[0:j-1]

BackTracking
    - needs recursive implementation, where done criteria is if there are
        surrounding ones in the matrix when creating a path
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
from typing import List, TypeAlias, Any, Deque
from pathlib import Path
from collections import deque

IntMatrix: TypeAlias = List[List[int]]
CharArray: TypeAlias = Deque[str]

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
    matrix: IntMatrix = field(default_factory = list)
    interleave_exists: bool = False

@dataclass
class InterleaveResultsFile:
    results: InterleaveResults
    interleave_count: int = 0
    s1_substrings: List[str] = field(default_factory = list)
    s2_substrings: List[str] = field(default_factory = list)

    def to_ascii_file(self, filename: Path):
        with open(filename, 'w', encoding = 'utf-8') as output_file:
            s1_substrings_str: str = ','.join(self.s1_substrings)
            s2_substrings_str: str = ','.join(self.s2_substrings)
            output_str = f"""Interleaving exists: {self.results.interleave_exists}, Count of interleavings: {self.interleave_count}
s1 substrings: {s1_substrings_str}
s2 substrings: {s2_substrings_str}
            """
            output_file.write(output_str)

def is_string_interleaved(string1: str, string2: str, string3: str) -> IntMatrix:
    if len(string1) + len(string2) != len(string3):
        return InterleaveResults(interleave_exists = False)

    n_columns: int = len(string1) # COLUMNS
    m_rows: int = len(string2) # ROWS
    interleave_matrix: IntMatrix = [[False] * (n_columns + 1) for _ in range(m_rows + 1)]
    interleave_matrix[m_rows][n_columns] = True

    # using a bottom-up approach, we will start at the bottom corner of the table and work our way back.
    for i in range(m_rows, -1, -1):
        for j in range(n_columns, -1, -1):
            if i < m_rows and string2[i] == string3[i + j] and interleave_matrix[i + 1][j]:
                interleave_matrix[i][j] = True
            elif j < n_columns and string1[j] == string3[i + j] and interleave_matrix[i][j + 1]:
                interleave_matrix[i][j] = True

    #results.interleave_exists = interleave_matrix[0][0]
    # we are interested in enumerating all the different substring combinations.
    # therefore, we will traverse the table and make use of backtracking.
    return InterleaveResults(
        interleave_exists = interleave_matrix[0][0], matrix = interleave_matrix)

@dataclass
class InterleaveSet:
    string1_parts: List[str] = field(default_factory = list)
    string2_parts: List[str] = field(default_factory = list)
    
    def __post_init__(self):
        """
        we need to enforce the precondition, that the strings alternate.
        Therefore, the subparts lengths must not differ by more than one!
        """        
        assert abs(len(self.string2_parts) - len(self.string1_parts)) <= 1

class Direction(enum.IntEnum):
    UP = 0 # Traversing a row
    LEFT = 1 # Traversing a column 

def find_substring_sets(matrix: List[int], string1: str, string2: str) -> List[InterleaveSet]:
    """
    Does a recursive backtracking path, similar to DFS.

    The rules of traversing the matrix and constructing strings are:
        - start from the bottom-left corner
        - if you move up in the matrix, you are adding a letter from the rows: STRING2
        - if you move left in the matrix, you are adding a letter from the columns: STRING1
        - if you go from left->up or up->left, alternate strings
        - if you go from left->left or up->up, add character froms string 1 or 2 respectively
    
    NOTE: in this function, I am also maintaining a global reference to a stack
    that records the history of direction from the previous call, which indicates
    whether the path has to switch between strings 1 and 2. This is important for
    the backtracking algorithm when it backtracks and makes a different viable
    decision in the case where the decision matrix has multiple paths.
    """

    results = []
    string_parts_1: Deque[CharArray] = deque([])
    string_parts_2: Deque[CharArray] = deque([])
    
    # mainly used to determine whether we need to alternate character source between
    # strings 1 and 2 and vice versa. This needs to be maintained on a stack, so that
    # when the backtracking algorithm recurses the tree, a history is maintained for
    # direction decisions that were made at previous nodes of the recursion tree.
    direction_stack: List[Direction] = []

    def backtrack(i: int, j: int):

        # termination condition is when we are at the top-left, or beginning of the matrix.
        if i == 0 and j == 0:
            string1_parts = [''.join(char_arr) for char_arr in string_parts_1]
            string2_parts = [''.join(char_arr) for char_arr in string_parts_2]
            results.append(InterleaveSet(
                string1_parts = string1_parts, string2_parts = string2_parts
            ))
            return

        # check the tabulated results in the input decision matrix
        # to determine where to go next
        
        # search left through columns
        if j > 0 and matrix[i][j - 1]:
            # decide if we alternated or not
            # if we were going up, and are now going left, we alternated.
            # if None, we are starting with an empty list, and need to add an empty str
            if not direction_stack or direction_stack[-1] == Direction.UP:
                string_parts_1.appendleft(deque([]))
            string_parts_1[0].appendleft(string1[j - 1])
            direction_stack.append(Direction.LEFT)
            backtrack(i, j - 1)
            
            # remove solution and direction that we just tried
            if direction_stack:
                direction_stack.pop()
            string_parts_1[0].popleft()
            if not string_parts_1[0]:
                string_parts_1.popleft()

        # search up through rows
        if i > 0 and matrix[i - 1][j]:
            # decide if we alternated or not
            # if we were going left, and are now going up, we alternated.
            # if None, we are starting with an empty list, and need to add an empty str
            if not direction_stack or direction_stack[-1] == Direction.LEFT:
                string_parts_2.appendleft(deque([]))
            string_parts_2[0].appendleft(string2[i - 1])
            direction_stack.append(Direction.UP)
            backtrack(i - 1, j)
            
            # remove solution and direction that we just tried
            if direction_stack:
                direction_stack.pop()
            string_parts_2[0].popleft()
            if not string_parts_2[0]:
                string_parts_2.popleft()
        
    # top of recursion tree
    n: int = len(matrix) - 1
    m: int = len(matrix[0]) - 1
    backtrack(n, m)

    return results

def print_matrix(matrix: List[List[Any]]):

    # find the maximum width of each column
    col_widths = [max(len(str(row[i])) for row in matrix) for i in range(len(matrix[0]))]

    # print each row with proper alignment
    for j, row in enumerate(matrix):
        print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)), "|")
        print("  ", "-" * ((3 + col_widths[j]) * (len(row) - 1) - 1))

def print_dp_matrix(matrix: IntMatrix, string1: str, string2: str):
    assert len(string1) == len(matrix) - 1 and len(string2) == len(matrix[0]) - 1

    columns = ["",]
    columns.extend(list(string1))
    columns.append(" ")

    rows = list(string2)
    rows.append(" ")

    full_matrix = [columns,]
    for i, row in enumerate(matrix):
        full_matrix.append([rows[i], *[str(int(element)) for element in row]])
    print_matrix(full_matrix)
    print("\n")

def test_print_matrix():
    string1 = "XXY"
    string2 = "XXZ"
    results = [
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 1, 1]
    ]
    print_dp_matrix(results, string1, string2)

def test_is_string_interleaved():
    # case 1.1
    string1 = "aab"
    string2 = "axy"
    string3 = "aaxaby"
    results_1_1: InterleaveResults = is_string_interleaved(string1, string2, string3)
    print("Interleaved String: ", string3, "\n")
    print_dp_matrix(results_1_1.matrix, string1, string2)
    assert results_1_1.interleave_exists

    # case 1.2 
    string3 = "abaaxy"
    results_1_2: InterleaveResults = is_string_interleaved(string1, string2, string3)
    print("Interleaved String: ", string3, "\n")
    print_dp_matrix(results_1_2.matrix, string1, string2)
    assert not results_1_2.interleave_exists

    # case 2.1
    string1 = "aabcc"
    string2 = "dbbca" 
    string3 = "aadbbcbcac"
    results_2_1: InterleaveResults = is_string_interleaved(string1, string2, string3)
    print("Interleaved String: ", string3, "\n")
    print_dp_matrix(results_2_1.matrix, string1, string2)
    assert results_2_1.interleave_exists

    # case 3
    string1 = "XXY"
    string2 = "XXZ"
    string3 = "XXZXXY"
    results_3: InterleaveResults = is_string_interleaved(string1, string2, string3)
    print("Interleaved String: ", string3, "\n")
    print_dp_matrix(results_3.matrix, string1, string2)
    assert results_3.interleave_exists

def test_back_tracker():
    string1 = "aabcc"
    string2 = "dbbca" 
    string3 = "aadbbcbcac"
    results_2_1: InterleaveResults = is_string_interleaved(string1, string2, string3)
    print("Interleaved String: ", string3, "\n")
    print_dp_matrix(results_2_1.matrix, string1, string2)
    assert results_2_1.interleave_exists

    string_subsets: List[InterleaveSet] = find_substring_sets(results_2_1.matrix, string1, string2)
    i: int
    subset: InterleaveSet
    for i, subset in enumerate(string_subsets):
        print(f"""-------------------------
Subset {i}:
String 1 Substrings: {subset.string1_parts}
String 2 Substrings: {subset.string2_parts}
-------------------------
        """)
    
if __name__ == "__main__":
    #test_print_matrix()
    #test_is_string_interleaved()
    test_back_tracker()


"""
TEST CASES:

CASE 1
------
1 aab
2 axy
3.1 aaxaby --> True
3.2 abaaxy --> False

CASE 2
------
1 aabcc
2 dbbca
3.1 aadbbcbcac --> True
3.2 aadbbbaccc --> False
"""