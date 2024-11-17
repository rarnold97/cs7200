"""
SUBPROBLEMS:
------------
Wanting to prove:
    s3[0:i+j-1] is an interleaving of s1[0:i-1] and s1[0:j-1]

BackTracking:
    - needs recursive implementation, where done criteria is if
    you are at the upper-left corner of the table (or end of table)
    - needs to be able to make decisions on whether to add from string1
    or string 2, based on the neighboring elements within the decision matrix

PSEUDOCODE:
-----------

dynamic problem, checking for interleaving: is_string_interleaved():

let m = length(string2), n = length(string1)

assert that m + n == length(string3)

# initialize a table, and default every value to False or 0:
matrix[m + 1][n + 1]
for i = 0:1:m
    for j = 0:1:n
        matrix[i][j] = False
matrix[m][n] = True

# iterate through the table and populate each cell wrt neighboring entries
# start at the bottom-right corner of the table, and work using a bottom-up approach
for i = m:-1:0
    for j = n:-1:0
        # check if substring can be taken from string2
        if i < m and string2[i+1] == string3[i + j]
            matrix[i][j] = True
        # check if substring can be taken from string1
        if j < n and string1[j+1] == string3[i + j]
            matrix[i][j] = True

is_interleaved = matrix[m][n]
return matrix


backtracking algorithm: find_substring_sets
general rules:
    - start from the bottom-left corner
    - if you move up in the matrix, you are adding a letter from the rows: STRING2
    - if you move left in the matrix, you are adding a letter from the columns: STRING1
    - if you go from left->up or up->left, alternate strings
    - if you go from left->left or up->up, add character froms string 1 or 2 respectively

pseudocode:

inputs: dp_matrix, string1, string2

results = empty list

# solution buffers that will get popped
string1_substrings = empty deque
string2_substrings = empty deque

direction_stack = empty stack

# implement recursive back tracker
function backtrack(i, j):

    if i == 0 and j == 0:
        # termination criteria, implies we are at the beginning of the table.
        # if this is not reached, we do not append a final result,
        # because it implies a bad/incomplete path.
        results.append(bundle string1_substrings + string2_substrings)
        return
    
    # traverse left
    if j > 0 and dp_matrix[i][j-1]
        if direction_stack[-1] != LEFT:
            alternate string in buffers

        string1_substrings[0].appendleft(string1[j-1])
        direction_stack.push(LEFT)
        backtrack(i, j-1)

        direction_stack.pop()
        string1_substrings.popleft()

    # traverse up
    if i > 0 and dp_matrix[i-1][j]:
        if direction_stack[-1] != UP:
            alternate string in buffers

        string2_substrings[0].appendleft(string2[i-1])
        direction_stack.push(UP)
        backtrack(i-1, j)

        direction_stack.pop()
        string2_substrings.popleft()

# start the recursion tree at the bottom-right corner of the table, and
# work backwards
n = length(string1)
m = length(string2)
backtrack(n, m)

return results

END PSEUDOCODE
-----------


TIME COMPLEXITY ANALYSIS:
-------------------------

DYNAMIC PROGRAMMING PROBLEM COMPLEXITY: is_string_interleaved()

Time: O(m * n)
Auxillary/Space: O(m * n)
where n and m are the lengths of string1 and string2 respectively.
The time complexity comes from having to compute an entry for all
elements that comprise the DP table: m * n.  Each computation can be done
in constant time.  The space complexity is also m * n, since you are required
to maintain a table as an array with m rows and n columns to represent the
interleaving combinations, and whether they lead to the interleaved string.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
BACKTRACKING SUBSTRING SET SOLUTION COMPLEXITY: find_substring_sets()

The backtracking function, which enumerates all the sets of substrings that
can interleave to comprise string 3, represented in: find_substring_sets()
is going to be:

let M = max(n+1, m+1), to represent the depth of the recursion tree.
Note, we add 1 here for the extra row and column in the matrix.
We know that each node of the tree can either go left or up, making
the choice a binary decision.  In the worst case, we will assume we visit
every node, or nearly every node.

the time complexity of a recursion tree is: O(b^d), where b is the branching
factor, and d is the depth of the tree.  The max depth of the tree will be
the maximum dimension of the DP table.  The decision factor is 2, since we can
either traverse left or up in the table (assuming we start in the bottom-right
corner, which is the case in my solution).  Therefore, it is a binary decision.
Moreover, see the big O representations below:

Time: O(2^M) or O(2^n)

let N = number of uniqu solutions
let L = length of string3, and m and n be lengths of strings 1 and 2
then L = m * n

Auxillary/Storage Complexity: O(N * L)
"""

from __future__ import annotations


from dataclasses import dataclass, field
import enum
from typing import List, TypeAlias, Any, Deque
from pathlib import Path
from collections import deque
import argparse

IntMatrix: TypeAlias = List[List[int]]
CharArray: TypeAlias = Deque[str]

class InputFileSchema(enum.IntEnum):
    STRING1 = 0
    STRING2 = 1
    STRING3 = 2
    LENGTH = 3

@dataclass
class InputData:
    string1: str = ""
    string2: str = ""
    string3: str = ""

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
            string1: str = data[InputFileSchema.STRING1].strip()
            string2: str = data[InputFileSchema.STRING2].strip()
            string3: str = data[InputFileSchema.STRING3].strip()
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
        """
        writes all results to prescribed output file schema.

        Args:
            filename (Path): output filename
        """        
        with open(filename, 'w', encoding = 'utf-8') as output_file:
            s1_substrings_str: str = ', '.join(self.s1_substrings)
            s2_substrings_str: str = ', '.join(self.s2_substrings)
            output_str = f"""Interleaving exists: {self.results.interleave_exists}, Count of interleavings: {self.interleave_count}
s1 substrings: {s1_substrings_str}
s2 substrings: {s2_substrings_str}"""
            output_file.write(output_str)

def is_string_interleaved(string1: str, string2: str, string3: str) -> IntMatrix:
    """
    checks whether two strings can be interleaved to form the third string.
    This is done using a dynamic programming approach, where I build up
    a decision path table using a bottom-up approach. blank results will
    be returned if the sum of the lengths of the two strings do not equal
    the length of the third string.

    Args:
        string1 (str): input string 1 - characters compose columns of table.
        string2 (str): input string 2 - characters compose rows of table.
        string3 (str): interleaved string to be checked against strings 1 and 2.

    Returns:
        IntMatrix: NxN matrix of decision values that will be one or zero.
            can be used to reconstruct the substrings of string 1 and 2 to
            form the interleaved string3
    """    

    if len(string1) + len(string2) != len(string3):
        # immediately return blank results if this precondition fails
        return InterleaveResults(interleave_exists = False)

    n_columns: int = len(string1) # COLUMNS
    m_rows: int = len(string2) # ROWS
    interleave_matrix: IntMatrix = [[False] * (n_columns + 1) for _ in range(m_rows + 1)]
    interleave_matrix[m_rows][n_columns] = True

    # using a bottom-up approach, we will start at the bottom-right corner
    # of the table and work our way back to entry (0, 0).
    for i in range(m_rows, -1, -1):
        for j in range(n_columns, -1, -1):
            # check if a substring can be extracted from string2, and check previous entries for non-zero.
            if i < m_rows and string2[i] == string3[i + j] and interleave_matrix[i + 1][j]:
                interleave_matrix[i][j] = True
            # check if a substring can be extracted from string1, and check previous entries for non-zero.
            elif j < n_columns and string1[j] == string3[i + j] and interleave_matrix[i][j + 1]:
                interleave_matrix[i][j] = True
    
    # if the entry: (0,0) is True, we know that string3 is an interleaving of strings 1 and 2.
    return InterleaveResults(
        interleave_exists = interleave_matrix[0][0], matrix = interleave_matrix)

@dataclass
class InterleaveSet:
    """
    Used to bundle together substrings when creating the list of results in
    the back tracking algorithm to follow.
    """
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

def find_substring_sets(dp_matrix: List[int], string1: str, string2: str) -> List[InterleaveSet]:
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

    Parameters
    ----------
    dp_matrix : IntMatrix
        matrix solution from dynamic programming function.

    string1 : str
        input to dynamic programming problem
    string2 : str
        input to dynamic programming problem
    """

    results = []

    # solution exploration buffers
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
        if j > 0 and dp_matrix[i][j - 1]:
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
        if i > 0 and dp_matrix[i - 1][j]:
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
    n: int = len(dp_matrix) - 1
    m: int = len(dp_matrix[0]) - 1
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
    """
    Prints the dynamic programming approach matrix for the interleaving problem.
    This is incredibly useful for debugging, and displaying results.

    Args:
        matrix (IntMatrix): solution from interleaving problem
        string1 (str): characters form the column headers
        string2 (str): characters form the row labels
    """
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


def main():
    script_dir = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_strings_filename",
                        default=script_dir.joinpath("Input.txt"), type = Path,
                        help = "ASCII Input file of interleave string inputs.")
    args = parser.parse_args()
    input_filename: Path = args.input_strings_filename
    test_dir = input_filename.parent.resolve()
    output_basename = input_filename.name.replace("Input", "Output")
    output_filename = test_dir.joinpath(output_basename)

    inputs = InputData.from_input_file(input_filename)
    interleave_matrix_results: InterleaveResults = is_string_interleaved(
        inputs.string1, inputs.string2, inputs.string3
    )
    print("Interleaved String: ", inputs.string3, "\n")
    substring_sets = []
    if len(inputs.string1) + len(inputs.string2) == len(inputs.string3):
        print_dp_matrix(interleave_matrix_results.matrix, inputs.string1, inputs.string2)

        substring_sets = find_substring_sets(
            interleave_matrix_results.matrix, inputs.string1, inputs.string2)
    
    output = InterleaveResultsFile(
        results = interleave_matrix_results,
        interleave_count = len(substring_sets),
        s1_substrings = substring_sets[0].string1_parts if substring_sets else [],
        s2_substrings = substring_sets[0].string2_parts if substring_sets else []
    )
    output.to_ascii_file(output_filename)

if __name__ == "__main__":
    main()
