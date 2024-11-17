# How to Run

The `assignment3.py` program is configured to accept a single input,
an ascii input file that follows the input schema prescribed in the assignment
document. If no arguments are passed, the default will be: Path-of-script/OutputN.txt,
where N will be appended if your Input filename contains a number (e.g., Input1.txt --> Output1.txt).
N will be blank if no number is provided (e.g., Input.txt --> Output.txt).

# Algorithms

There are two parts to my approach.

1. Generation of the decision matrix, via dynamic programming
2. Enumeration of all string subsets that lead to the interleaved String 3.

Part 1 is contained in `is_string_interleaved()`.  I found that rather than starting at
element (0,0), it is more succinct to start at the bottom-right corner of the table instead:
(m, n), and initialize the value to True. The table is also m + 1 by n + 1 entries,
where m and n are the lengths of string2 and string1 respectively. The extra column and row
serve as padding, and conceptually aid in priming the beginning of the table, where
a decision has not been made yet, and the neighboring cells must be examined to decide where to
traverse.

Part2 is contained in `find_substring_sets()`. I implemented this using a backtracking algorithm,
so that when one path was traversed, the function can recursively backtrack to a point where there 
is a decision fork, and try other solutions.  If the tree does not end at element (0,0), then it is
not a valid path. This is enforced by making the termination criteria be when both the bookeeping
indexes are 0. Otherwise, the function will just bubble up the stacks without returning a solution.
Solutions are added to a global list called results.  After each recursive decision call, I pop 
attempted solutions off a buffer variable.  I chose to use a deque as my solution buffer, since it
efficiently implements `appendleft()`.  Since I was traversing the table backwards, this was preferred.
Appending to the left makes it much simpler to reconstruct the substrings in order, while preserving
efficiency. I also maintain a global stack called: `direction_stack`. This is maintains the direction
of the most recent decision (either UP or LEFT). Knowing this information is what determines
whether you append characters to the substring set for string1 or string2.  I chose to maintain this
on a stack so that the decision history is maintained for when the recursion tree traverses to an earlier
point in the tree, and need to know the previous move when trying another direction. Maintaining this
state in a single variable would not preserve this history, which is imperative to ensure that characters
get appended to the correct subset when the algorithm makes alternate decisions along the tree and 
decsion matrix paths.

# Complexity Analysis

## DYNAMIC PROGRAMMING PROBLEM COMPLEXITY: is_string_interleaved()

Time: O(m * n)
Auxillary/Space: O(m * n)
where n and m are the lengths of string1 and string2 respectively.
The time complexity comes from having to compute an entry for all
elements that comprise the DP table: m * n.  Each computation can be done
in constant time.  The space complexity is also m * n, since you are required
to maintain a table as an array with m rows and n columns to represent the
interleaving combinations, and whether they lead to the interleaved string.

## BACKTRACKING SUBSTRING SET SOLUTION COMPLEXITY: find_substring_sets()

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

# Tests

During testing, I examined several cases, which were the basis for the text file I provide 
in the `Tests/` folder. Each test case is described in detail below:


## CASE 1

> NOTE: I tested two versions of this, denoted by the different values of String3.

1 aab
2 axy
3.1 aaxaby --> True
3.2 abaaxy --> False

## CASE 2

> NOTE: I tested two versions of this, denoted by the different values of String3.

1 aabcc
2 dbbca
3.1 aadbbcbcac --> True
3.2 aadbbbaccc --> False

## CASE 3

1. XXY
2. XXZ
3. XXZXXY --> True

## CASE 4

1. ABC
2. DEF
3. ADBECF --> True

# CASE 5

This test case was a plumbing test, to make sure the algorithm properly
reports blank results when the lengths of string1 and string2 do not add up to the
length of string3:

1. AAA
2. BBB
3. AAABBBX --> False (does not get far in algorithm, due to validation code).