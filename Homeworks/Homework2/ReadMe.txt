# Team Members
1. Ryan Arnold

# Running the Code
As prescribed in the instructions, the code can be ran as follows:

```bash
python assignment2.py <input-filename.txt>
```

By default, the input filename will be: CWD/Input.txt.
The output file will be written in the same directory as the input file, and
follows the pattern: Output{%d}.txt.  The code will deduce the integer, based on the input file.
For example, if you provide the following args:

```bash
python assignment2.py Tests/Input1.txt
```

The output file will be: `Test/Output1.txt`

The working directory is assumed to be the CWD, and all relative paths provided will be
appended to the CWD. In the above example, you can expect Tests/Input1.txt/Output1.txt to be:
`CWD/Tests/Input1.txt` and `CWD/Tests/Output1.txt`.

# Description of Tests
The provided tests are inspired by the edge cases presented in the proofs of the Textbook in
Section 4.1 (KleinBerg & Tardos, Algorithm Design 4th ed.).  Input0 is based on the sample job listing
provided in the assignment. Inputs 1-3 are simpler examples,
while Input4 is more of a stress test example.
Below are visual aids for each of the Inputs 1-4:

## Input 1 - 5 Jobs
  |---| |---| |---| |---|
|--------------------------|

## Input 2 - 3 Jobs
|---------------|    |-------------|
        |-------------------|

## Input 3 - 9 Jobs
|------|  |------|  |------|  |------|
    |-------| |-------| |--------|
    |-------|           |--------|

## Input 4 - 9 Jobs
|----------------------------|    |----|
|----|    |-------|   |----|    |--------|
|-------------| |---|   |-----|

# Results and Analysis
- Output1 can accept all jobs, which makes sense since we have 2 machines.
- Output2 can similarly accept all jobs, having 2 machines.
- Output3 can accept all the jobs except 2.  Looking at the intervals, there are 2 pairs
    of jobs that share the same intervals, and they overlap with other jobs in the top row of the diagram.
    Intuitively, it makes sense that it cannot accomodate them, but seeing that it accomodates the others,
    this seems to be optimal, based on the limited resources.
- Output4 can accept 8 jobs.  The interesting result is the job it left out. It leaves out the longest job.
    We label this job1 in the input file.  This is correct, because dropping the longest job lets the
    machines process several other jobs. In a non-weighted/no-cost scenario, this is the correct result.
    Accepting the longest job incurrs the cost of missing several other smaller jobs.