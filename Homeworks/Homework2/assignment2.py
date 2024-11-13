"""
CS 7200 Homework 2
Ryan Arnold

GREEDY ALGORITHM PSEUDOCODE
-----------------------------------------------------------------------------------------------
let A be the set of optimally scheduled jobs.
let n be the total number of jobs.
let m be the total number of machines.
    we assume m = 2 in this assignment.

store array/list called: Machines containing all machines
store array/list called: machine_latest_finish_times
    initialize with zeroes

NOTE: for machine_latest_finish_times, we implement these as floats
    as a small optimization (i.e., bypassing need to construct list).
    however, in a more complex scenario with an abitrary number of machines,
    an array-like structure is better.  For the sake of the pseudo-code,
    we will still use an array to aid in readability.

Initialize empty A
Initialize empty job queue, Q

sort all jobs by finish time: f1 < ... < fn
    do this by storing sorted jobs in Q (heap queue)

for j = 1:n
    job = heap_pop(Q)
    for m = 1:2
        machine = Machines[m]
        fi = machine_latest_finish_times[m]
        if job.start >= fi (i.e., "compatible"):
            A <-- job
            machine_latest_finish_times[m] = job.finish
        else
            drop job
return A

-----------------------------------------------------------------------------------------------

GREEDY HEURISTIC: Earliest Finish Time

INFORMAL ARGUMENT OF CORRECTNESS
-----------------------------------------------------------------------------------------------
In the single machine case, we have seen the proof that we can find a schdule that has
has the optimal number of jobs scheduled (albeit, not guaranteed to be unique).
The greedy approach of selecting earliest finish time guarantees that each job will finish no later
than the analogous indexed job in any other optimal solution.

In my solution, I hypothesize that the earliest finish time compatibility criteria
can be extended to the double (or multi) machine case.  We know that machine 1 will maximize compatible, non-overlapping
jobs to produce an optimal schedule for that machine. However, machine 1 will not be able to handle every
job it encounters, so the jobs that are incompatible with machine 1 will be assigned to machine 2,
assuming it is also compatible with machine 2. There are some similarities to interval partitioning in this regard.
When both machines are busy, then the overlapping job will be dropped.
Assigning the other overlapping jobs that cannot be handled by machine 1 to machine 2, following the same greedy
heuristic, should still produce an optimal schedule. This approach is effectively paralleizing the algorithm.
If we consider the set V-S to be the remainder of jobs that were not scheduled by Machine1,
it is as if we are running the greedy algorithm on the job set: V-S. Moverover, I expect
that the greedy algorithm applied in parallel to the V-S set shoud produce an optimal schedule for
the jobs available in V-S.  Overall, this will produce an optimal schedule after Machines 1 & 2
select their jobs using the greedy heuristic I proposed.

Since we are not able to change the quantity or partition new machines, this approach seems reasonable.

COMPUTATIONAL COMPLEXITY ANALYSIS
-----------------------------------------------------------------------------------------------
let n = total number of jobs.

sorting the priority queue: O(nlogn)
    - pushing to queue is O(logn)
    - need to push n times
iterating through all jobs O(nlogn):
    - n jobs
    - checking compatibility is O(1)
    - poping the queue is O(logn)
    - storing and bookeeping saved jobs: O(1)
Total complexity = O(nlogn) + O(nlogn) = 2O(nlogn) = cO(nlogn) --> Theta(nlogn)
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import heapq as heap
from pathlib import Path
from typing import List
import argparse

MAX_TIME = int(10e6)


class InputFileSchema(enum.IntEnum):
    """
    Used to organize the row ordering
    based on the schema of the input asii file
    """
    N = 0
    JOBS = 1


@dataclass(order = True)
class Job:
    """
    records data about each job that needs scheduling.

    priority field is required for heap ordering, see the following link:
    https://docs.python.org/3/library/heapq.html
    """
    priority: int

    id: int
    start_time: int
    end_time: int


@dataclass
class Schedule:
    """
    Used to record optimized schedule results.
    keeps records of the jobs (including their IDs) for each machine.
    """

    machine1: List[Job] = field(default_factory = list)
    machine2: List[Job] = field(default_factory = list)

    def export_to_ascii(self, output_filename: Path):
        """
        Used to export the results to an output ascii file,
        according to the prescribed output schema given in the assignment

        Parameters
        ----------
        output_filename : Path
            file to write to, defaults to: CWD/Output{%d}.txt
        """
        with open(output_filename, 'w', encoding = 'utf-8') as file:
            file.write(str(len(self.machine1 + self.machine2)) + '\n')
            machine1_job_ids_str = ' '.join([str(job.id) for job in self.machine1])
            file.write(f"{machine1_job_ids_str}\n")
            machine2_job_ids_str = ' '.join([str(job.id) for job in self.machine2])
            file.write(f"{machine2_job_ids_str}")


# for this standalone script, we are using a global job queue to simplify
# the python heapq API
job_queue = []

def load_input_file(input_filename: Path):
    """
    Loads all input job data according to the prescribed input ascii file schema,
    prescribed in the assignment.

    Parameters
    ----------
    input_filename : Path
        expected to be in working directory, and named: Input{%d}.txt
    """
    assert input_filename.exists()
    with open(input_filename, 'r', encoding = 'utf-8') as file:
        file_contents: List[str] = file.read().splitlines()
        n_jobs = int(file_contents[0])
        assert n_jobs > 0, "no jobs to process ???"
        n_rows = len(file_contents)
        assert n_rows - 1 == n_jobs, \
            "invalid input file, length does not match n jobs in line 1"
        job_data = file_contents[InputFileSchema.JOBS:n_rows]
        for job_entry in job_data:
            # if the line does not have 3 elements, this will throw an exception
            job_id_str, job_start_str, job_end_str = job_entry.split()
            job_start = int(job_start_str)
            job_end = int(job_end_str)
            assert job_start >= 0
            assert job_start <= MAX_TIME
            assert job_end <= MAX_TIME
            assert job_start < job_end

            # prioritize based on finish time
            job = Job(id = int(job_id_str), start_time = job_start, end_time = job_end,
                      priority = job_end)
            heap.heappush(job_queue, job)


def greedy_job_scheduler() -> Schedule:
    """
    Runs a greedy algorithm for maximizing jobs scheduled between two machines (m = 2)

    The algorithm is as follows:
        Sort all jobs by ascending finish time.
        iterate through jobs 1:n
            iterate through all machines 1:m
                if job n is compatible with all machines, prefer machine m.
                else if job n is compatible with machine m
                    assign finish time and record job id to machine m.
                else
                    discard job

    Returns
    -------
    Schedule
        data structure that has mapping of job ids to machines in chronological order.
    """

    schedule = Schedule()
    machine_1_last_finish_time, machine_2_last_finish_time = int(0), int(0)

    while job_queue:
        next_job: Job = heap.heappop(job_queue)

        # even if machine 1 and machine 2 are both compatible, this control flow is structured
        # to prefer machine 1 if both are compatible.
        if machine_1_last_finish_time <= next_job.start_time:
            schedule.machine1.append(next_job)
            machine_1_last_finish_time = next_job.end_time
        elif machine_2_last_finish_time <= next_job.start_time:
            schedule.machine2.append(next_job)
            machine_2_last_finish_time = next_job.end_time
        # if neither of the above conditions are met, the job is incompatible,
        # and we therefore drop it.
    return schedule


if __name__ == "__main__":
    # setup cli args for user
    cwd = Path.cwd()
    parser = argparse.ArgumentParser()
    parser.add_argument("input_job_data_filename",
                        default=cwd.joinpath("Input.txt"), type = Path,
                        help = "ASCII Input file of job data.")
    args = parser.parse_args()
    input_path: Path = args.input_job_data_filename
    test_dir = input_path.parent.resolve()
    output_basename = input_path.name.replace("Input", "Output")
    output_path = test_dir.joinpath(output_basename)

    load_input_file(input_path)
    optimal_schedule: Schedule = greedy_job_scheduler()
    optimal_schedule.export_to_ascii(output_path)
