"""
CS 7200 Assignment 1
Ryan Arnold

Pseudo-code for the General GS Marriage Algorithm:

while (men are free)
    man = next(free man)
    for woman in orderd preferences not proposed to yet
        if woman is free
            propose(man, woman)
        else
            if woman prefers man
                man and woman become engaged
                man' is free
            elif woman prefers man'
                man is still free
return (stable pairs)
"""
from __future__ import annotations

from typing import TypeAlias, List, Dict
from collections import deque
import enum
import argparse
from pathlib import Path

List2DStr: TypeAlias = List[List[str]]
List2DInt: TypeAlias = List[List[int]]

class InputFileSchema(enum.IntEnum):
    N = 0
    MEN_PREFERENCES = 1
    WOMEN_PREFERENCES = 2

class RankingMatrices:
    def __init__(self, men_preference_matrix: List2DStr, women_preference_matrix: List2DStr):
        self.n = len(men_preference_matrix)

        all_men = [row[0] for row in men_preference_matrix]
        assert len(all_men) == self.n
        self.men_str_to_int: Dict[str, int] = {name: i for i,name in enumerate(all_men)}
        self.men_int_to_str: Dict[int, str] = {i: name for i, name in enumerate(all_men)}
        # checking for duplicate entries

        all_women = [row[0] for row in women_preference_matrix]
        assert len(all_women) == self.n
        self.women_str_to_int: Dict[str, int] = {name: i for i, name in enumerate(all_women)}
        self.women_int_to_str: Dict[int, str] = {i: name for i, name in enumerate(all_women)}
        # checking for duplicate entries

        self.men_rank_women_matrix: List2DInt = [[self.women_str_to_int[woman_name] for woman_name in man_ranks[1::]] \
            for man_ranks in men_preference_matrix]
        self.women_rank_men_matrix: List2DInt = [[self.men_str_to_int[man_name] for man_name in woman_ranks[1::]] \
            for woman_ranks in women_preference_matrix]
        self.women_rank_men_matrix_inverse = [[None]*self.n for _ in \
            range(len(self.women_rank_men_matrix))]
        for i, preference_list in enumerate(self.women_rank_men_matrix):
            for j, preferred_man in enumerate(preference_list):
                self.women_rank_men_matrix_inverse[i][preferred_man] = j

    @classmethod
    def from_text_file(cls, filename: Path) -> RankingMatrices:
        """
        Create a ranking matrix, for both men and women, from an ascii input file.

        Parameters
        ----------
        filename : Path
            ascii input file that follows schema provided in assignment 1 document.

        Returns
        -------
        RankingMatrices
            matrices for both men and women preferences.

        Raises
        ------
        RuntimeError
            does error handling for when input file does not match prescirbed schema.
        """
        with open(filename, 'r', encoding = 'utf-8') as file:
            file_contents = file.read().splitlines()
            if not file_contents:
                raise RuntimeError(f"Cannot parse empty file: {filename}")
            matrix_rank = int(file_contents[InputFileSchema.N])
            assert len(file_contents) == 2*matrix_rank + 1, \
                f"Input file: {filename} does not match required schema"
            men_preferences_start: int = InputFileSchema.MEN_PREFERENCES
            men_preferences_stop: int = InputFileSchema.MEN_PREFERENCES + matrix_rank
            men_preferences: List2DStr = [file_contents[i].split() for i in \
                range(men_preferences_start, men_preferences_stop)]
            women_preferences_start: int = men_preferences_stop
            women_preferences_stop: int = 1 + InputFileSchema.WOMEN_PREFERENCES * matrix_rank
            women_preferences: List2DStr = [file_contents[j].split() for j in \
                range(women_preferences_start, women_preferences_stop)]
            return cls(men_preferences, women_preferences)

def gale_shapely_marriage_scenario(input_filename: Path, output_filename: Path):
    """
    Parameters
    ----------
    input_filename : Path
        Input ascii file with prescribed marriage pairing schema.
    output_filename : Path
        location to write the output pairings file.
    """
    rankings = RankingMatrices.from_text_file(input_filename)

    men_values = tuple(rankings.men_str_to_int.values())
    single_men = deque(men_values)
    husbands = {woman: None for woman in rankings.women_str_to_int.values()}
    proposal_counts = {man_value: 0 for man_value in men_values}
    counter = 0
    while single_men:
        bachelor = single_men[0]
        next_proposal_index = proposal_counts[bachelor]
        assert next_proposal_index < len(rankings.men_rank_women_matrix[bachelor])
        current_man_preference_list = rankings.men_rank_women_matrix[bachelor][next_proposal_index::]
        for woman in current_man_preference_list:
            # propose, and only allow the woman to swap if it means matching up

            # man proposes on each pass of the loop, whether he gets rejected or not
            proposal_counts[bachelor] += 1
            counter += 1

            # if the woman has not been proposed to, engage the current bachelor and the indexed woman.
            current_partner = husbands[woman]
            if current_partner is None:
                husbands[woman] = bachelor
                single_men.popleft()
                break
            # determine if the woman prefers the bachelor to her current partner
            if rankings.women_rank_men_matrix_inverse[woman][bachelor] < \
                    rankings.women_rank_men_matrix_inverse[woman][current_partner]:
                husbands[woman] = bachelor
                single_men.popleft()
                # put the current male partner back in the front of the queue.
                single_men.appendleft(current_partner)
                break

    total_proposals: int = sum(proposal_counts.values())
    # sort the husbands dictionary to retain husband ordering in input file
    # in this case, we sort by value, which are the men names
    sorted_husbands = {key: husbands[key] for key in sorted(husbands, key = husbands.get)}

    # print result to console
    results_str = '\n'.join([f"{rankings.men_int_to_str[man_value]} {rankings.women_int_to_str[woman_value]}" \
        for woman_value, man_value in sorted_husbands.items()])
    print(f"""
Engagement Results
------------------
{results_str}
total men proposals: {total_proposals}
        """)

    # write results to file
    with open(output_filename, 'w', encoding = 'UTF-8') as file:
        file.write(f"{results_str}\n{total_proposals}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_preferences_filename",
                        default="Input.txt", type = Path,
                        help = "ASCII Input file of marriage preferences as an NxN matrix.")
    args = parser.parse_args()

    cwd = Path.cwd()
    default_input_path = cwd.joinpath(args.input_preferences_filename)
    default_output_path = cwd.joinpath("Output.txt")

    gale_shapely_marriage_scenario(default_input_path, default_output_path)
