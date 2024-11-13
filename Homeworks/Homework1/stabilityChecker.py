from __future__ import annotations

from typing import List, Dict
import argparse
from pathlib import Path
import importlib.util
from itertools import combinations


# import ranking class from assignment1 script. The proper way to do this would be to employ a
# python package via an __init__.py file.  However, do to the standalone nature of these scripts,
# according to the assingment requirements, this solution will be adequate.
script_dir = Path(__file__).parent.resolve()
spec = importlib.util.spec_from_file_location("assignment1", script_dir.joinpath("assignment1.py"))
assignment1_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assignment1_module)
RankingMatrices = assignment1_module.RankingMatrices


def stabilityChecker(input_filename: Path, output_filename_to_be_validated: Path):
    """
    Examines output file and checks for instabilities.  Preference rankings
    are generated from the provided input config file.
    The function generates all combinations of pairs of engaged couples, and
    cross compares them against their inverse ranking matrix.
    Moreover, if both the current partners prefer a partner in the other couple-set,
    Then it is marked as "unstable."

    The cross comparison between two couple pairings is shown below:
     man1    woman1
        \\  //
         \\//
        // \\
       //   \\
    man2    woman2

    Parameters
    ----------
    input_filename : Path
        input config that was used to generate output from assignment1.py
    output_filename_to_be_validated : Path
        output contents form output of assingment1.py, parametrized by input_filename
    """
    rankings: RankingMatrices = RankingMatrices.from_text_file(input_filename)
    pairings: List[str]
    with open(output_filename_to_be_validated, 'r', encoding = 'UTF-8') as file:
        output_content = file.readlines()
        pairings = output_content[0:rankings.n]
    split_pairings = [(line.split()) for line in pairings]
    pairings_int = [(rankings.men_str_to_int[pair[0]], rankings.women_str_to_int[pair[1]]) for pair in split_pairings]
    couple_combination_indexes = combinations(range(len(pairings_int)), 2)

    men_rank_women_matrix_inverse = [[None]*rankings.n for _ in \
            range(len(rankings.men_rank_women_matrix))]
    for i, preference_list in enumerate(rankings.men_rank_women_matrix):
        for j, preferred_woman in enumerate(preference_list):
            men_rank_women_matrix_inverse[i][preferred_woman] = j

    # Examinate all pairing combinations to check for instabilities
    stable: bool = True
    for combo in couple_combination_indexes:
        pairing1_index, pairing2_index = combo
        man1, woman1 = pairings_int[pairing1_index]
        man2, woman2 = pairings_int[pairing2_index]

        # check for instability based on cross preferences, using inverse preference matrices
        if men_rank_women_matrix_inverse[man1][woman2] < men_rank_women_matrix_inverse[man1][woman1] and \
                rankings.women_rank_men_matrix_inverse[woman2][man1] < rankings.women_rank_men_matrix_inverse[woman2][man2]\
            :
            # if this is true, then we know an instability exists, because both man1 and woman1
            # prefer woman2 and man2 respectively over each other
            stable = False
            break

    stable_str: str = 'stable' if stable else 'unstable'
    with open(script_dir.joinpath('Verified.txt'), 'w', encoding = 'utf-8') as file:
        file.write(stable_str if stable else 'unstable')

    print(f"output file: {output_filename_to_be_validated.name} is {stable_str}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_preferences_filename",
                        default="Input.txt", type = Path,
                        help = "ASCII Input file of marriage preferences as an NxN matrix.")
    parser.add_argument("output_to_be_verified",
                        default="OutputToBeVerified.txt", type = Path,
                        help = "Name of output file for stability checking")
    args = parser.parse_args()

    cwd = Path.cwd()
    input_path = cwd.joinpath(args.input_preferences_filename)
    output_validation_path = cwd.joinpath(args.output_to_be_verified)

    stabilityChecker(input_filename = input_path,
                     output_filename_to_be_validated = output_validation_path)
