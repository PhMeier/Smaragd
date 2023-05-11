"""
This script compares the predicted alignments of the model to the gold alignments.
Example call:
python3 eval_ablation --test_source a_test.src --gold_align a_test.tgt --pred_align 00305000.hyps.test
--baseline --first_amr gpla_first_test_set_reduced.txt --second_amr gpla_second_test_set_reduced.txt
"""
import re
import argparse
import penman
from penman import DecodeError
from scipy import stats
import numpy as np
import random
import pickle


# from eval_helper import filter_variable_count


def read_file_per_line(filename):
    """
    Reads in a given file linewise.
    :param filename: string
    :return data: list of strings
    """
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line)
    return data


def compute_scores(amr1_triples, amr2_triples):
    """
    Helper function of _overlap_pair. Computes precision, recall and f1 score given triples of two AMR graphs.
    :param amr1_triples: list of strings
    :param amr2_triples: list of strings
    :return f1, p, r: float values
    """
    intersection = list(set(amr1_triples) & set(amr2_triples))
    p = len(intersection) / len(amr1_triples)
    r = len(intersection) / len(amr2_triples)
    if p > 0 and r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return f1, p, r


def randomizer(alignment):
    """
    Creates the random baseline by randomizing the alignment.
    Example: 1:2 10:5 9:7 to 1:7 10:2 9:2
    :param alignment: alignment string to be randomized
    :return new_alignment: changed alignment as string
    """
    splitted_alig = alignment.split("|")
    print("ALIGN: ", alignment)
    align_pos1 = [i.split()[0] for i in splitted_alig]
    align_pos2 = [i.split()[1] for i in splitted_alig]
    random.shuffle(align_pos2)
    new_alignment = ""
    for i, j in zip(align_pos1, align_pos2):
        new_alignment += str(i) + " " + str(j) + " | "
    new_alignment = new_alignment[:len(new_alignment) - 2]
    # print(new_alignment)
    return new_alignment


def align_vars(alignment, amr):
    """
    Final function for baseline creation. After randomizing the baseline (smart or normal), the variables are
    reconstructed.

    :param alignment: string
    :param amr: string
    :return graph_placeholder: string
    """
    splitted_alig = alignment.split("|")
    align = [i.split() for i in splitted_alig]
    # print(y)
    graph_placeholder = amr
    for j in align:
        # print(j[0], j[1])
        if len(j) > 1:
            if j[1] == "None":
                continue
            graph_placeholder = graph_placeholder.replace(" " + j[1] + " ", " " + j[0] + " ")
        else:
            pass
    return graph_placeholder


def _overlap_pair(amr1, amr2):
    """
    Helper function of compute_overlap_pair. Computes overlap length and union length between the two AMRs, as well
    F1, Precision and Recall scores.

    :param amr1: string
    :param amr2: string
    :return overlap_length, union_length, f1, p, r: int
    """
    # convert triples
    # amr1 = penman....(amr1)
    # amr2 = penman....(amr2)
    try:
        amr1 = penman.decode(amr1)
        amr2 = penman.decode(amr2)
        amr1 = amr1.triples
        amr2 = amr2.triples
        f1, p, r = compute_scores(amr1, amr2)
        amr1 = [str(t) for t in amr1]
        amr2 = [str(t) for t in amr2]
        overlap = set(amr1).intersection(amr2)
        union = set(amr1).union(amr2)
        overlap_length = len(overlap)
        union_length = len(union)
        return overlap_length, union_length, f1, p, r
    except DecodeError:
        return 0, 0, 0, 0, 0


def compute_overlap_pair(alignment, firstamr, secondamr, make_baseline, smart_baseline):
    """
    Computes the overlap of a pair of AMR graphs. This function also creates a random baseline for comparison.
    Wraps the function _overlap_pair.

    :param alignment: str
    :param firstamr: str
    :param secondamr: str
    :param make_baseline: bool
    :param smart_baseline: bool
    :return overlap/norm, f1, p, r: int
    """
    # Set the variables with the help of the alignment
    if make_baseline:  # randomize the alignment, step 1
        if smart_baseline:
            alignment = align_vars_baseline(secondamr, firstamr)  # now reconstruct the alignment --> smarter baseline
        else:
            alignment = randomizer(alignment)
        #alignment = align_vars_baseline(secondamr, firstamr)  # now reconstruct the alignment --> smarter baseline
    secondamr = align_vars(alignment, secondamr) # create a graph placeholder for the second AMR
    overlap, union, f1, p, r = _overlap_pair(firstamr, secondamr)  # norm, overlap
    try:
        return overlap / union, f1, p, r  # overlap / norm
    except ZeroDivisionError:
        return 0, 0, 0, 0


def align_vars_baseline(amr, firstamr):
    """
    Procedure for the smart baseline.
    :param amr: str
    :param firstamr: str
    :return alig:
    """
    #pattern = "\w\d \/ [\w\d-]*"
    # changed regex due to anonymous data
    print("First AMR: ", firstamr)
    print("Second AMR: ", amr)
    matches_amr1 = re.findall("\w*\d \/ [\d-]*", firstamr)
    matches_amr2 = re.findall("\w*\d \/ [\d-]*", amr)
    alig = ""
    # for i in range(len(matches_amr1)):
    # for match in matches_amr1:
    for i in range(len(matches_amr1)):
        # match = re.findall("\/ (\w+)-", match)
        leaf = matches_amr1[i].split("/")[1]
        # leaf = match.split("/")[1]
        for m2 in matches_amr2:
            if leaf in m2:
                # print(matches_amr1[i], leaf, m2)
                var1 = matches_amr1[i].split()[0]
                var2 = m2.split()[0]
                # print("var1: ", var1)
                # print("var2: ", var2)
                if i != len(matches_amr1) - 1:
                    alig += var1 + " " + var2 + " | "
                else:
                    alig += var1 + " " + var2
    # print(alig)
    return alig


def compute_overlap_sample_constrained(alignments, firstamrs, secondamrs, make_baseline, smart_baseline, upper_bound, lower_bound):
    """
    Compute the overlap given a minimum and a maximum length. Wraps the function compute_overlap_pair.
    :param alignments: list of strings
    :param firstamrs: list of strings
    :param secondamrs: list of strings
    :param make_baseline: bool
    :param smart_baseline: bool
    :param upper_bound: int
    :param lower_bound: int
    :return scores, f1_scores, p_scores, r_scores:
    """
    scores = []
    f1_scores = []
    p_scores = []
    r_scores = []
    #lower_bound = 35  # 35
    #upper_bound = 130  # 130
    non_evaluated = []
    for i, amr in enumerate(firstamrs):
        if len(amr) + len(secondamrs[i]) <= 9999:
            num_amr1 = sum(map(str.isdigit, amr))
            num_amr2 = sum(map(str.isdigit, secondamrs[i]))
            if num_amr1 > lower_bound and num_amr2 > lower_bound and num_amr1 < upper_bound and num_amr2 < upper_bound:
                score, f1, p, r = compute_overlap_pair(alignments[i], amr, secondamrs[i], make_baseline, smart_baseline)
                scores.append(score)
                f1_scores.append(f1)
                p_scores.append(p)
                r_scores.append(r)
            else:
                non_evaluated.append([amr, secondamrs[i]])
                # continue
            # print(i, amr)
        else:
            continue
    return scores, f1_scores, p_scores, r_scores


def compute_overlap_sample(alignments, firstamrs, secondamrs, make_baseline, smart_baseline):
    """
    Compute the overlaps between two AMR graphs using no constrains.
    :param alignments: list of strings
    :param firstamrs: list of strings
    :param secondamrs: list of strings
    :param make_baseline: bool
    :param smart_baseline: bool
    :return:
    """
    scores = []
    f1_scores = []
    p_scores = []
    r_scores = []
    indices = []
    for i, amr in enumerate(firstamrs):
        if len(amr) + len(secondamrs[i]) <= 99999:
            # print(i, len(alignments), len(secondamrs))
            score, f1, p, r = compute_overlap_pair(alignments[i], amr, secondamrs[i], make_baseline, smart_baseline)
            scores.append(score)
            f1_scores.append(f1)
            p_scores.append(p)
            r_scores.append(r)
            indices.append(i)
            # print(i, amr)
        else:
            continue
    pickle.dump(indices, open("indices.p", "wb"))
    return scores, f1_scores, p_scores, r_scores


def evaluation_procedure(gold_align, pred_align, amr1, amr2, study_bool, make_baseline, smart_baseline, mutlisents_index, lower_bound, upper_bound):
    """
    This function wraps the evaluation procedure. It compares the predicted alignments to the original gold alignments
    by evaluating F1, Precision and Recall.
    First, the scores for the gold alignments are calculated.
    :param gold_align: list of the gold alignments
    :param pred_align: list of the predicted alignments
    :param amr1: list of AMR1 strings
    :param amr2: list of AMR2 strings
    :param study_bool: boolean value, set to true for constrained evaluation
    :param make_baseline: boolean value, creates the standard baseline
    :param smart_baseline: boolean value, creates a smarter version of the baseline
    :param mutlisents_index: list of indexes of the multi-sentences
    :param lower_bound: int, lower bound for constrained evaluation
    :param upper_bound: int, upper bound for constrained evaluation
    """
    # lade die predicted alignments und dazugehörige AMRs
    # lade die gold alignments und dazugehörige AMRs
    # berechne separat die scores
    # also einfach:

    #print(amr1[0])
    #print(amr2[0])
    if study_bool: # used for constrained evaluation
        # calculate the scores for the gold alignment
        gold, f1_scores_g, p_scores_g, r_scores_g = compute_overlap_sample_constrained(gold_align, amr1, amr2, make_baseline, smart_baseline, lower_bound, upper_bound)
        test_source = read_file_per_line(args.test_source)
        amr1, amr2 = separator(test_source)
        # calculate the scores for the predicted alignments
        pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample_constrained(pred_align, amr1, amr2, make_baseline, smart_baseline, lower_bound, upper_bound)

    else:
        # calculate the scores for the gold alignment
        gold, f1_scores_g, p_scores_g, r_scores_g = compute_overlap_sample(gold_align, amr1, amr2, False, smart_baseline)
        test_source = read_file_per_line(args.test_source)
        test_source = filter_list(test_source, mutlisents_index)
        amr1, amr2 = separator(test_source)
        # 16/02: Meeting: pred_align mit gold_align getauscht
        # calculate the scores for the predicted alignments
        if make_baseline:
            pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample(gold_align, amr1, amr2, make_baseline, smart_baseline)
        else:
            pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample(pred_align, amr1, amr2, make_baseline, smart_baseline)

    # pearson = stats.pearsonr(pred, gold)
    # print("Pearson Coeff: ", pearson)
    print(len(f1_scores_p), len(f1_scores_g))
    f1_pred = sum(f1_scores_p) / (len(f1_scores_p))
    prec_pred = sum(p_scores_p) / (len(p_scores_p))
    rec_pred = sum(r_scores_p) / (len(r_scores_p))
    std_dev_pred = np.std(pred)
    pearson_1 = stats.pearsonr(f1_scores_p, f1_scores_g)[0]
    pearson_2 = stats.pearsonr(f1_scores_p, f1_scores_g)[1]
    print("F1-Gold: ", sum(f1_scores_g) / (len(f1_scores_g)))
    print("F1-Pred: ", sum(f1_scores_p) / (len(f1_scores_p)))
    print("Prec-Gold: ", sum(p_scores_g) / (len(p_scores_g)))
    print("Prec-Pred: ", sum(p_scores_p) / (len(p_scores_p)))
    print("Rec-Gold: ", sum(r_scores_g) / (len(r_scores_g)))
    print("Rec-Pred: ", sum(r_scores_p) / (len(r_scores_p)))
    print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p, f1_scores_g))
    # print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p[:1000], f1_scores_g[:1000]))
    print("std dev gold: ", np.std(gold))
    print("std dev pred: ", np.std(pred))
    # print(f1_scores_g)
    # print(f1_scores_p)
    # print("G:", f1_scores_g)
    # print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p, f1_scores_g))
    # print(len(f1_scores_p))
    # print(len(f1_scores_g))
    return f1_pred, prec_pred, rec_pred, std_dev_pred, pearson_1, pearson_2
    # print(pearsonr(compute_overlap_sample(....), compute_overlap_sample(....)))


def separator(data):
    """
    Separates two AMR graphs at the SEP token.
    :param data:
    :return:
    """
    amr1 = []
    amr2 = []
    amr1_l = []
    amr2_l = []
    over_800 = 0
    line_length = []
    for line in data:
        #print(line)
        token_length = len(line.split())
        line_length.append(len(line.split()))
        a1 = line.split("SEP")[0]
        a2 = line.split("SEP")[1]
        amr1.append(line.split("SEP")[0])
        amr2.append(line.split("SEP")[1])
        amr1_l.append(len(a1.split()))
        amr2_l.append(len(a2.split()))
        if token_length > 800:
            over_800 += 1
    print("Line length: ", line_length)
    print("Len Line AVG: ", sum(line_length) / len(line_length))
    print("Avg Length amr1: ", sum(amr1_l) / len(amr1_l))
    print("Avg Length amr3: ", sum(amr2_l) / len(amr2_l))
    print("Instances over 800: ", over_800)
    return amr1, amr2


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            # if ":snt" in line
            data.append(line)
    return data


def filter_list(data, index):
    return [data[i] for i in index]


def filter_for_multisents(first_a, second_b, test_source, gold_align, pred_align):
    mutlisents_index = list()
    sec = []
    for i in range(len(first_a)):
        if ":snt" in first_a[i]:
            mutlisents_index.append(i)

    for i in range(len(second_b)):
        if ":snt" in second_b[i]:
            sec.append(i)
            if i not in mutlisents_index:
                mutlisents_index.append(i)
    single_sentences = [i for i in range(len(gold_align))]
    single = [i for i in single_sentences if i not in mutlisents_index]
    print(single)
    mutlisents_index = mutlisents_index  # single
    test_source = filter_list(test_source, mutlisents_index)
    gold_align = filter_list(gold_align, mutlisents_index)
    pred_align = filter_list(pred_align, mutlisents_index)
    return test_source, gold_align, pred_align, mutlisents_index


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test_source", help="Test source data", required=True, type=str)  # "a_test.src"
    parser.add_argument("--gold_align", help="Gold alignments for the test data", required=True,
                        type=str)  # "a_test.tgt"
    parser.add_argument("--pred_align", help="Predicted alignments for the test data", required=True,
                        type=str)  # "00305000.hyps.test"
    parser.add_argument("--multisentence", help="Whether or not to evaluate multisentences",
                        action='store_true')
    parser.add_argument("--first_amr", help="First part of the AMR graph",
                        required=False, default="", type=str)  # "../C/gpla_first_test_set_reduced.txt"
    parser.add_argument("--second_amr", help="Second part of the AMR graph",
                        required=False, default="", type=str)  # "../C/gpla_second_test_set_reduced.txt"
    parser.add_argument("--baseline", help="Whether or not to create a baseline", action='store_true')
    parser.add_argument("--smart_baseline", help="Whether or not to create a smart_baseline or not", action='store_true')
    parser.add_argument("--constrained_study", help="Using a length constrained evaluation. Requires setting lower and upper bound.",
                        action='store_true')
    parser.add_argument("--lower_bound", help="Lower bound used for constrained evaluation. Required for constrained study. Default value is 35.",
                        required=False, default=35, type=int)
    parser.add_argument("--upper_bound", help="Upper bound used for constrained evaluation. Required for constrained study. Default value is 130.",
                        required=False, default=130, type=int)

    args = parser.parse_args()

    # pred_align = read_file_per_line("00305000.hyps.test")

    # used for multisentence

    #path = "C:/Users/phMei/PycharmProjects/AMR_ablation/A"
    test_source = read_file_per_line(args.test_source)  # "old/a_test_reduced.src")
    gold_align = read_file_per_line(args.gold_align)
    pred_align = read_file_per_line(args.pred_align)  # "00305000.hyps.test")

    lower_bound = args.lower_bound
    upper_bound = args.upper_bound

    # for the evaluuation of multisentence instances
    if args.multisentence:
        first_a = read_data(args.first_amr_multisentence)  # "../C/gpla_first_test_set_reduced.txt")
        second_b = read_data(args.second_amr_multisentence)  # "../C/gpla_second_test_set_reduced.txt")
        test_source, gold_align, pred_align, mutlisents_index = filter_for_multisents(first_a, second_b, test_source, gold_align,
                                                                                      pred_align)
    else:
        mutlisents_index = [i for i in range(len(gold_align))]

    # whether or not to produce a baseline result or not
    make_baseline = args.baseline  # False
    smart_baseline = args.smart_baseline
    study_bool = args.constrained_study

    amr1, amr2 = separator(test_source)

    if study_bool:
        if lower_bound == 0 and upper_bound == 0:
            print("Please define an upper and a lower bound for the constrained evaluation, through --lower_bound and"
                  "--upper_bound")

    # Length Check
    print("Len AMR1: ", len(amr1))
    print("Len AMR2: ", len(amr2))
    print("Gold Align: ", len(gold_align))
    print("Pred Align: ", len(pred_align))

    f1_pred, prec_pred, rec_pred, std_dev_pred, pearson_1, pearson_2 = 0, 0, 0, 0, 0, 0
    # here we did an average over 10 runs
    for i in range(10):
        a, b, c, d, e, f = evaluation_procedure(gold_align, pred_align, amr1, amr2, study_bool, make_baseline, smart_baseline, mutlisents_index, lower_bound, upper_bound)
        f1_pred += a
        prec_pred += b
        rec_pred += c
        std_dev_pred += d
        pearson_1 += e
        pearson_2 += f
    print("Final:")
    print("F1_pred: ", f1_pred / 10)
    print("Prec Pred: ", prec_pred / 10)
    print("Rec Pred: ", rec_pred / 10)
    print("STD: ", std_dev_pred / 10)
    print("Pearson1: ", pearson_1 / 10)
    print("Pearson2: ", pearson_2 / 10)
