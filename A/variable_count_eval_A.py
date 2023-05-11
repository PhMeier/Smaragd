import re

import penman
from penman import DecodeError
from scipy import stats
import numpy as np
import random
import pickle
from eval_helper import filter_variable_count


def read_file_per_line(filename):
    """
    Reads in a given file linewise.
    :param filename:
    :return:
    """
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line)
    return data


def compute_scores(amr1_triples, amr2_triples):
    intersection = list(set(amr1_triples) & set(amr2_triples))
    p = len(intersection) / len(amr1_triples)
    r = len(intersection) / len(amr2_triples)
    f1 = 0
    if p > 0 and r > 0:
        f1 = 2 * p * r / (p + r)
    else:
        f1 = 0
    return f1, p, r


def randomizer(alignment, amr1, amr2):
    """
    Creates the random baseline.
    Example: 1:2 10:5 9:7 to 1:7 10:2 9:2
    :param alignment:
    :param amr1:
    :param amr2:
    :return:
    """
    splitted_alig = alignment.split("|")
    print("ALIGN: ", alignment)
    align_pos1 = [i.split()[0] for i in splitted_alig]
    align_pos2 = [i.split()[1] for i in splitted_alig]
    random.shuffle(align_pos2)
    new_alignment = ""
    for i,j in zip(align_pos1, align_pos2):
        new_alignment += str(i) + " " + str(j) + " | "
    new_alignment = new_alignment[:len(new_alignment)-2]
    #print(new_alignment)
    return new_alignment


def align_vars(alignment, amr):
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
    #convert triples
    #amr1 = penman....(amr1)
    #amr2 = penman....(amr2)
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
        return len(overlap), len(union), f1, p, r
    except DecodeError:
        return 0,0,0,0,0



def compute_overlap_pair(alignment, firstamr, secondamr, make_baseline):
    # hier setzt du die variablen mit Hilfe des alignments
    if make_baseline: # randomize the alignment, step 1
        alignment = randomizer(alignment, firstamr, secondamr)
        alignment = align_vars_baseline(alignment, secondamr, firstamr) # now recostruct the alignment
    secondamr = align_vars(alignment, secondamr)
    overlap, union, f1, p, r = _overlap_pair(firstamr, secondamr) # norm, overlap
    try:
        return overlap / union, f1, p, r #overlap / norm
    except ZeroDivisionError:
        return 0,0,0,0


def align_vars_baseline(alignment, amr, firstamr):
    pattern = "\w\d \/ [\w\d-]*"
    # changed regex due to anonymous data
    print("First AMR: ", firstamr)
    print("Second AMR: ", amr)
    matches_amr1 = re.findall("\w*\d \/ [\d-]*", firstamr)
    matches_amr2 = re.findall("\w*\d \/ [\d-]*", amr)
    alig = ""
    #for i in range(len(matches_amr1)):
    #for match in matches_amr1:
    for i in range(len(matches_amr1)):
        #match = re.findall("\/ (\w+)-", match)
        leaf =matches_amr1[i].split("/")[1]
        #leaf = match.split("/")[1]
        for m2 in matches_amr2:
            if leaf in m2:
                #print(matches_amr1[i], leaf, m2)
                var1 = matches_amr1[i].split()[0]
                var2 = m2.split()[0]
                #print("var1: ", var1)
                #print("var2: ", var2)
                if i != len(matches_amr1)-1:
                    alig += var1 + " " + var2 + " | "
                else:
                    alig += var1 + " " + var2
    #print(alig)
    return alig


def compute_overlap_sample_constrained(alignments, firstamrs, secondamrs):
    scores = []
    f1_scores = []
    p_scores = []
    r_scores = []
    lower_bound = 35 # 35
    upper_bound = 130 # 130
    non_evaluated = []
    for i, amr in enumerate(firstamrs):
        if len(amr) + len(secondamrs[i]) <= 9999:
            num_amr1 = sum(map(str.isdigit, amr))
            num_amr2 = sum(map(str.isdigit, secondamrs[i]))
            if num_amr1 > lower_bound and num_amr2 > lower_bound and num_amr1 < upper_bound and num_amr2 < upper_bound:
                score, f1, p, r = compute_overlap_pair(alignments[i], amr, secondamrs[i])
                scores.append(score)
                f1_scores.append(f1)
                p_scores.append(p)
                r_scores.append(r)
            else:
                non_evaluated.append([amr, secondamrs[i]])
                #continue
            #print(i, amr)
        else:
            continue

    return scores, f1_scores, p_scores, r_scores


def compute_overlap_sample(alignments, firstamrs, secondamrs, make_baseline):
    scores = []
    f1_scores = []
    p_scores = []
    r_scores = []
    indices = []
    for i, amr in enumerate(firstamrs):
        if len(amr) + len(secondamrs[i]) <= 99999:
            #print(i, len(alignments), len(secondamrs))
            score, f1, p, r = compute_overlap_pair(alignments[i], amr, secondamrs[i], make_baseline)
            scores.append(score)
            f1_scores.append(f1)
            p_scores.append(p)
            r_scores.append(r)
            indices.append(i)
            #print(i, amr)
        else:
            continue
    pickle.dump(indices, open("indices.p", "wb"))

    return scores, f1_scores, p_scores, r_scores



def eval(gold_align, pred_align, amr1, amr2, study_bool, make_baseline):
    # lade die predicted alignments und dazugehörige AMRs
    # lade die gold alignments und dazugehörige AMRs
    # berechne separat die scores
    # also einfach:


    print(amr1[0])
    print(amr2[0])
    if study_bool:
        gold, f1_scores_g, p_scores_g, r_scores_g = compute_overlap_sample_constrained(gold_align, amr1, amr2)
        test_source = read_file_per_line("a_test.src")
        amr1, amr2 = separator(test_source)
        pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample_constrained(pred_align, amr1, amr2)

    else:
        gold, f1_scores_g, p_scores_g, r_scores_g = compute_overlap_sample(gold_align, amr1, amr2, False)
        test_source = read_file_per_line("a_test.src")
        test_source = filter_list(test_source, index)


        amr1, amr2 = separator(test_source)
        # 16/02: Meeting: pred_align mit gold_align getauscht
        if make_baseline:
            pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample(gold_align, amr1, amr2, make_baseline)
        else:
            pred, f1_scores_p, p_scores_p, r_scores_p = compute_overlap_sample(pred_align, amr1, amr2, make_baseline)


    #pearson = stats.pearsonr(pred, gold)
    #print("Pearson Coeff: ", pearson)
    print(len(f1_scores_p), len(f1_scores_g))
    f1_pred = sum(f1_scores_p) / (len(f1_scores_p))
    prec_pred = sum(p_scores_p) / (len(p_scores_p))
    rec_pred = sum(r_scores_p) / (len(r_scores_p))
    std_dev_pred = np.std(pred)
    pearson_1 = stats.pearsonr(f1_scores_p, f1_scores_g)[0]
    pearson_2 = stats.pearsonr(f1_scores_p, f1_scores_g)[1]
    print("F1-Gold: ", sum(f1_scores_g)/(len(f1_scores_g)))
    print("F1-Pred: ", sum(f1_scores_p) / (len(f1_scores_p)))
    print("Prec-Gold: ", sum(p_scores_g)/(len(p_scores_g)))
    print("Prec-Pred: ", sum(p_scores_p) / (len(p_scores_p)))
    print("Rec-Gold: ", sum(r_scores_g)/(len(r_scores_g)))
    print("Rec-Pred: ", sum(r_scores_p) / (len(r_scores_p)))
    print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p, f1_scores_g))
   #print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p[:1000], f1_scores_g[:1000]))
    print("std dev gold: ", np.std(gold))
    print("std dev pred: ", np.std(pred))
    #print(f1_scores_g)
    #print(f1_scores_p)
    #print("G:", f1_scores_g)
    #print("Pearson F1 Scores: ", stats.pearsonr(f1_scores_p, f1_scores_g))
    #print(len(f1_scores_p))
    #print(len(f1_scores_g))
    return f1_pred, prec_pred, rec_pred, std_dev_pred, pearson_1, pearson_2
    #print(pearsonr(compute_overlap_sample(....), compute_overlap_sample(....)))


def separator(data):
    amr1 = []
    amr2 = []
    amr1_l = []
    amr2_l = []
    over_800 = 0
    line_length = []
    for line in data:
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
    print("Len Line AVG: ", sum(line_length)/len(line_length))
    print("Avg Length amr1: ", sum(amr1_l)/len(amr1_l))
    print("Avg Length amr3: ", sum(amr2_l) / len(amr2_l))
    print("Instances over 800: ", over_800)
    return amr1, amr2


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            #if ":snt" in line
            data.append(line)
    return data


def filter_list(data, index):
    return [data[i] for i in index]



if __name__ == "__main__":

    first_a = read_data("../C/gpla_first_test_set_reduced.txt")
    second_b = read_data("../C/gpla_second_test_set_reduced.txt")

    # variable count eval
    index = filter_variable_count(first_a, second_b, 35)



    path = "C:/Users/phMei/PycharmProjects/AMR_ablation/A"
    test_source = read_file_per_line("a_test.src") #"old/a_test_reduced.src")
    gold_align = read_file_per_line("a_test.tgt")
    pred_align = read_file_per_line("00305000.hyps.test")

    test_source = filter_list(test_source, index)
    gold_align = filter_list(gold_align, index)
    pred_align = filter_list(pred_align, index)


    make_baseline = False

    amr1, amr2 = separator(test_source)

    # Length Check
    print("Len AMR1: ", len(amr1))
    print("Len AMR2: ", len(amr2))
    print("Gold Align: ", len(gold_align))
    print("Pred Align: ", len(pred_align))

    f1_pred, prec_pred, rec_pred, std_dev_pred, pearson_1, pearson_2 = 0,0,0,0,0,0
    for i in range(10):
        a,b,c,d,e,f = eval(gold_align, pred_align, amr1, amr2, False, make_baseline)
        f1_pred += a
        prec_pred += b
        rec_pred += c
        std_dev_pred += d
        pearson_1 += e
        pearson_2 += f
    print("Final:")
    print("F1_pred: ", f1_pred/10)
    print("Prec Pred: ", prec_pred / 10)
    print("Rec Pred: ", rec_pred / 10)
    print("STD: ", std_dev_pred / 10)
    print("Pearson1: ", pearson_1 / 10)
    print("Pearson2: ", pearson_2 / 10)

    """
    amr1_name = "combined_amr3.gpla_combined_amr3.txt_first.amr_tokenized.txt_reduced.txt"
    amr2_name = "combined_amr3.gpla_combined_amr3.txt_second.amr_tokenized.txt_reduced.txt"
    #C:phMei\PycharmProjects
    path = "C:/Users/phMei/PycharmProjects/algorithm-synthesis-for-smatch/1c"
    #path = "C:/Users/Meier/PycharmProjects/algorithm-synthesis-for-smatch/1c/" #one_eval/"
    #gold_align = read_file_per_line("test.tgt") #"combined_amr3.gpla_combined_amr3.txt_first.amr_tokenized.txt")
    gold_align = read_file_per_line(path + "test.tgt") #"test.tgt") # "r1_no_Si.tgt") #"test_smatch1.tgt") #path + "test_smatch1.tgt") #"prediction_short_800.txt") #"prediction_short_small.txt") #"predictions_no_shuffle.txt") #"combined_amr3.gpla_combined_amr3.txt_second.amr_tokenized.txt")
    pred_align = read_file_per_line(path + "1c_predictions.txt")
    amr1 = read_file_per_line(path+amr1_name) #path
    amr2 = read_file_per_line(path+amr2_name)
    print(gold_align[0])
    #pred_align = pred_align[:5000]
    # Testdaten in den AMR graphen
    amr1 = amr1[len(amr1)-7000:len(amr1)-3500]
    amr2 = amr2[len(amr2)-7000:len(amr2)-3500]
    print(len(amr1))

    eval(gold_align, pred_align, amr1, amr2, False)
    """