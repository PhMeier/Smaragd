"""
This module randmizes source and target files in the same way and
creates a development and test set.
@author: Philipp Meier
@date 11/03/21
"""
import random
import sys


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


def randomize_and_split(x, y):
    """
    Randomizes source and target equally and splits into dev and test.
    :param x:
    :param y:
    :return:
    """
    seed = 24
    random.seed(seed)
    comb = list(zip(x, y))
    #random.shuffle(comb)
    #print(comb)
    #print(len(comb)*0.7)

    train = comb[:len(comb) - 18499]
    test_set = comb[len(comb) - 18499:len(comb) - 3499]  # [int(len(comb)*0.7):int(len(comb)*0.8)]
    dev_set = comb[len(comb) - 3499:]  # [int(len(comb)*0.8):]

    """
    train = comb[:len(comb)-7000]
    test_set =comb[len(comb)-7000:len(comb)-3500] #[int(len(comb)*0.7):int(len(comb)*0.8)]
    dev_set = comb[len(comb)-3500:] #[int(len(comb)*0.8):]
    """
    return train, dev_set, test_set


def combine_sources(source1, source2):
    """
    This function combines first.amr and second.amr in one string, separated with SEP
    :param source1:
    :param source2:
    :return:
    """
    new_data = []
    for line1, line2 in zip(source1, source2):
        line1 = line1.replace("\n", "")
        new_line = line1 + " SEP " + line2
        new_data.append(new_line)
    return new_data


def write_to_file(filename_source, filename_target, data):
    with open(filename_source, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(line[0])
    with open(filename_target, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(line[1])


if __name__ == "__main__":
    """
    Takes in amr_first, amr_second and alignment
    """
    source = sys.argv[1]
    source2 = sys.argv[2]
    target = sys.argv[3]
    x = read_file_per_line(source)
    x2 = read_file_per_line(source2)
    y = read_file_per_line(target)
    combined_data = combine_sources(x, x2)
    train, dev, test = randomize_and_split(combined_data, y)

    write_to_file("a_train.src", "a_train.tgt", train)
    write_to_file("a_dev.src", "a_dev.tgt", dev)
    write_to_file("a_test.src", "a_test.tgt", test)

    """
    write_to_file("train.src", "train.tgt", train)
    write_to_file("dev.src", "dev.tgt", dev)
    write_to_file("test.src", "r1_no_Si.tgt", test)
    """
