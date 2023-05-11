import re


def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data.append(line)
    return data


def filter_variable_count(first, second, threshold):
    index = []
    for i, (amr1, amr2) in enumerate(zip(first, second)):
        num_amr1 = len(re.findall("a[0-9]+", amr1))
        num_amr2 = len(re.findall("b[0-9]+", amr2))
        if num_amr1 > threshold or num_amr2 > threshold:
            #print(num_amr1, num_amr2)
            index.append(i)
    return index


if __name__ == "__main__":
    first_a = read_data("../C/gpla_first_test_set_reduced.txt")
    second_b = read_data("../C/gpla_second_test_set_reduced.txt")
    #print(second_b)
    #filter_variable_count(first_a, second_b, 15)
    """
    mutlisents = list()
    sec = []


    for i in range(len(first_a)):
        if ":snt" in first_a[i]:
            #print(first_a[i])
            mutlisents.append(i)

    for i in range(len(second_b)):
        if ":snt" in second_b[i]:
            #print(second_b[i])
            sec.append(i)
            if i not in mutlisents:
                mutlisents.append(i)
    print(len(mutlisents))
    print(len(sec))
    print(len(set(mutlisents).intersection(set(sec))))

    """

