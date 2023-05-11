import sys


def filter_file(filename, n):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for no, line in enumerate(f):
            if no % n == 0:
                data.append(line)
    with open(filename+"_reduced.txt", "w", encoding="utf-8") as f:
        for line in data:
            f.write(line)



if __name__ == "__main__":
    filename = sys.argv[1]
    filter_file(filename, 10)
