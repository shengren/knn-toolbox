#!/usr/bin/python

import sys

sys.argv.pop(0)
assert len(sys.argv) == 2

filename_a = sys.argv.pop(0)
filename_b = sys.argv.pop(0)
result_a = [line.strip() for line in open(filename_a)]
result_b = [line.strip() for line in open(filename_b)]
assert len(result_a) == len(result_b)
num_query = len(result_a)

accuracy = 0.0
for i in range(num_query):
    knn_a = result_a[i].split()
    knn_b = result_b[i].split()
    assert len(knn_a) == len(knn_b)
    k = len(knn_a)
    num_match = k
    for j in range(k):
        if knn_a[j] != knn_b[j]:
            num_match -= 1
    accuracy += num_match / float(k)
accuracy /= num_query
print '{0:.1f}% match'.format(accuracy * 100)
