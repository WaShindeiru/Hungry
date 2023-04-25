## Hungary
from typing import List


def reduce_row(matrix: List[List[int]]):

    fi = 0
    for row in range(len(matrix)):
        min_in_row = min(matrix[row])
        fi += min_in_row

        for i in range(len(matrix)):
            matrix[row][i] -= min_in_row
    return matrix, fi

Matrix = [[5, 2, 3, 2, 7],
          [6, 8, 4, 2, 5],
          [6, 4, 3, 7, 2],
          [6, 9, 0, 4, 0],
          [4, 1, 2, 4, 0]]
aaaa = reduce_row(Matrix)
print(aaaa[0], aaaa[1])

