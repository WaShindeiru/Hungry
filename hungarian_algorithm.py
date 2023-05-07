from typing import List
import math
from copy import deepcopy, copy

INF = math.inf


def reduce_row(matrix: List[List[int]]):
    fi = 0
    for row in range(len(matrix)):
        min_in_row = min(matrix[row])
        fi += min_in_row

        for i in range(len(matrix)):
            matrix[row][i] -= min_in_row
    return matrix, fi


def reduce_column(matrix: List[List[int]], fi: int):
    for col in range(len(matrix[0])):
        col_values = [matrix[row][col] for row in range(len(matrix))]
        min_in_col = min(col_values)
        fi += min_in_col

        for row in range(len(matrix)):
            matrix[row][col] -= min_in_col
    return matrix, fi


def get_col(matrix, col_index):
    return [row[col_index] for row in matrix]


def set_col(matrix, col_index, val):
    for row_it in range(len(matrix)):
        matrix[row_it][col_index] = val


def find_zeros(reduced_matrix):
    matrix_row_size = len(reduced_matrix)
    matrix_col_size = len(reduced_matrix[0])

    zero_matrix = [[0 for j in range(matrix_col_size)] for i in range(matrix_row_size)]

    for row_it in range(matrix_row_size):
        for col_it in range(matrix_col_size):
            if reduced_matrix[row_it][col_it] == 0:
                zero_matrix[row_it][col_it] = 1 if 1 not in zero_matrix[row_it] and 1 not in get_col(zero_matrix,
                                                                                                     col_it) else 2
    return zero_matrix


def cross_matrix(zero_matrix):
    matrix_row_size = len(zero_matrix)
    matrix_col_size = len(zero_matrix[0])

    def find_path(row_index, col_index):
        starred_zeros_coords = []
        primed_zeros_coords = []

        def substep_one(row, col):
            column = get_col(zero_matrix, col)
            if 1 in column:
                starred_zero_index = column.index(1)
                starred_zeros_coords.append((starred_zero_index, col))
                substep_two(starred_zero_index)

            else:
                return

        def substep_two(starred_zero_row):
            if 3 in zero_matrix[starred_zero_row]:
                primed_zeros_coords.append((starred_zero_row, zero_matrix[starred_zero_row].index(3)))
                substep_one(starred_zero_row, zero_matrix[starred_zero_row].index(3))

            else:
                print('Problem')
                return

        primed_zeros_coords.append((row_index, col_index))
        substep_one(row_index, col_index)
        return starred_zeros_coords, primed_zeros_coords

    enable_first_step = True

    while True:

        break_twice = False

        if enable_first_step:
            covered_rows = set()
            covered_cols = set()

            # Cover all columns containing starred zero
            for col_it in range(matrix_col_size):
                if 1 in get_col(zero_matrix, col_it):
                    covered_cols.add(col_it)

        found_not_covered = False

        for row_it in range(matrix_row_size):
            for col_it in range(matrix_col_size):
                if zero_matrix[row_it][col_it] == 2 and row_it not in covered_rows and col_it not in covered_cols:
                    found_not_covered = True
                    zero_matrix[row_it][col_it] = 3

                    if 1 in zero_matrix[row_it]:
                        covered_rows.add(row_it)
                        covered_cols.discard(zero_matrix[row_it].index(1))
                        enable_first_step = False
                        break_twice = True
                        break

                    else:
                        starred_zeros_coords, primed_zeros_coords = find_path(row_it, col_it)

                        for primed_zero_coord in primed_zeros_coords:
                            zero_matrix[primed_zero_coord[0]][primed_zero_coord[1]] = 1

                        for starred_zero_coord in starred_zeros_coords:
                            zero_matrix[starred_zero_coord[0]][starred_zero_coord[1]] = 2

                        for row_it2 in range(matrix_row_size):
                            for col_it2 in range(matrix_col_size):
                                if zero_matrix[row_it2][col_it2] == 3:
                                    zero_matrix[row_it2][col_it2] = 2
                        enable_first_step = True
                        break_twice = True
                        break
            if break_twice:
                break

        if not found_not_covered:
            break

    return zero_matrix, covered_rows, covered_cols


def change_matrix(reduced_matrix, covered_rows, covered_cols, fi):
    matrix_row_size = len(reduced_matrix)
    matrix_col_size = len(reduced_matrix[0])

    minimal_value = INF
    for row_it in range(matrix_row_size):
        for col_it in range(matrix_col_size):
            if row_it not in covered_rows and col_it not in covered_cols:
                if reduced_matrix[row_it][col_it] < minimal_value:
                    minimal_value = reduced_matrix[row_it][col_it]

    for row_it in range(matrix_row_size):
        for col_it in range(matrix_col_size):
            if row_it not in covered_rows and col_it not in covered_cols:
                reduced_matrix[row_it][col_it] -= minimal_value
            elif row_it in covered_rows and col_it in covered_cols:
                reduced_matrix[row_it][col_it] += minimal_value
            else:
                continue
    
    if minimal_value != INF:
        fi += minimal_value

    return reduced_matrix, fi


def hungarian_algorithm(matrix):
    matrix_row_size = len(matrix)
    matrix_col_size = len(matrix[0])

    # Nie mam pojęcia czemu .copy tworzy tutaj deep copy więc póki co zostawiam taki egzorcyzm
    cloned_matrix = [[0 for j in range(matrix_col_size)] for i in range(matrix_row_size)]
    for row_it in range(matrix_row_size):
        for col_it in range(matrix_col_size):
            cloned_matrix[row_it][col_it] = matrix[row_it][col_it]

    # Robione na podstawie wikipedii
    # https://en.wikipedia.org/wiki/Hungarian_algorithm

    # Step 1
    reduced_row_matrix, fi = reduce_row(cloned_matrix)
    reduced_row_copy_matrix = deepcopy(reduced_row_matrix)

    # Step 2
    reduced_matrix, fi2 = reduce_column(reduced_row_copy_matrix, fi)

    # Total Cost
    totalCost = fi2

    # Step 3
    reduced_matrix_copy = deepcopy(reduced_matrix)
    zero_matrix = find_zeros(reduced_matrix_copy)

    # Step 4
    new_zero_matrix, covered_rows, covered_cols = cross_matrix(zero_matrix)

    # Step 5
    final_matrix, totalCost = change_matrix(reduced_matrix_copy, covered_rows, covered_cols, totalCost)

    status = 'Completed'

    # Żeby nie wpadać w nieskończoną pętle, zakładam, że jeżeli po 20 iteracjach algorytm nie jest w stanie znaleźć
    # przydziału to oznacza że coś jest nie tak
    loop_counter = 0
    while len(covered_rows) + len(covered_cols) < len(matrix):
        zero_matrix = find_zeros(final_matrix)
        new_zero_matrix, covered_rows, covered_cols = cross_matrix(zero_matrix)
        final_matrix, totalCost = change_matrix(reduced_matrix_copy, covered_rows, covered_cols, totalCost)
        loop_counter += 1

        if loop_counter > 20:
            status = 'Failed'
            break

    result_matrix = [['' for j in range(matrix_col_size)] for i in range(matrix_row_size)]

    for row_it in range(matrix_row_size):
        for col_it in range(matrix_col_size):
            if new_zero_matrix[row_it][col_it] == 1:
                result_matrix[row_it][col_it] = matrix[row_it][col_it]

    return result_matrix, final_matrix, new_zero_matrix, covered_rows, covered_cols, status, totalCost, reduced_row_matrix, reduced_matrix


def list2string_matrix(matrix):
    matrix_string = ''
    for row_iterator in range(len(matrix)):
        row_string = ''
        for col_iterator in range(len(matrix[row_iterator])):
            row_string += '{:3}'.format(matrix[row_iterator][col_iterator])
        matrix_string += row_string + '\n'
    return matrix_string


def print_assignment(matrix):
    result, modified, zeros, rows, cols, status, totalCost, reduced_row_matrix, reduced_matrix = hungarian_algorithm(matrix)
    print('Oryginalna macierz: ')
    print(list2string_matrix(matrix))
    print("Macierz zredukowana wierszowo: ")
    print(list2string_matrix(reduced_row_matrix))
    print("Macierz zredukowana wierszowo oraz kolumnowo: ")
    print(list2string_matrix(reduced_matrix))
    print("Macierz po przekształceniach: ")
    print(list2string_matrix(modified))
    print("Wyznaczony przydział: ")
    print(list2string_matrix(result))
    print("Macierz zer (1 - zero niezależne, 2 - zero zależne, 3 - zero prim)")
    print(list2string_matrix(zeros))
    print("Koszt: ")
    print(totalCost)
    print('Wykreślone wiersze: ')
    print(rows)
    print('Wykreślone kolumny: ')
    print(cols)
    print('Status')
    print(status + '\n')


# Matrix = [[5, 2, 3, 2, 7],
#           [6, 8, 4, 2, 5],
#           [6, 4, 3, 7, 2],
#           [6, 9, 0, 4, 0],
#           [4, 1, 2, 4, 0]]
# print_assignment(Matrix)

# Znalazłem taką macierz na stronie https://www.hungarianalgorithm.com/examplehungarianalgorithm.php
# i rozwiązanie programu zgadza się z rozwiązaniem na stronie
# Matrix2 = [[82, 83, 69, 92], [77, 37, 49, 92], [11, 69, 5, 86], [8, 9, 98, 23]]
# print_assignment(Matrix2)

matrix_last = [[91, 83, 22, 82, 1, 44], [68, 6, 94, 95, 88, 94], [7, 29, 4, 62, 20, 36],
               [15, 90, 52, 4, 39, 89], [42, 88, 80, 61, 3, 98], [42, 82, 9, 47, 39, 93]]
print_assignment(matrix_last)

# cleaningMatrix = [[8, 4, 7], [5, 2, 3], [9, 4, 8]]
# print_assignment(cleaningMatrix)

# Matrix3 = [[random.randint(1, 25) for j in range(5)] for i in range(5)]
# print_assignment(Matrix3)

# Macierz psujaca algorytm (już naprawiłem i działa)
# Matrix4 = [[0, 4, 9, 10, 0], [8, 7, 13, 0, 0], [19, 14, 0, 0, 7], [15, 0, 6, 2, 4], [9, 14, 11, 0, 9]]
# print_assignment(Matrix4)

# Matrix5 = [[random.randint(1, 9) for j in range(10)] for i in range(10)]
# print_assignment(Matrix5)

# Matrix6 = [[7, 5, 2, 7, 7, 1, 8, 7, 9, 8], [1, 8, 3, 8, 1, 9, 9, 4, 4, 8], [5, 9, 8, 6, 5, 6, 1, 8, 5, 3], [1, 6, 3, 2, 8, 5, 2, 9, 8, 9], [2, 1, 9, 2, 8, 6, 1, 7, 5, 8], [6, 5, 8, 6, 1, 3, 7, 6, 7, 8], [4, 9, 9, 7, 6, 1, 3, 7, 1, 3], [8, 7, 9, 9, 1, 9, 7, 1, 7, 9], [5, 1, 2, 8, 8, 9, 8, 5, 9, 7], [5, 1, 3, 5, 6, 3, 7, 8, 8, 2]]
# print_assignment(Matrix6)

# https://math.stackexchange.com/questions/4415703/why-my-hungarian-algorithm-isnt-working-in-specific-cases
# Matrix7 = [[0, 0, 0], [0, 83, 58], [0, 72, 10]]
# print_assignment(Matrix7)
