### Problem 1: Dot Product of Two Vectors

def dot_product(a, b):
    # UNVECTORIZE THIS
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result
#TAGS: loops, multiplication

### Problem 2: Matrix-Vector Multiplication

def matrix_vector_multiplication(A, v):
    # UNVECTORIZE THIS
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j] * v[j]
    return result
#TAGS: loops, multiplication, nested loops

### Problem 3: Matrix Transposition

def transpose(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 4: Matrix Addition

def matrix_addition(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
    return result
#TAGS: loops, addition, nested loops

### Problem 5: Scalar Multiplication of Matrix

def scalar_multiplication(A, s):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * s
    return result
#TAGS: loops, multiplication, nested loops

### Problem 6: Row-wise Sum

def row_sum(A):
    # UNVECTORIZE THIS
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
    return result
#TAGS: loops, addition, nested loops

### Problem 7: Column-wise Sum

def column_sum(A):
    # UNVECTORIZE THIS
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
    return result
#TAGS: loops, addition, nested loops

### Problem 8: Diagonal of a Matrix

def diagonal(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(len(A)):
        result.append(A[i][i])
    return result
#TAGS: loops

### Problem 9: Matrix Multiplication

def matrix_multiplication(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result
#TAGS: loops, multiplication, triple nested loops

### Problem 10: Elementwise Multiplication

def elementwise_multiplication(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * B[i][j]
    return result
#TAGS: loops, multiplication, nested loops

### Problem 11: Row-wise Mean

def row_mean(A):
    # UNVECTORIZE THIS
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
        result[i] /= len(A[0])
    return result
#TAGS: loops, addition, nested loops, division

### Problem 12: Column-wise Mean

def column_mean(A):
    # UNVECTORIZE THIS
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
        result[j] /= len(A)
    return result
#TAGS: loops, addition, nested loops, division

### Problem 13: Matrix Trace

def matrix_trace(A):
    # UNVECTORIZE THIS
    result = 0
    for i in range(len(A)):
        result += A[i][i]
    return result
#TAGS: loops, addition

### Problem 14: Elementwise Division

def elementwise_division(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / B[i][j]
    return result
#TAGS: loops, division, nested loops

### Problem 15: Matrix Subtraction

def matrix_subtraction(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] - B[i][j]
    return result
#TAGS: loops, subtraction, nested loops

### Problem 16: Row-wise Maximum

def row_max(A):
    # UNVECTORIZE THIS
    result = [float('-inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > result[i]:
                result[i] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 17: Column-wise Maximum

def column_max(A):
    # UNVECTORIZE THIS
    result = [float('-inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] > result[j]:
                result[j] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 18: Matrix Scalar Division

def scalar_division(A, s):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / s
    return result
#TAGS: loops, division, nested loops

### Problem 19: Check Square Matrix

def is_square(A):
    # UNVECTORIZE THIS
    return len(A) == len(A[0])
#TAGS: loops, comparison

### Problem 20: Elementwise Power

def elementwise_power(A, p):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** p
    return result
#TAGS: loops, power, nested loops

### Problem 21: Row-wise Minimum

def row_min(A):
    # UNVECTORIZE THIS
    result = [float('inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < result[i]:
                result[i] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 22: Column-wise Minimum

def column_min(A):
    # UNVECTORIZE THIS
    result = [float('inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] < result[j]:
                result[j] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 23: Matrix Negation

def negate_matrix(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = -A[i][j]
    return result
#TAGS: loops, negation, nested loops

### Problem 24: Count Positive Numbers

def count_positive(A):
    # UNVECTORIZE THIS
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > 0:
                count += 1
    return count
#TAGS: loops, nested loops, comparison

### Problem 25: Count Negative Numbers

def count_negative(A):
    # UNVECTORIZE THIS
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0:
                count += 1
    return count
#TAGS: loops, nested loops, comparison

### Problem 26: Zero Matrix

def zero_matrix(rows, cols):
    # UNVECTORIZE THIS
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        result.append(row)
    return result
#TAGS: loops, nested loops

### Problem 27: Identity Matrix

def identity_matrix(n):
    # UNVECTORIZE THIS
    result = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1)
            else:
                row.append(0)
        result.append(row)
    return result
#TAGS: loops, nested loops, comparison

### Problem 28: Flatten Matrix

def flatten_matrix(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            result.append(A[i][j])
    return result
#TAGS: loops, nested loops

### Problem 29: Check Symmetry

def is_symmetric(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != A[j][i]:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 30: Matrix Elementwise Absolute

def elementwise_absolute(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = abs(A[i][j])
    return result
#TAGS: loops, nested loops, absolute

### Problem 31: Replace Diagonal

def replace_diagonal(A, d):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        A[i][i] = d[i]
    return A
#TAGS: loops

### Problem 32: Upper Triangular Matrix

def upper_triangular(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i, len(A[0])):
            result[i][j] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 33: Lower Triangular Matrix

def lower_triangular(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i+1):
            result[i][j] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 34: Count Zeroes in Matrix

def count_zeroes(A):
    # UNVECTORIZE THIS
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 0:
                count += 1
    return count
#TAGS: loops, nested loops, comparison

### Problem 35: Check Lower Triangular

def is_lower_triangular(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            if A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 36: Check Upper Triangular

def is_upper_triangular(A):
    # UNVECTORIZE THIS
    for i in range(1, len(A)):
        for j in range(i):
            if A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 37: Check Diagonal Matrix

def is_diagonal_matrix(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 38: Rotate Matrix 90 degrees

def rotate_90(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][len(A)-1-i] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 39: Matrix Elementwise Square Root

def elementwise_sqrt(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** 0.5
    return result
#TAGS: loops, nested loops, power

### Problem 40: Replace Negative Numbers with Zero

def replace_negatives_with_zero(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0:
                A[i][j] = 0
    return A
#TAGS: loops, nested loops, comparison

### Problem 41: Matrix Determinant (for 2x2 matrix)

def determinant_2x2(A):
    # UNVECTORIZE THIS
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]
#TAGS: multiplication, subtraction

### Problem 42: Copy Matrix

def copy_matrix(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 43: Matrix Scaling

def scale_matrix(A, scale_factor):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * scale_factor
    return result
#TAGS: loops, nested loops, multiplication

### Problem 44: Matrix Reflection (Reflect over main diagonal)

def reflect_matrix(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[j][i]
    return result
#TAGS: loops, nested loops

### Problem 45: Matrix Shearing (Add multiples of one row to another)

def shear_matrix(A, k, i, j):
    # UNVECTORIZE THIS (i-th row is changed by adding k times the j-th row)
    for col in range(len(A[0])):
        A[i][col] += k * A[j][col]
    return A
#TAGS: loops, multiplication, addition

### Problem 46: Count Non-zero elements in Matrix

def count_non_zero(A):
    # UNVECTORIZE THIS
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != 0:
                count += 1
    return count
#TAGS: loops, nested loops, comparison

### Problem 47: Matrix Inverse (for 2x2 matrix)

def inverse_2x2(A):
    # UNVECTORIZE THIS
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    inv_det = 1 / det
    result = [[A[1][1] * inv_det, -A[0][1] * inv_det],
              [-A[1][0] * inv_det, A[0][0] * inv_det]]
    return result
#TAGS: multiplication, subtraction, division

### Problem 48: Check if Matrix is Invertible (for 2x2 matrix)

def is_invertible_2x2(A):
    # UNVECTORIZE THIS
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return det != 0
#TAGS: multiplication, subtraction, comparison

### Problem 49: Matrix Translation (Add a constant to every element)

def translate_matrix(A, c):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + c
    return result
#TAGS: loops, nested loops, addition

### Problem 50: Elementwise Natural Logarithm

def elementwise_log(A):
    # UNVECTORIZE THIS
    import math
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = math.log(A[i][j])
    return result
#TAGS: loops, nested loops, logarithm

### Problem 51: Row-wise Summation

def row_sum(A):
    # UNVECTORIZE THIS
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
    return result
#TAGS: loops, addition, nested loops

### Problem 52: Column-wise Summation

def column_sum(A):
    # UNVECTORIZE THIS
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
    return result
#TAGS: loops, addition, nested loops

### Problem 53: Check Zero Matrix

def is_zero_matrix(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 54: Row-wise Product

def row_product(A):
    # UNVECTORIZE THIS
    result = [1] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] *= A[i][j]
    return result
#TAGS: loops, multiplication, nested loops

### Problem 55: Matrix Frobenius Norm

def frobenius_norm(A):
    # UNVECTORIZE THIS
    result = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            result += A[i][j]**2
    return result**0.5
#TAGS: loops, nested loops, power, square root

### Problem 56: Count Matrix Elements Greater than a Value

def count_greater_than(A, value):
    # UNVECTORIZE THIS
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > value:
                count += 1
    return count
#TAGS: loops, nested loops, comparison

### Problem 57: Matrix Elementwise Modulo

def elementwise_modulo(A, mod_value):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] % mod_value
    return result
#TAGS: loops, nested loops, modulo

### Problem 58: Replace Matrix Elements Less than a Value

def replace_less_than(A, value, replacement):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < value:
                A[i][j] = replacement
    return A
#TAGS: loops, nested loops, comparison

### Problem 59: Check if All Elements in Matrix Are Equal

def all_elements_equal(A):
    # UNVECTORIZE THIS
    value = A[0][0]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != value:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 60: Matrix Rank (Number of Non-zero Rows)

def matrix_rank(A):
    # UNVECTORIZE THIS
    rank = 0
    for i in range(len(A)):
        non_zero_row = False
        for j in range(len(A[0])):
            if A[i][j] != 0:
                non_zero_row = True
                break
        if non_zero_row:
            rank += 1
    return rank
#TAGS: loops, nested loops, comparison

### Problem 61: Check if Matrix is Orthogonal

def is_orthogonal(A):
    # UNVECTORIZE THIS
    # Assuming A is a square matrix
    for i in range(len(A)):
        for j in range(len(A)):
            dot_product = 0
            for k in range(len(A)):
                dot_product += A[i][k] * A[j][k]
            if i == j and dot_product != 1:
                return False
            elif i != j and dot_product != 0:
                return False
    return True
#TAGS: loops, triple nested loops, multiplication, addition, comparison

### Problem 62: Matrix Anti-Diagonal (Elements from top right to bottom left)

def anti_diagonal(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(len(A)):
        result.append(A[i][len(A)-1-i])
    return result
#TAGS: loops

### Problem 63: Rotate Matrix 180 degrees

def rotate_180(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[len(A)-1-i][len(A[0])-1-j] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 64: Row-wise Sorting

def sort_rows(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        A[i] = sorted(A[i])
    return A
#TAGS: loops, sorting

### Problem 65: Column-wise Sorting

def sort_columns(A):
    # UNVECTORIZE THIS
    for j in range(len(A[0])):
        # Extract column
        col = [A[i][j] for i in range(len(A))]
        # Sort column
        col = sorted(col)
        # Replace old column with sorted column
        for i in range(len(A)):
            A[i][j] = col[i]
    return A
#TAGS: loops, nested loops, sorting

### Problem 66: Matrix Cumulative Sum

def matrix_cumsum(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + (result[i-1][j] if i > 0 else 0) + (result[i][j-1] if j > 0 else 0) - (result[i-1][j-1] if i > 0 and j > 0 else 0)
    return result
#TAGS: loops, nested loops, addition, subtraction

### Problem 67: Matrix Reversal

def reverse_matrix(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[len(A)-1-i][len(A[0])-1-j]
    return result
#TAGS: loops, nested loops

### Problem 68: Matrix Upside Down

def upside_down_matrix(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[len(A)-1-i][j]
    return result
#TAGS: loops, nested loops

### Problem 69: Extract Unique Elements

def unique_elements(A):
    # UNVECTORIZE THIS
    unique_set = set()
    for i in range(len(A)):
        for j in range(len(A[0])):
            unique_set.add(A[i][j])
    return list(unique_set)
#TAGS: loops, nested loops, set

### Problem 70: Check for Identity Matrix

def is_identity(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == j and A[i][j] != 1:
                return False
            elif i != j and A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 71: Replace Matrix Elements Greater than a Value

def replace_greater_than(A, value, replacement):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > value:
                A[i][j] = replacement
    return A
#TAGS: loops, nested loops, comparison

### Problem 72: Check if Matrix is Toeplitz

def is_toeplitz(A):
    # UNVECTORIZE THIS
    for i in range(1, len(A)):
        for j in range(1, len(A[0])):
            if A[i][j] != A[i-1][j-1]:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 73: Matrix Main Diagonal Product

def diagonal_product(A):
    # UNVECTORIZE THIS
    product = 1
    for i in range(len(A)):
        product *= A[i][i]
    return product
#TAGS: loops, multiplication

### Problem 74: Check if Matrix is Scalar

def is_scalar(A):
    # UNVECTORIZE THIS
    value = A[0][0]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
            elif i == j and A[i][j] != value:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 75: Extract Upper Triangle Elements

def upper_triangle_elements(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            result.append(A[i][j])
    return result
#TAGS: loops, nested loops

### Problem 76: Extract Lower Triangle Elements

def lower_triangle_elements(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(1, len(A)):
        for j in range(i):
            result.append(A[i][j])
    return result
#TAGS: loops, nested loops

### Problem 77: Check if Matrix is Normal (A*A = A^2)

def is_normal(A):
    # UNVECTORIZE THIS
    product_matrix = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            for k in range(len(A)):
                product_matrix[i][j] += A[i][k] * A[k][j]

    for i in range(len(A)):
        for j in range(len(A[0])):
            if product_matrix[i][j] != A[i][j] * A[i][j]:
                return False
    return True
#TAGS: loops, triple nested loops, multiplication, comparison

### Problem 78: Matrix Anti-diagonal Sum

def anti_diagonal_sum(A):
    # UNVECTORIZE THIS
    result = 0
    for i in range(len(A)):
        result += A[i][len(A[0])-1-i]
    return result
#TAGS: loops

### Problem 79: Matrix Left Rotation (Rotate Columns to the Left)

def left_rotate_matrix(A, k):
    # UNVECTORIZE THIS
    num_cols = len(A[0])
    k %= num_cols
    result = [[0] * num_cols for _ in range(len(A))]
    for j in range(num_cols):
        for i in range(len(A)):
            result[i][j] = A[i][(j + k) % num_cols]
    return result
#TAGS: loops, nested loops, modulo

### Problem 80: Matrix Right Rotation (Rotate Columns to the Right)

def right_rotate_matrix(A, k):
    # UNVECTORIZE THIS
    num_cols = len(A[0])
    k %= num_cols
    result = [[0] * num_cols for _ in range(len(A))]
    for j in range(num_cols):
        for i in range(len(A)):
            result[i][j] = A[i][(j - k + num_cols) % num_cols]
    return result
#TAGS: loops, nested loops, modulo

### Problem 81: Matrix Scalar Addition

def matrix_scalar_addition(A, scalar):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + scalar
    return result
#TAGS: loops, nested loops, addition

### Problem 82: Check if Matrix is Symmetric

def is_symmetric(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(i):
            if A[i][j] != A[j][i]:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 83: Matrix Main Diagonal Average

def diagonal_average(A):
    # UNVECTORIZE THIS
    total = 0
    for i in range(len(A)):
        total += A[i][i]
    return total / len(A)
#TAGS: loops, addition, division

### Problem 84: Sum of Border Elements of Matrix

def border_sum(A):
    # UNVECTORIZE THIS
    total = 0
    for i in range(len(A)):
        total += A[i][0] + A[i][-1]
    for j in range(1, len(A[0]) - 1):
        total += A[0][j] + A[-1][j]
    return total
#TAGS: loops, addition

### Problem 85: Matrix Transpose

def transpose(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 86: Matrix Scalar Multiplication

def matrix_scalar_multiplication(A, scalar):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * scalar
    return result
#TAGS: loops, nested loops, multiplication

### Problem 87: Check if Matrix is Lower Triangular

def is_lower_triangular(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            if A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 88: Matrix Elementwise Power

def elementwise_power(A, power):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** power
    return result
#TAGS: loops, nested loops, power

### Problem 89: Row-wise Maximum

def row_max(A):
    # UNVECTORIZE THIS
    result = [float('-inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > result[i]:
                result[i] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 90: Column-wise Minimum

def column_min(A):
    # UNVECTORIZE THIS
    result = [float('inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] < result[j]:
                result[j] = A[i][j]
    return result
#TAGS: loops, nested loops, comparison

### Problem 91: Check if Matrix is Magic Square

def is_magic_square(A):
    # UNVECTORIZE THIS
    if len(A) != len(A[0]):
        return False

    total = sum(A[0])
    for i in range(1, len(A)):
        if sum(A[i]) != total:
            return False

    for j in range(len(A[0])):
        col_sum = 0
        for i in range(len(A)):
            col_sum += A[i][j]
        if col_sum != total:
            return False

    diag_sum1 = sum([A[i][i] for i in range(len(A))])
    diag_sum2 = sum([A[i][len(A[0])-1-i] for i in range(len(A))])

    return diag_sum1 == total and diag_sum2 == total
#TAGS: loops, nested loops, addition, comparison

### Problem 92: Matrix Anti-diagonal Product

def anti_diagonal_product(A):
    # UNVECTORIZE THIS
    product = 1
    for i in range(len(A)):
        product *= A[i][len(A[0])-1-i]
    return product
#TAGS: loops, multiplication

### Problem 93: Extract Non-Border Elements

def non_border_elements(A):
    # UNVECTORIZE THIS
    result = []
    for i in range(1, len(A)-1):
        for j in range(1, len(A[0])-1):
            result.append(A[i][j])
    return result
#TAGS: loops, nested loops

### Problem 94: Matrix Rotation by 90 Degrees

def rotate_90(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][len(A)-1-i] = A[i][j]
    return result
#TAGS: loops, nested loops

### Problem 95: Check if Square Matrix

def is_square(A):
    # UNVECTORIZE THIS
    return len(A) == len(A[0])
#TAGS: comparison

### Problem 96: Check if Matrix is Diagonal

def is_diagonal(A):
    # UNVECTORIZE THIS
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
    return True
#TAGS: loops, nested loops, comparison

### Problem 97: Elementwise Square Root

def elementwise_sqrt(A):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** 0.5
    return result
#TAGS: loops, nested loops, square root

### Problem 98: Elementwise Matrix Division

def elementwise_divide(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / B[i][j]
    return result
#TAGS: loops, nested loops, division

### Problem 99: Matrix Absolute Difference

def matrix_absolute_difference(A, B):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = abs(A[i][j] - B[i][j])
    return result
#TAGS: loops, nested loops, subtraction, absolute

### Problem 100: Elementwise Exponential

def elementwise_exp(A, base):
    # UNVECTORIZE THIS
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = base ** A[i][j]
    return result
#TAGS: loops, nested loops, exponentiation
