
def function_00(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


def function_01(A, v):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j] * v[j] * 2
    return result


def function_02(A):
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result


def function_03(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + B[i][j]
    return result


def function_04(A, s):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * s
    return result


def function_05(A):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
    return result


def function_06(A):
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
    return result


def function_07(A):
    result = []
    for i in range(len(A)):
        result.append(A[i][i])
    return result


def function_08(A, B):
    result = [[0] * len(B[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result


def function_09(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * B[i][j]
    return result


def function_10(A):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
        result[i] /= len(A[0])
    return result

def function_11(A):
    result = 0
    for i in range(len(A)):
        result += A[i][i]
    return result


def function_12(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / B[i][j]
    return result


def function_13(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] - B[i][j]
    return result


def function_14(A):
    result = [float('-inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > result[i]:
                result[i] = A[i][j]
    return result


def function_15(A):
    result = [float('-inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] > result[j]:
                result[j] = A[i][j]
    return result


def function_16(A, s):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / s
    return result


def function_17(A):
    return len(A) == len(A[0])


def function_18(A, p):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** p
    return result


def function_19(A):
    result = [float('inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < result[i]:
                result[i] = A[i][j]
    return result


def function_20(A):
    result = [float('inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] < result[j]:
                result[j] = A[i][j]
    return result

def function_21(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = -A[i][j]
    return result


def function_22(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > 0:
                count += 1
    return count


def function_23(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0:
                count += 1
    return count


def function_24(rows, cols):
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append(0)
        result.append(row)
    return result


def function_25(n):
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


def function_26(A):
    result = []
    for i in range(len(A)):
        for j in range(len(A[0])):
            result.append(A[i][j])
    return result


def function_27(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != A[j][i]:
                return False
    return True


def function_28(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = abs(A[i][j])
    return result


def function_29(A, d):
    for i in range(len(A)):
        A[i][i] = d[i]
    return A


def function_30(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i, len(A[0])):
            result[i][j] = A[i][j]
    return result


def function_31(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(i+1):
            result[i][j] = A[i][j]
    return result


def function_32(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] == 0:
                count += 1
    return count


def function_33(A):
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            if A[i][j] != 0:
                return False
    return True


def function_34(A):
    for i in range(1, len(A)):
        for j in range(i):
            if A[i][j] != 0:
                return False
    return True


def function_35(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
    return True


def function_36(A):
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][len(A)-1-i] = A[i][j]
    return result


def function_37(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** 0.5
    return result


def function_38(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < 0:
                A[i][j] = 0
    return A


def function_39(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]


def function_40(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j]
    return result


def function_41(A, scale_factor):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * scale_factor
    return result


def function_42(A):
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[j][i]
    return result


def function_43(A, k, i, j):
    for col in range(len(A[0])):
        A[i][col] += k * A[j][col]
    return A


def function_44(A):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != 0:
                count += 1
    return count


def function_45(A):
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    inv_det = 1 / det
    result = [[A[1][1] * inv_det, -A[0][1] * inv_det],
              [-A[1][0] * inv_det, A[0][0] * inv_det]]
    return result


def function_46(A):
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return det != 0


def function_47(A, c):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + c
    return result


def function_48(A):
    import math
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = math.log(A[i][j])
    return result


def function_49(A):
    result = [0] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] += A[i][j]
    return result


def function_50(A):
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
    return result


def function_51(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != 0:
                return False
    return True


def function_52(A):
    result = [1] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i] *= A[i][j]
    return result


def function_53(A):
    result = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            result += A[i][j]**2
    return result**0.5


def function_54(A, value):
    count = 0
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > value:
                count += 1
    return count


def function_55(A, mod_value):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] % mod_value
    return result


def function_56(A, value, replacement):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] < value:
                A[i][j] = replacement
    return A


def function_57(A):
    value = A[0][0]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] != value:
                return False
    return True


def function_58(A):
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


def function_59(A):
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


def function_60(A):
    result = []
    for i in range(len(A)):
        result.append(A[i][len(A)-1-i])
    return result


def function_61(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[len(A)-1-i][len(A[0])-1-j] = A[i][j]
    return result


def function_62(A):
    for i in range(len(A)):
        A[i] = sorted(A[i])
    return A


def function_63(A):
    for j in range(len(A[0])):
        col = [A[i][j] for i in range(len(A))]
        col = sorted(col)
        for i in range(len(A)):
            A[i][j] = col[i]
    return A


def function_64(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + (result[i-1][j] if i > 0 else 0) + (result[i][j-1] if j > 0 else 0) - (result[i-1][j-1] if i > 0 and j > 0 else 0)
    return result


def function_65(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[len(A)-1-i][len(A[0])-1-j]
    return result


def function_66(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[len(A)-1-i][j]
    return result


def function_67(A):
    unique_set = set()
    for i in range(len(A)):
        for j in range(len(A[0])):
            unique_set.add(A[i][j])
    return list(unique_set)


def function_68(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i == j and A[i][j] != 1:
                return False
            elif i != j and A[i][j] != 0:
                return False
    return True


def function_69(A, value, replacement):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > value:
                A[i][j] = replacement
    return A


def function_70(A):
    for i in range(1, len(A)):
        for j in range(1, len(A[0])):
            if A[i][j] != A[i-1][j-1]:
                return False
    return True


def function_71(A):
    product = 1
    for i in range(len(A)):
        product *= A[i][i]
    return product


def function_72(A):
    value = A[0][0]
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
            elif i == j and A[i][j] != value:
                return False
    return True


def function_73(A):
    result = []
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            result.append(A[i][j])
    return result


def function_74(A):
    result = []
    for i in range(1, len(A)):
        for j in range(i):
            result.append(A[i][j])
    return result


def function_75(A):
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


def function_76(A):
    result = 0
    for i in range(len(A)):
        result += A[i][len(A[0])-1-i]
    return result


def function_77(A, k):
    num_cols = len(A[0])
    k %= num_cols
    result = [[0] * num_cols for _ in range(len(A))]
    for j in range(num_cols):
        for i in range(len(A)):
            result[i][j] = A[i][(j + k) % num_cols]
    return result


def function_78(A, k):
    num_cols = len(A[0])
    k %= num_cols
    result = [[0] * num_cols for _ in range(len(A))]
    for j in range(num_cols):
        for i in range(len(A)):
            result[i][j] = A[i][(j - k + num_cols) % num_cols]
    return result


def function_79(A, scalar):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] + scalar
    return result


def function_80(A):
    for i in range(len(A)):
        for j in range(i):
            if A[i][j] != A[j][i]:
                return False
    return True


def function_81(A):
    total = 0
    for i in range(len(A)):
        total += A[i][i]
    return total / len(A)


def function_82(A):
    total = 0
    for i in range(len(A)):
        total += A[i][0] + A[i][-1]
    for j in range(1, len(A[0]) - 1):
        total += A[0][j] + A[-1][j]
    return total


def function_83(A):
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][i] = A[i][j]
    return result


def function_84(A, scalar):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] * scalar
    return result


def function_85(A):
    for i in range(len(A)):
        for j in range(i+1, len(A[0])):
            if A[i][j] != 0:
                return False
    return True


def function_86(A, power):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** power
    return result


def function_87(A):
    result = [float('-inf')] * len(A)
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j] > result[i]:
                result[i] = A[i][j]
    return result


def function_88(A):
    result = [float('inf')] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            if A[i][j] < result[j]:
                result[j] = A[i][j]
    return result


def function_89(A):
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

def function_90(A):
    result = [0] * len(A[0])
    for j in range(len(A[0])):
        for i in range(len(A)):
            result[j] += A[i][j]
        result[j] /= len(A)
    return result

def function_91(A):
    product = 1
    for i in range(len(A)):
        product *= A[i][len(A[0])-1-i]
    return product


def function_92(A):
    result = []
    for i in range(1, len(A)-1):
        for j in range(1, len(A[0])-1):
            result.append(A[i][j])
    return result


def function_93(A):
    result = [[0] * len(A) for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[j][len(A)-1-i] = A[i][j]
    return result


def function_94(A):
    return len(A) == len(A[0])


def function_95(A):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j and A[i][j] != 0:
                return False
    return True


def function_96(A):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] ** 0.5
    return result


def function_97(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = A[i][j] / B[i][j]
    return result


def function_98(A, B):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = abs(A[i][j] - B[i][j])
    return result


def function_99(A, base):
    result = [[0] * len(A[0]) for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            result[i][j] = base ** A[i][j]
    return result
