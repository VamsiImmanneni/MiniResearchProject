import numpy as np

def function_00(a, b):
    return np.dot(a, b)

def function_01(A, v):
    return np.dot(A, v)

def function_02(A):
    return np.transpose(A)

def function_03(A, B):
    return np.add(A, B)

def function_04(A, s):
    return np.multiply(A, s)

def function_05(A):
    return np.sum(A, axis=1)

def function_06(A):
    return np.sum(A, axis=0)

def function_07(A):
    return np.diagonal(A)

def function_08(A, B):
    return np.dot(A, B)

def function_09(A, B):
    return np.multiply(A, B)

def function_10(A):
    return np.mean(A, axis=1)

def function_11(A):
    return np.trace(A)

def function_12(A, B):
    return np.divide(A, B)

def function_13(A, B):
    return np.subtract(A, B)

def function_14(A):
    return np.max(A, axis=1)

def function_15(A):
    return np.max(A, axis=0)

def function_16(A, s):
    return np.divide(A, s)

def function_17(A):
    return len(A) == len(A[0])

def function_18(A, p):
    return np.power(A, p)

def function_19(A):
    return np.min(A, axis=1)

def function_20(A):
    return np.min(A, axis=0)

def function_21(A):
    return -A

def function_22(A):
    return np.count_nonzero(A > 0)

def function_23(A):
    return np.count_nonzero(A < 0)

def function_24(rows, cols):
    return np.zeros((rows, cols))

def function_25(n):
    return np.eye(n)

def function_26(A):
    return np.ravel(A)

def function_27(A):
    return np.array_equal(A, A.T)

def function_28(A):
    return np.abs(A)

def function_29(A, d):
    np.fill_diagonal(A, d)
    return A

def function_30(A):
    return np.triu(A)

def function_31(A):
    return np.tril(A)

def function_32(A):
    return np.count_nonzero(A == 0)

def function_33(A):
    return np.all(A[np.triu_indices(len(A), k=1)] == 0)

def function_34(A):
    return np.all(A[np.tril_indices(len(A), k=-1)] == 0)

def function_35(A):
    return np.all(np.diag(A) == 0)

def function_36(A):
    return np.rot90(A)

def function_37(A):
    return np.sqrt(A)

def function_38(A):
    A[A < 0] = 0
    return A

def function_39(A):
    return np.linalg.det(A)

def function_40(A):
    return np.copy(A)

def function_41(A, scale_factor):
    return A * scale_factor

def function_42(A):
    return np.transpose(A)

def function_43(A, k, i, j):
    A[i] += k * A[j]
    return A

def function_44(A):
    return np.count_nonzero(A)

def function_45(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    inv_det = 1 / det
    result = np.array([[A[1, 1] * inv_det, -A[0, 1] * inv_det],
                       [-A[1, 0] * inv_det, A[0, 0] * inv_det]])
    return result

def function_46(A):
    det = np.linalg.det(A)
    return det != 0

def function_47(A, c):
    return A + c

def function_48(A):
    return np.log(A)

def function_49(A):
    return np.sum(A, axis=1)

def function_50(A):
    return np.sum(A, axis=0)

def function_51(A):
    return np.all(A == 0)

def function_52(A):
    return np.prod(A, axis=1)

def function_53(A):
    return np.linalg.norm(A)

def function_54(A, value):
    return np.count_nonzero(A > value)

def function_55(A, mod_value):
    return np.mod(A, mod_value)

def function_56(A, value, replacement):
    A[A < value] = replacement
    return A

def function_57(A):
    return np.all(A == A[0, 0])

def function_58(A):
    return np.linalg.matrix_rank(A)

def function_59(A):
    return np.allclose(np.eye(len(A)), A)

def function_60(A):
    return np.diagonal(A[::-1])

def function_61(A):
    return np.flip(A)

def function_62(A):
    return np.sort(A)

def function_63(A):
    return np.sort(A, axis=0)

def function_64(A):
    return np.cumsum(np.cumsum(A, axis=0), axis=1)

def function_65(A):
    return np.flip(np.flip(A, axis=0), axis=1)

def function_66(A):
    return np.flipud(A)

def function_67(A):
    return list(np.unique(A))

def function_68(A):
    return np.allclose(np.eye(len(A)), A)

def function_69(A, value, replacement):
    A[A > value] = replacement
    return A

def function_70(A):
    return np.all(np.diag(A) == A[0, 0])

def function_71(A):
    return np.prod(np.diag(A))

def function_72(A):
    return np.all(np.diag(A) == A[0, 0])

def function_73(A):
    upper_triangular = np.triu(A, k=1)
    return upper_triangular[np.nonzero(upper_triangular)]

def function_74(A):
    lower_triangular = np.tril(A, k=-1)
    return lower_triangular[np.nonzero(lower_triangular)]

def function_75(A):
    product_matrix = np.dot(A, A)
    return np.all(product_matrix == A * A)

def function_76(A):
    return np.trace(np.flipud(A))

def function_77(A, k):
    num_cols = len(A[0])
    k %= num_cols
    return np.roll(A, k, axis=1)

def function_78(A, k):
    num_cols = len(A[0])
    k %= num_cols
    return np.roll(A, -k, axis=1)

def function_79(A, scalar):
    return A + scalar

def function_80(A):
    return np.all(A == np.transpose(A))

def function_81(A):
    return np.mean(np.diag(A))

def function_82(A):
    total = np.sum(A[0, :]) + np.sum(A[-1, :]) + np.sum(A[1:-1, 0]) + np.sum(A[1:-1, -1])
    return total

def function_83(A):
    return np.transpose(A)

def function_84(A, scalar):
    return A * scalar

def function_85(A):
    upper_triangular = np.triu(A, k=1)
    return np.all(upper_triangular == 0)

def function_86(A, power):
    return np.power(A, power)

def function_87(A):
    return np.max(A, axis=1)

def function_88(A):
    return np.min(A, axis=0)

def function_89(A):
    if len(A) != len(A[0]):
        return False

    total = np.sum(A[0])
    row_sums = np.sum(A, axis=1)
    col_sums = np.sum(A, axis=0)

    if not np.all(row_sums == total) or not np.all(col_sums == total):
        return False

    diag_sum1 = np.trace(A)
    diag_sum2 = np.trace(np.flipud(A))

    return diag_sum1 == total and diag_sum2 == total

def function_90(A):
    return np.mean(A, axis=0)

def function_91(A):
    return np.prod(np.diag(np.flipud(A)))

def function_92(A):
    return A[1:-1, 1:-1].flatten()

def function_93(A):
    return np.rot90(A)

def function_94(A):
    return len(A) == len(A[0])

def function_95(A):
    return np.all(np.diag(A) == 0)

def function_96(A):
    return np.sqrt(A)

def function_97(A, B):
    return np.divide(A, B, where=(B != 0))

def function_98(A, B):
    return np.abs(A - B)

def function_99(A, base):
    return np.power(base, A)
