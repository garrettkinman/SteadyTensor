# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import std / math

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# strided vector + helper functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# A view into a contiguous array with a stride
type
    StridedVector*[T; n: static int] = object
        data*: ptr UncheckedArray[T]  # Raw pointer to the first element
        stride*: int                   # Distance between consecutive elements

func `[]`*[T; n: static int](v: StridedVector[T, n], index: int): T {.inline.} =
    result = v.data[index * v.stride]

func `[]=`*[T; n: static int](v: var StridedVector[T, n], index: int, value: T) {.inline.} =
    v.data[index * v.stride] = value

# ~~~~~~~~~~~~~~~~~~~~~
# arithmetic operations
# ~~~~~~~~~~~~~~~~~~~~~

func add*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] + b[i]

func addScalar*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] + b

func subtract*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] - b[i]

func subtractScalar*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] - b

func negate*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = -a[i]

func hadamard*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] * b[i]

func scale*[T; n: static int](a: StridedVector[T, n], b: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i] * b

func dot*[T; n: static int](a, b: StridedVector[T, n]): T =
    result = T(0)
    for i in 0..<n:
        result += a[i] * b[i]

# ~~~~~~~~~~~~~~~~~~~~
# reduction operations
# ~~~~~~~~~~~~~~~~~~~~

func sum*[T; n: static int](a: StridedVector[T, n]): T =
    for i in 0..<n:
        result += a[i]

func mean*[T; n: static int](a: StridedVector[T, n]): T =
    result = a.sum() * (1.T / n.T)

func max*[T; n: static int](a: StridedVector[T, n]): T =
    result = a[0]
    for i in 1..<n:
        let val = a[i]
        if val > result:
            result = val

func min*[T; n: static int](a: StridedVector[T, n]): T =
    result = a[0]
    for i in 1..<n:
        let val = a[i]
        if val < result:
            result = val

func argmax*[T; n: static int](a: StridedVector[T, n]): int =
    result = 0
    var maxVal = a[0]
    for i in 1..<n:
        let val = a[i]
        if val > maxVal:
            maxVal = val
            result = i

func argmin*[T; n: static int](a: StridedVector[T, n]): int =
    result = 0
    var minVal = a[0]
    for i in 1..<n:
        let val = a[i]
        if val < minVal:
            minVal = val
            result = i

# ~~~~~~~~~~~~~~~~~~~~~
# comparison operations
# ~~~~~~~~~~~~~~~~~~~~~

# TODO: make these return tensors of 1s and 0s (converted to type T) instead of bool?

func equal*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] == b[i]

func notEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] != b[i]

func greater*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] > b[i]

func greaterEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] >= b[i]

func less*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] < b[i]

func lessEqual*[T; n: static int](a, b: StridedVector[T, n], result: var StridedVector[bool, n]) =
    for i in 0..<n:
        result[i] = a[i] <= b[i]


# ~~~~~~~~~~~~~~~~~~~~~~~
# miscellaneous functions
# ~~~~~~~~~~~~~~~~~~~~~~~

func recip*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = 1.T / a[i]

func abs*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = abs(a[i])

func sqrt*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = sqrt(a[i])

func square*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let val = a[i * a.stride]
        result[i] = val * val

func ln*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = ln(a[i])

func exp*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = exp(a[i])

func sin*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = sin(a[i])

func clamp*[T; n: static int](a: StridedVector[T, n], min_val, max_val: T, result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = clamp(a[i], min_val .. max_val) # TODO: avoid slice?

# ~~~~~~~~~~~~~~~~~~~~
# activation functions
# ~~~~~~~~~~~~~~~~~~~~

func identity*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        result[i] = a[i]

func relu*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let val = a[i]
        result[i] = if val > T(0): val else: T(0)

func sigmoid*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let x = a[i]
        result[i] = 1.T / (1.T + exp(-x))

func tanh*[T; n: static int](a: StridedVector[T, n], result: var StridedVector[T, n]) =
    for i in 0..<n:
        let 
            x = a[i]
            expX = exp(x)
            expNegX = exp(-x)
        result[i] = (expX - expNegX) / (expX + expNegX)

# TODO:
# Element-wise Leaky ReLu (?)
# Element-wise Swish (?)
# Element-wise GELU (?)
# Element-wise ELU (?)