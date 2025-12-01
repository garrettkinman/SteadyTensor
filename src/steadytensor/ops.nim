# Copyright (c) 2025 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors, kernels

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SCALAR MATH (Broadcasting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T, S](t: Tensor[T, S], val: T): Tensor[T, S] {.inline.} =
    ## Returns t + scalar
    for i in 0..<t.size:
        result.data[i] = t.data[i] + val

func `-`*[T, S](t: Tensor[T, S], val: T): Tensor[T, S] {.inline.} =
    ## Returns t - scalar
    for i in 0..<t.size:
        result.data[i] = t.data[i] - val

func `*`*[T, S](t: Tensor[T, S], val: T): Tensor[T, S] {.inline.} =
    ## Returns t * scalar (Scaling)
    scale(t, val, result)

func `/`*[T, S](t: Tensor[T, S], val: T): Tensor[T, S] {.inline.} =
    ## Returns t / scalar
    scale(t, 1/val, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE ARITHMETIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `+`*[T, S](a, b: Tensor[T, S]): Tensor[T, S] {.inline.} =
    add(a, b, result)

func `-`*[T, S](a, b: Tensor[T, S]): Tensor[T, S] {.inline.} =
    sub(a, b, result)

func `.*`*[T, S](a, b: Tensor[T, S]): Tensor[T, S] {.inline.} =
    ## Element-wise multiplication (Hadamard product)
    ## Note: We use `.*` to distinguish from Matrix Multiplication `*`
    mul(a, b, result)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# IN-PLACE OPERATORS (Crucial for MCUs)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These modify the left-hand side operand. 
# They are zero-allocation operations.

func `+=`*[T, S](a: var Tensor[T, S], b: Tensor[T, S]) {.inline.} =
    add(a, b, a)

func `-=`*[T, S](a: var Tensor[T, S], b: Tensor[T, S]) {.inline.} =
    sub(a, b, a)

func `.*=`*[T, S](a: var Tensor[T, S], b: Tensor[T, S]) {.inline.} =
    mul(a, b, a)

func `+=`*[T, S](a: var Tensor[T, S], val: T) {.inline.} =
    for i in 0..<a.size: a.data[i] += val

func `*=`*[T, S](a: var Tensor[T, S], val: T) {.inline.} =
    scale(a, val, a)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LINEAR ALGEBRA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `*`*[T; shapeA, shapeB: static TensorShape[2]](a: Tensor[T, shapeA], b: Tensor[T, shapeB]): Tensor[T, matmulShape(shapeA, shapeB)] {.inline.} =
    ## Smart Matrix Multiplication Operator.
    ## Detects at compile-time if B is a vector and uses the optimized kernel.
    
    # Check if the second tensor is a column vector (N=1)
    when shapeB.len == 2 and shapeB[1] == 1:
        result = mvMul(a, b)
    else:
        result = matmul(a, b)