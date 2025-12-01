# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CONFIGURATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Global Layout Switch
# Pass -d:colMajor to the compiler to switch to Column-Major (Fortran/MATLAB) layout.
# Default is Row-Major (C/Numpy) layout.
# Use ColMajor for Sparse Distributed Representations (SDR) or specific MCU prefetch patterns.
const ColMajor*: bool = defined(colMajor)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CORE TYPES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

type
    TensorShape*[rank: static int] = array[rank, int]

# Calculate total size at compile time
func totalSize*[rank: static int](shape: static TensorShape[rank]): static int =
    var size = 1
    for dim in shape:
        size *= dim
    size

type
    Tensor*[T; shape: static TensorShape] = object
        # The backing store is a contiguous static array.
        # We access this differently depending on ColMajor/RowMajor.
        data*: array[totalSize(shape), T]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# INDEXING LOGIC
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func flatIndex[shape: static TensorShape](indices: varargs[int]): int {.inline.} =
    var idx = 0
    var stride = 1
    when ColMajor:
        # COLUMN-MAJOR: First dimension changes fastest (Left-to-Right)
        # Ideal for: Sparse Distributed Representations (SDRs), Spiking Neural Networks (SNNs)
        for dim in 0 .. shape.high:
            assert indices[dim] >= 0 and indices[dim] < shape[dim], "Index out of bounds"
            idx += indices[dim] * stride
            stride *= shape[dim]
    else:
        # ROW-MAJOR: Last dimension changes fastest (Right-to-Left)
        # Ideal for: Standard Dense Deep Learning, C Interop
        for dim in countdown(shape.high, 0):
            assert indices[dim] >= 0 and indices[dim] < shape[dim], "Index out of bounds"
            idx += indices[dim] * stride
            stride *= shape[dim]
    idx

func `[]`*[T; shape: static TensorShape](t: Tensor[T, shape], indices: varargs[int]): T =
    assert indices.len == shape.len, "Invalid number of indices"
    result = t.data[flatIndex[shape](indices)]

func `[]=`*[T; shape: static TensorShape](t: var Tensor[T, shape], indices: varargs[int], value: T) {.inline.} =
    assert indices.len == shape.len, "Invalid number of indices"
    t.data[flatIndex[shape](indices)] = value

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPERS & INITIALIZATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Helper to get dimensions at compile time
func dim*[rank: static int](shape: static TensorShape[rank], d: static int): static int =
    const val = shape[d]
    val

func shape*[T; shape: static TensorShape](t: Tensor[T, shape]): TensorShape[shape.len] =
    shape

func size*[T; shape: static TensorShape](t: Tensor[T, shape]): int =
    totalSize(shape)

func initTensor*[T; shape: static TensorShape](defaultVal: T = default(T)): Tensor[T, shape] =
    # Result is implicitly initialized to zero/default by Nim, 
    # but we can force set it if a specific default is provided.
    result = Tensor[T, shape]() 
    if defaultVal != default(T):
        for i in 0..<result.data.len:
            result.data[i] = defaultVal

func initTensor*[T; shape: static TensorShape; N: static int](data: array[N, T]): Tensor[T, shape] =
    static:
        # Verify at compile time that the array size matches tensor shape
        assert N == totalSize(shape), "Input array size must match tensor shape"
    
    result = Tensor[T, shape](data: data)

# Initialization functions
func zeros*[T; shape: static TensorShape](): Tensor[T, shape] =
    result = initTensor[T, shape](0.T)

func ones*[T; shape: static TensorShape](): Tensor[T, shape] =
    result = initTensor[T, shape](1.T)

proc rand*[T; shape: static TensorShape](min: T = 0.T, max: T = 1.T): Tensor[T, shape] =
    # This has side effects (on global RNG state), hence `proc` instead of `func`
    result = initTensor[T, shape]()
    for i in 0..<result.data.len:
        result.data[i] = rand(min..max)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DIRECT ACCESS ITERATORS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These allow kernels to iterate over raw data without worrying about shape.

iterator items*[T; shape: static TensorShape](t: Tensor[T, shape]): T =
    for i in 0..<totalSize(shape):
        yield t.data[i]

iterator mitems*[T; shape: static TensorShape](t: var Tensor[T, shape]): var T =
    for i in 0..<totalSize(shape):
        yield t.data[i]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UTILITIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func `$`*[T; shape: static TensorShape](t: Tensor[T, shape]): string =
    result = "Tensor["
    result.add $shape
    result.add "]("
    for i, val in t.data:
        if i > 0: result.add ", "
        result.add $val
    result.add ")"

func copy*[T; shape: static TensorShape](t: Tensor[T, shape]): Tensor[T, shape] =
    ## Creates a deep copy of a tensor
    result = initTensor[T, shape]()
    for i in 0..<t.data.len:
        result.data[i] = t.data[i]

func map*[T, U; shape: static TensorShape](t: Tensor[T, shape], f: static proc(x: T): U {.noSideEffect.}): Tensor[U, shape] =
    result = initTensor[U, shape]()
    for i in 0..<t.data.len:
        result.data[i] = f(t.data[i])

func zip*[T, U, V; shape: static TensorShape](
    t1: Tensor[T, shape],
    t2: Tensor[U, shape],
    f: static proc(x: T, y: U): V {.noSideEffect.}
): Tensor[V, shape] =
    result = initTensor[V, shape]()
    for i in 0..<t1.data.len:
        result.data[i] = f(t1.data[i], t2.data[i])
