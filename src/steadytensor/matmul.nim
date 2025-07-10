# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors, primops

# Create a row view into a tensor
func rowView*[T; shape: static TensorShape](
    A: Tensor[T, shape],
    row: int
): StridedVector[T, dim(shape, 1)] =
    static:
        assert shape.len == 2, "rowView requires a 2D tensor"
    
    StridedVector[T, dim(shape, 1)](
        data: cast[ptr UncheckedArray[T]](addr A.data[row * dim(shape, 1)]),
        stride: 1
    )

# Create a column view into a tensor
func colView*[T; shape: static TensorShape](
    A: Tensor[T, shape],
    col: int
): StridedVector[T, dim(shape, 0)] =
    static:
        assert shape.len == 2, "colView requires a 2D tensor"
    
    StridedVector[T, dim(shape, 0)](
        data: cast[ptr UncheckedArray[T]](addr A.data[col]),
        stride: dim(shape, 1)
    )

# Helper function to compute matmul output shape at compile time
func matmulShape*(shape1, shape2: static TensorShape[2]): static TensorShape[2] {.compileTime.} =
    const 
        M = shape1[0]
        K1 = shape1[1]
        K2 = shape2[0]
        N = shape2[1]
    
    static:
        assert K1 == K2, "Inner dimensions must match"
    
    result = [M, N]

func `*`*[T; shape1, shape2: static TensorShape[2]](
    A: Tensor[T, shape1], 
    B: Tensor[T, shape2]
): Tensor[T, matmulShape(shape1, shape2)] =
    ## Matrix multiplication operator
    ## Returns A × B as a new tensor
    const
        M = shape1[0]
        K = shape1[1]
        N = shape2[1]
    
    result = initTensor[T, matmulShape(shape1, shape2)]()
    var 
        rowA: StridedVector[T, K]
        colB: StridedVector[T, K]

    # Iterate through each element of the result matrix
    for i in 0..<M:
        # Get row view of A
        rowA = A.rowView(i)
        
        for j in 0..<N:
            # Get column view of B
            colB = B.colView(j)
            
            # Compute dot product of row i of A with column j of B
            result[i, j] = dot(rowA, colB)