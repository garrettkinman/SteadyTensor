# Copyright (c) 2025 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensors

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# HELPER: Output Shape Calculation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func matmulShape*(shape1, shape2: static TensorShape[2]): static TensorShape[2] {.compileTime.} =
    const 
        M = shape1[0]
        K1 = shape1[1]
        K2 = shape2[0]
        N = shape2[1]
    
    static:
        assert K1 == K2, "Inner dimensions must match for matrix multiplication"
    
    result = [M, N]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX MULTIPLICATION (The Core Kernel)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func matmul*[T; shapeA, shapeB: static TensorShape[2]](
    A: Tensor[T, shapeA], 
    B: Tensor[T, shapeB]
): Tensor[T, matmulShape(shapeA, shapeB)] =
    ## High-performance Matrix Multiplication Kernel.
    ## Automatically adapts loop order to memory layout (RowMajor vs ColMajor).
    
    const
        M = shapeA[0]
        K = shapeA[1]
        N = shapeB[1]
    
    # Initialize result with zeros
    result = zeros[T, matmulShape(shapeA, shapeB)]()

    when ColMajor:
        # COLUMN-MAJOR OPTIMIZATION (JIK Loop)
        # Optimal for: SDRs, Spiking NNs, Fortran-style layout
        # Access Pattern: Streaming down columns of C and A.
        
        for j in 0..<N:            # Iterate Columns of Result/B
            for k in 0..<K:        # Iterate Rows of B / Columns of A
                let bVal = B[k, j] # Scalar load (invariant in inner loop)
                
                for i in 0..<M:    # Iterate Rows of Result/A (Contiguous)
                    # C[i, j] += A[i, k] * B[k, j]
                    # TODO: add operator/macro to enable writing `result[i, j] += A[i, k] * bVal`
                    result[i, j] = result[i, j] + (A[i, k] * bVal)

    else:
        # ROW-MAJOR OPTIMIZATION (IKJ Loop)
        # Optimal for: Dense DL, C-style layout
        # Access Pattern: Streaming across rows of C and B.
        
        for i in 0..<M:            # Iterate Rows of Result/A
            for k in 0..<K:        # Iterate Columns of A / Rows of B
                let aVal = A[i, k] # Scalar load (invariant in inner loop)
                
                for j in 0..<N:    # Iterate Columns of Result/B (Contiguous)
                    # C[i, j] += A[i, k] * B[k, j]
                    # TODO: add operator/macro to enable writing `result[i, j] += aVal * B[k, j]`
                    result[i, j] = result[i, j] + (aVal * B[k, j])

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MATRIX-VECTOR MULTIPLICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func mvMul*[T; shapeW, shapeX: static TensorShape[2]](
    W: Tensor[T, shapeW], 
    x: Tensor[T, shapeX]
): Tensor[T, matmulShape(shapeW, shapeX)] =
    ## Optimized Matrix-Vector Multiplication (y = Wx)
    
    const
        M = shapeW[0]
        K = shapeW[1]

    static:
        assert shapeW.len == 2, "Weight matrix must be 2D"
        assert shapeX.len == 2, "Input must be 2D (vector as matrix)"
        assert shapeX[0] == K, "Inner dimensions must match"
        assert shapeX[1] == 1, "Input must be a column vector [K, 1]"

    result = zeros[T, [M, 1]]()

    when ColMajor:
        # ColMajor MV Mul: Linear combination of columns
        # Ideally suited for Sparse Inputs (SDRs)
        for j in 0..<K:         # Columns of W
            let xVal = x[j, 0]
            if xVal != T(0):    # Cheap check for sparsity!
                for i in 0..<M: # Rows of W (Contiguous)
                    result[i, 0] += W[i, j] * xVal
    else:
        # RowMajor MV Mul: Dot product of rows
        for i in 0..<M:         # Rows of W
            var sum = T(0)
            for j in 0..<K:     # Columns of W (Contiguous)
                sum += W[i, j] * x[j, 0]
            result[i, 0] = sum

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ELEMENT-WISE OPS (Linear Kernels)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These functions iterate 0..totalSize, ignoring shape entirely. 
# This generates the most efficient assembly for element-wise ops.

func add*[T; shape: static TensorShape](a, b: Tensor[T, shape], res: var Tensor[T, shape]) =
    for i in 0..<totalSize(shape):
        res.data[i] = a.data[i] + b.data[i]

func sub*[T; shape: static TensorShape](a, b: Tensor[T, shape], res: var Tensor[T, shape]) =
    for i in 0..<totalSize(shape):
        res.data[i] = a.data[i] - b.data[i]

func mul*[T; shape: static TensorShape](a, b: Tensor[T, shape], res: var Tensor[T, shape]) =
    # Element-wise multiplication (Hadamard product)
    for i in 0..<totalSize(shape):
        res.data[i] = a.data[i] * b.data[i]

func scale*[T; shape: static TensorShape](t: Tensor[T, shape], s: T, res: var Tensor[T, shape]) =
    for i in 0..<totalSize(shape):
        res.data[i] = t.data[i] * s