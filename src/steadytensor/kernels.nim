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

func matmulTShape*(shape1, shape2: static TensorShape[2]): static TensorShape[2] {.compileTime.} =
    # Infers shape for C = A.T * B
    # A: [M, K] -> A.T: [K, M]
    # B: [M, N]
    # Result: [K, N]
    const 
        M1 = shape1[0]
        K = shape1[1]
        M2 = shape2[0]
        N = shape2[1]
    
    static:
        assert M1 == M2, "First dimension (rows) must match for Transposed Matrix Multiplication (A.T * B)"
    
    result = [K, N]

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
                    # TODO: add operator/macro to enable writing `result[i, 0] += W[i, j] * xVal`
                    result[i, 0] = result[i, 0] + (W[i, j] * xVal)
    else:
        # RowMajor MV Mul: Dot product of rows
        for i in 0..<M:         # Rows of W
            var sum = T(0)
            for j in 0..<K:     # Columns of W (Contiguous)
                sum += W[i, j] * x[j, 0]
            result[i, 0] = sum

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# TRANSPOSED MATRIX MULTIPLICATION (Virtual Transpose)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

func matmulT*[T; shapeA, shapeB: static TensorShape[2]](
    A: Tensor[T, shapeA], 
    B: Tensor[T, shapeB]
): Tensor[T, matmulTShape(shapeA, shapeB)] =
    ## High-performance Transposed Matrix Multiplication (C = A.T * B)
    ## Crucial for Backpropagation and Predictive Coding (Error projection).
    ## Does NOT allocate memory for a transpose; reads A virtually.
    
    const
        M = shapeA[0] # Common Dimension (Rows of A and B)
        K = shapeA[1] # Rows of Result (Cols of A)
        N = shapeB[1] # Cols of Result (Cols of B)
    
    # Initialize result with zeros
    result = zeros[T, matmulTShape(shapeA, shapeB)]()

    when ColMajor:
        # ColMajor Optimization
        # In ColMajor, the first dimension (Rows) changes fastest.
        # Since we sum over M (Rows of A and B), we scan both inputs linearly!
        # This is the ideal memory layout for A.T * B.
        
        for j in 0..<N:            # Iterate Cols of Result/B
            for i in 0..<K:        # Iterate Rows of Result / Cols of A
                var sum = T(0)
                # Inner loop scans down the columns of A and B (Contiguous)
                for k in 0..<M:    
                    sum += A[k, i] * B[k, j]
                result[i, j] = sum
    else:
        # RowMajor Optimization (Outer Product / Rank-1 Update approach)
        # Standard dot-product (scanning down columns) is strided/slow in RowMajor.
        # Instead, we iterate the common dimension 'k' (Rows of A and B) first.
        # We read Row k of A and Row k of B (both contiguous) and add their product to C.
        
        for k in 0..<M:            # Iterate Common Dim (Rows of A and B)
            for i in 0..<K:        # Iterate Rows of Result / Cols of A
                let aVal = A[k, i] # Load scalar from A (Invariant for inner loop)
                
                for j in 0..<N:    # Iterate Cols of Result / Cols of B
                    # C[i, j] += A[k, i] * B[k, j]
                    # We update the accumulator state in C
                    result[i, j] = result[i, j] + (aVal * B[k, j])

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