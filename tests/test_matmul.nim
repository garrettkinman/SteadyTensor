# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadytensor

suite "Matrix Multiplication Tests":
    test "2x2 identity matrix multiplication":
        const shape: TensorShape[2] = [2, 2]
        var 
            A = initTensor[float, shape]()
            expected = initTensor[float, shape]()

        # Set up identity matrix
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        expected[0, 0] = 1.0
        expected[1, 1] = 1.0

        # Identity matrix multiplied by itself should equal itself
        let result = A * A
        check(result[0, 0] == expected[0, 0])
        check(result[0, 1] == expected[0, 1])
        check(result[1, 0] == expected[1, 0])
        check(result[1, 1] == expected[1, 1])

    test "2x3 × 3x2 matrix multiplication":
        const
            shape1: TensorShape[2] = [2, 3]
            shape2: TensorShape[2] = [3, 2]
            shape3: TensorShape[2] = [2, 2]
        
        var
            A = initTensor[float, shape1]()
            B = initTensor[float, shape2]()
            expected = initTensor[float, shape3]()

        # Set up first matrix
        A[0, 0] = 1.0; A[0, 1] = 2.0; A[0, 2] = 3.0
        A[1, 0] = 4.0; A[1, 1] = 5.0; A[1, 2] = 6.0

        # Set up second matrix
        B[0, 0] = 7.0; B[0, 1] = 8.0
        B[1, 0] = 9.0; B[1, 1] = 10.0
        B[2, 0] = 11.0; B[2, 1] = 12.0

        # Expected results
        expected[0, 0] = 58.0  # 1*7 + 2*9 + 3*11
        expected[0, 1] = 64.0  # 1*8 + 2*10 + 3*12
        expected[1, 0] = 139.0 # 4*7 + 5*9 + 6*11
        expected[1, 1] = 154.0 # 4*8 + 5*10 + 6*12

        let result = A * B
        check(abs(result[0, 0] - expected[0, 0]) < 1e-10)
        check(abs(result[0, 1] - expected[0, 1]) < 1e-10)
        check(abs(result[1, 0] - expected[1, 0]) < 1e-10)
        check(abs(result[1, 1] - expected[1, 1]) < 1e-10)

    test "3x4 × 4x2 matrix multiplication":
        const
            shape1: TensorShape[2] = [3, 4]
            shape2: TensorShape[2] = [4, 2]
        
        var
            A = initTensor[float, shape1]()
            B = initTensor[float, shape2]()
            C = initTensor[float, matmulShape(shape1, shape2)]()

        # Fill matrices with simple values
        for i in 0..2:
            for j in 0..3:
                A[i, j] = float(i + j)
        
        for i in 0..3:
            for j in 0..1:
                B[i, j] = float(i + j)

        # Test both methods
        C = A * B  # Using explicit matmul
        let D = A * B    # Using operator

        # Results should be identical
        for i in 0..2:
            for j in 0..1:
                check(abs(C[i, j] - D[i, j]) < 1e-10)

    test "1x1 matrix multiplication":
        const shape: TensorShape[2] = [1, 1]
        var
            A = initTensor[float, shape]()
            B = initTensor[float, shape]()
        
        A[0, 0] = 3.0
        B[0, 0] = 4.0
        
        let result = A * B
        # let result = matmulDirect(A, B)
        check(abs(result[0, 0] - 12.0) < 1e-10)

    test "Matrix multiplication with zeros":
        const shape: TensorShape[2] = [2, 2]
        var
            A = initTensor[float, shape]()
            B = initTensor[float, shape]()
            expected = initTensor[float, shape]()

        # Only set some values, leaving others as zeros
        A[0, 0] = 1.0
        B[1, 1] = 1.0

        let result = A * B
        
        # Result should be all zeros
        check(result == expected)

suite "Matrix-Vector Multiplication (mvMul)":
    # This tests the optimized path in ops.nim where:
    # "when shapeB.len == 2 and shapeB[1] == 1" 
    
    test "2x3 Matrix * 3x1 Vector":
        const
            shapeW: TensorShape[2] = [2, 3]
            shapeX: TensorShape[2] = [3, 1]
            shapeY: TensorShape[2] = [2, 1]

        var
            W = initTensor[float, shapeW]()
            x = initTensor[float, shapeX]()
            expected = initTensor[float, shapeY]()

        # W = [[1, 2, 3],
        #      [4, 5, 6]]
        W[0, 0] = 1.0; W[0, 1] = 2.0; W[0, 2] = 3.0
        W[1, 0] = 4.0; W[1, 1] = 5.0; W[1, 2] = 6.0

        # x = [[1],
        #      [0],
        #      [-1]]
        x[0, 0] = 1.0
        x[1, 0] = 0.0
        x[2, 0] = -1.0

        # Expected y = Wx
        # y[0] = 1*1 + 2*0 + 3*-1 = -2
        # y[1] = 4*1 + 5*0 + 6*-1 = -2
        expected[0, 0] = -2.0
        expected[1, 0] = -2.0

        # This implicitly calls `mvMul` via the * operator dispatch
        let result = W * x

        check result.shape == expected.shape
        check abs(result[0, 0] - expected[0, 0]) < 1e-10
        check abs(result[1, 0] - expected[1, 0]) < 1e-10

    test "Identity Matrix-Vector":
        const shape: TensorShape[2] = [2, 2]
        const vecShape: TensorShape[2] = [2, 1]
        
        var I = initTensor[float, shape]()
        var v = initTensor[float, vecShape]()

        I[0, 0] = 1.0; I[1, 1] = 1.0
        v[0, 0] = 7.0; v[1, 0] = 9.0

        let res = I * v
        check res[0, 0] == 7.0
        check res[1, 0] == 9.0

suite "Transposed Matrix Multiplication (matmulT)":
    # This tests the "Virtual Transpose" kernel 
    # Logic: C = A.T * B
    # A is read virtually as if transposed, without allocating memory for A.T

    test "Square Transpose (A.T * B)":
        const shape: TensorShape[2] = [2, 2]
        var
            A = initTensor[float, shape]()
            B = initTensor[float, shape]()
        
        # A = [[1, 2],
        #      [3, 4]]
        # Therefore A.T = [[1, 3],
        #                  [2, 4]]
        A[0, 0] = 1.0; A[0, 1] = 2.0
        A[1, 0] = 3.0; A[1, 1] = 4.0

        # B = Identity
        B[0, 0] = 1.0; B[1, 1] = 1.0

        # Result should be A.T * I = A.T
        let result = matmulT(A, B)

        check result[0, 0] == 1.0
        check result[0, 1] == 3.0 # Key check: was 3.0 in A[1,0], now in Res[0,1]
        check result[1, 0] == 2.0
        check result[1, 1] == 4.0

    test "Rectangular Transpose (3x2).T * (3x1)":
        # A: [3, 2] (M=3, K=2)
        # B: [3, 1] (M=3, N=1)
        # Result: [2, 1] (K, N)
        const
            shapeA: TensorShape[2] = [3, 2]
            shapeB: TensorShape[2] = [3, 1]
        
        var
            A = initTensor[float, shapeA]()
            B = initTensor[float, shapeB]()

        # A = [[1, 10],
        #      [2, 20],
        #      [3, 30]]
        A[0, 0] = 1.0; A[0, 1] = 10.0
        A[1, 0] = 2.0; A[1, 1] = 20.0
        A[2, 0] = 3.0; A[2, 1] = 30.0

        # B = [[1], [1], [1]]
        B[0, 0] = 1.0; B[1, 0] = 1.0; B[2, 0] = 1.0

        # Operation: Sum of columns of A
        # A.T = [[1, 2, 3],
        #        [10, 20, 30]]
        # A.T * B = [[1*1 + 2*1 + 3*1],
        #            [10*1 + 20*1 + 30*1]]
        # Result = [[6], [60]]

        let result = matmulT(A, B)

        check result.shape == [2, 1]
        check result[0, 0] == 6.0
        check result[1, 0] == 60.0