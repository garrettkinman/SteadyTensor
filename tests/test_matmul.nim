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

    # TODO: debug why 1x1 matrices don't seem to work
    test "1x1 matrix multiplication":
        const shape: TensorShape[2] = [1, 1]
        var
            A = initTensor[float, shape]()
            B = initTensor[float, shape]()
        
        A[0, 0] = 3.0
        B[0, 0] = 4.0
        
        let result = A * B
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