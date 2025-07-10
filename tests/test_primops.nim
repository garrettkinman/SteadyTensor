# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# test_primops.nim
import unittest
import steadytensor
import math

suite "Primitive Operations Tests":
    const n = 4  # Size for test vectors
    let epsilon = 1e-6  # For float comparisons

    setup:
        # Setup arrays and vectors for each test
        var 
            data1 = [1.0, 2.0, 3.0, 4.0]
            data2 = [0.5, 1.5, 2.5, 3.5]
            data3 = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0, 99.0]
            result_data: array[n, float]
            bool_result_data: array[n, bool]
            strided_result_data: array[n, float]
        
        var 
            vec1 = StridedVector[float, n](data: cast[ptr UncheckedArray[float]](addr data1[0]), stride: 1)
            vec2 = StridedVector[float, n](data: cast[ptr UncheckedArray[float]](addr data2[0]), stride: 1)
            vec3 = StridedVector[float, n](data: cast[ptr UncheckedArray[float]](addr data3[0]), stride: 2)
            result_vec = StridedVector[float, n](data: cast[ptr UncheckedArray[float]](addr result_data[0]), stride: 1)
            bool_result_vec = StridedVector[bool, n](data: cast[ptr UncheckedArray[bool]](addr bool_result_data[0]), stride: 1)
            strided_result_vec = StridedVector[float, n](data: cast[ptr UncheckedArray[float]](addr strided_result_data[0]), stride: 1)

    test "Arithmetic Operations":
        # Test addition
        add(vec1, vec2, result_vec)
        check abs(result_vec[0] - 1.5) < epsilon
        check abs(result_vec[1] - 3.5) < epsilon
        check abs(result_vec[2] - 5.5) < epsilon
        check abs(result_vec[3] - 7.5) < epsilon

        # Test scalar addition
        addScalar(vec1, 1.0, result_vec)
        check abs(result_vec[0] - 2.0) < epsilon
        check abs(result_vec[1] - 3.0) < epsilon
        check abs(result_vec[2] - 4.0) < epsilon
        check abs(result_vec[3] - 5.0) < epsilon

        # Test subtraction
        subtract(vec1, vec2, result_vec)
        check abs(result_vec[0] - 0.5) < epsilon
        check abs(result_vec[1] - 0.5) < epsilon
        check abs(result_vec[2] - 0.5) < epsilon
        check abs(result_vec[3] - 0.5) < epsilon

        # Test scalar subtraction
        subtractScalar(vec1, 1.0, result_vec)
        check abs(result_vec[0] - 0.0) < epsilon
        check abs(result_vec[1] - 1.0) < epsilon
        check abs(result_vec[2] - 2.0) < epsilon
        check abs(result_vec[3] - 3.0) < epsilon

        # Test negation
        negate(vec1, result_vec)
        check abs(result_vec[0] - (-1.0)) < epsilon
        check abs(result_vec[1] - (-2.0)) < epsilon
        check abs(result_vec[2] - (-3.0)) < epsilon
        check abs(result_vec[3] - (-4.0)) < epsilon

        # Test Hadamard (element-wise) multiplication
        hadamard(vec1, vec2, result_vec)
        check abs(result_vec[0] - 0.5) < epsilon
        check abs(result_vec[1] - 3.0) < epsilon
        check abs(result_vec[2] - 7.5) < epsilon
        check abs(result_vec[3] - 14.0) < epsilon

        # Test scalar multiplication
        scale(vec1, 2.0, result_vec)
        check abs(result_vec[0] - 2.0) < epsilon
        check abs(result_vec[1] - 4.0) < epsilon
        check abs(result_vec[2] - 6.0) < epsilon
        check abs(result_vec[3] - 8.0) < epsilon

        # Test dot product
        let dp = dot(vec1, vec2)
        check abs(dp - 25.0) < epsilon

    test "Reduction Operations":
        # Test sum
        check abs(sum(vec1) - 10.0) < epsilon

        # Test mean
        check abs(mean(vec1) - 2.5) < epsilon

        # Test max
        check abs(max(vec1) - 4.0) < epsilon

        # Test min
        check abs(min(vec1) - 1.0) < epsilon

        # Test argmax
        check argmax(vec1) == 3

        # Test argmin
        check argmin(vec1) == 0

    test "Comparison Operations":
        # Test equal
        equal(vec1, vec1, bool_result_vec)  # Compare with itself
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

        # Test not equal
        notEqual(vec1, vec2, bool_result_vec)
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

        # Test greater
        greater(vec1, vec2, bool_result_vec)
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

        # Test greater equal
        greaterEqual(vec1, vec2, bool_result_vec)
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

        # Test less
        less(vec2, vec1, bool_result_vec)
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

        # Test less equal
        lessEqual(vec2, vec1, bool_result_vec)
        check bool_result_vec[0] == true
        check bool_result_vec[1] == true
        check bool_result_vec[2] == true
        check bool_result_vec[3] == true

    test "Miscellaneous Functions":
        # Test reciprocal
        recip(vec1, result_vec)
        check abs(result_vec[0] - 1.0) < epsilon
        check abs(result_vec[1] - 0.5) < epsilon
        check abs(result_vec[2] - (1.0/3.0)) < epsilon
        check abs(result_vec[3] - 0.25) < epsilon

        # Test absolute value
        data1[0] = -1.0  # Modify some values to be negative
        data1[2] = -3.0
        abs(vec1, result_vec)
        check abs(result_vec[0] - 1.0) < epsilon
        check abs(result_vec[1] - 2.0) < epsilon
        check abs(result_vec[2] - 3.0) < epsilon
        check abs(result_vec[3] - 4.0) < epsilon

        # Test square
        data1 = [1.0, 2.0, 3.0, 4.0]  # Reset values
        square(vec1, result_vec)
        check abs(result_vec[0] - 1.0) < epsilon
        check abs(result_vec[1] - 4.0) < epsilon
        check abs(result_vec[2] - 9.0) < epsilon
        check abs(result_vec[3] - 16.0) < epsilon

        # Test square root
        sqrt(vec1, result_vec)
        check abs(result_vec[0] - 1.0) < epsilon
        check abs(result_vec[1] - sqrt(2.0)) < epsilon
        check abs(result_vec[2] - sqrt(3.0)) < epsilon
        check abs(result_vec[3] - 2.0) < epsilon

        # Test natural log
        ln(vec1, result_vec)
        check abs(result_vec[0]) < epsilon
        check abs(result_vec[1] - ln(2.0)) < epsilon
        check abs(result_vec[2] - ln(3.0)) < epsilon
        check abs(result_vec[3] - ln(4.0)) < epsilon

        # Test exponential
        exp(vec1, result_vec)
        check abs(result_vec[0] - exp(1.0)) < epsilon
        check abs(result_vec[1] - exp(2.0)) < epsilon
        check abs(result_vec[2] - exp(3.0)) < epsilon
        check abs(result_vec[3] - exp(4.0)) < epsilon

        # Test sine
        sin(vec1, result_vec)
        check abs(result_vec[0] - sin(1.0)) < epsilon
        check abs(result_vec[1] - sin(2.0)) < epsilon
        check abs(result_vec[2] - sin(3.0)) < epsilon
        check abs(result_vec[3] - sin(4.0)) < epsilon

        # Test clamp
        clamp(vec1, 1.5, 3.5, result_vec)
        check abs(result_vec[0] - 1.5) < epsilon
        check abs(result_vec[1] - 2.0) < epsilon
        check abs(result_vec[2] - 3.0) < epsilon
        check abs(result_vec[3] - 3.5) < epsilon

    test "Activation Functions":
        # Test identity
        identity(vec1, result_vec)
        check abs(result_vec[0] - 1.0) < epsilon
        check abs(result_vec[1] - 2.0) < epsilon
        check abs(result_vec[2] - 3.0) < epsilon
        check abs(result_vec[3] - 4.0) < epsilon

        # Test ReLU
        data1 = [-1.0, 0.0, 1.0, 2.0]  # Reset with some negative values
        relu(vec1, result_vec)
        check abs(result_vec[0] - 0.0) < epsilon
        check abs(result_vec[1] - 0.0) < epsilon
        check abs(result_vec[2] - 1.0) < epsilon
        check abs(result_vec[3] - 2.0) < epsilon

        # Test sigmoid
        sigmoid(vec1, result_vec)
        check abs(result_vec[0] - (1.0 / (1.0 + exp(1.0)))) < epsilon
        check abs(result_vec[1] - 0.5) < epsilon
        check abs(result_vec[2] - (1.0 / (1.0 + exp(-1.0)))) < epsilon
        check abs(result_vec[3] - (1.0 / (1.0 + exp(-2.0)))) < epsilon

        # Test tanh
        tanh(vec1, result_vec)
        check abs(result_vec[0] - tanh(-1.0)) < epsilon
        check abs(result_vec[1] - 0.0) < epsilon
        check abs(result_vec[2] - tanh(1.0)) < epsilon
        check abs(result_vec[3] - tanh(2.0)) < epsilon

    test "Strided Access":
        # Test basic access
        check abs(vec3[0] - 1.0) < epsilon
        check abs(vec3[1] - 2.0) < epsilon
        check abs(vec3[2] - 3.0) < epsilon
        check abs(vec3[3] - 4.0) < epsilon

        # Test a simple operation (scaling by 2)
        scale(vec3, 2.0, strided_result_vec)
        check abs(strided_result_vec[0] - 2.0) < epsilon
        check abs(strided_result_vec[1] - 4.0) < epsilon
        check abs(strided_result_vec[2] - 6.0) < epsilon
        check abs(strided_result_vec[3] - 8.0) < epsilon
