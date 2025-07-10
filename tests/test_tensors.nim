# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import unittest
import steadytensor
import math
import random

suite "Tensor Initialization and Basic Operations":
    test "initTensor with default value":
        let t = initTensor[float, [2, 3]]()
        check t.data.len == 6
        for val in t.data:
            check val == 0.0

    test "initTensor with custom default":
        let t = initTensor[float, [2, 3]](42.0)
        check t.data.len == 6
        for val in t.data:
            check val == 42.0

    test "initTensor with input array":
        let data: array[6, float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let t = initTensor[float, [2, 3], 6](data)
        check t.data == data
        check t[0, 0] == 1.0
        check t[1, 2] == 6.0

    test "zeros initialization":
        let t = zeros[float, [2, 3]]()
        check t.data.len == 6
        for val in t.data:
            check val == 0.0

    test "ones initialization":
        let t = ones[float, [2, 3]]()
        check t.data.len == 6
        for val in t.data:
            check val == 1.0

    test "rand initialization":
        # Set seed for reproducibility
        randomize(42)
        let t = rand[float, [2, 3]](-1.0, 1.0)
        check t.data.len == 6
        for val in t.data:
            check val >= -1.0
            check val <= 1.0

suite "Tensor Indexing and Access":
    test "1D tensor indexing":
        let data: array[3, float] = [1.0, 2.0, 3.0]
        let t = initTensor[float, [3], 3](data)
        check t[0] == 1.0
        check t[1] == 2.0
        check t[2] == 3.0

    test "2D tensor indexing":
        let data: array[6, float] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        let t = initTensor[float, [2, 3], 6](data)
        check t[0, 0] == 1.0
        check t[0, 1] == 2.0
        check t[0, 2] == 3.0
        check t[1, 0] == 4.0
        check t[1, 1] == 5.0
        check t[1, 2] == 6.0

    test "3D tensor indexing":
        let t = initTensor[float, [2, 2, 2]](1.0)
        check t[0, 0, 0] == 1.0
        check t[1, 1, 1] == 1.0

suite "Tensor Properties":
    test "shape property":
        let t = initTensor[float, [2, 3]]()
        check t.shape == [2, 3]

    test "size property":
        let t = initTensor[float, [2, 3]]()
        check t.size == 6

    test "string representation":
        let data: array[4, float] = [1.0, 2.0, 3.0, 4.0]
        let t = initTensor[float, [2, 2], 4](data)
        check $t == "Tensor[[2, 2]](1.0, 2.0, 3.0, 4.0)"

suite "Tensor Operations":
    test "map operation":
        let t1 = initTensor[float, [2, 2], 4]([1.0, 2.0, 3.0, 4.0])
        let t2 = t1.map(proc(x: float): float = x * 2)
        check t2.data == [2.0, 4.0, 6.0, 8.0]

    test "zip operation":
        let t1 = initTensor[float, [2, 2], 4]([1.0, 2.0, 3.0, 4.0])
        let t2 = initTensor[float, [2, 2], 4]([1.0, 1.0, 1.0, 1.0])
        let t3 = zip(t1, t2, proc(x, y: float): float = x + y)
        check t3.data == [2.0, 3.0, 4.0, 5.0]

suite "Error Handling":
    test "index out of bounds":
        let t = initTensor[float, [2, 2]]()
        expect(AssertionDefect):
            discard t[2, 0]
        expect(AssertionDefect):
            discard t[0, 2]

    test "invalid number of indices":
        let t = initTensor[float, [2, 2]]()
        expect(AssertionDefect):
            discard t[0]
        expect(AssertionDefect):
            discard t[0, 0, 0]