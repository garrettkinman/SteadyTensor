# Copyright (c) 2024 Garrett Kinman
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

# TODO: implement variants of primops that apply for an entire tensor, calling primops for each dimView

import tensors, primops

func softmax*[T; n: static int](x: StridedVector[T, n], result: var StridedVector[T, n]) =
    ## Element-wise softmax function that computes the softmax of every element of a given `StridedVector`
    ## Uses max value subtraction trick for numerical stability
    
    # Create temporary vectors for exp values
    var expValues: array[n, T]
    var expVector = StridedVector[T, n](
        data: cast[ptr UncheckedArray[T]](addr expValues[0]),
        stride: 1
    )
    
    let maxVal = max(x)
    var shifted = x  # Creates a copy
    subtractScalar(shifted, maxVal, shifted)  # x - max
    exp(shifted, expVector)
    let sumExp = sum(expVector)
    
    let scale = 1.T / sumExp
    scale(expVector, scale, result)