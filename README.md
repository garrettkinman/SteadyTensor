<!--
 Copyright (c) 2024 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

A lightweight, dependency-free tensor library that uses zero dynamic memory allocation. Intended for microcontrollers.

# Concept
This library is part of the broader [SteadyNN](https://github.com/garrettkinman/SteadyNN) framework for TinyML, which is built around a few key principles:
1. **No dynamic memory allocation.** This is to improve performance and to provide compile-time guarantees of memory usage. Currently no other TinyML frameworks (that I know of) use only static memory allocation.
2. **A small set of primitive operators.** All higher-level operations (e.g., matrix multiplication) can be broken down to some combination of these primitive operations. So long as these primitive operators (primops) are optimized for hardware, you will have reasonably performant higher-level operations.
3. **Portable to new hardware.** Porting to new hardware (including dedicated accelerators) is as easy as implementing the primops on the hardware. If that isn't efficient enough, you can implement a new tensor subtype to optimize to your heart's content.
4. **No dependencies.** Working with other TinyML frameworks can be a pain, as there are so many dependencies that can (and often do) give you problems. By avoiding dependencies, this framework is much easier to use, simpler to understand and debug, and less of a pain to set up and use.

# Hardware Acceleration
If you want to accelerate this library on custom hardware (e.g., vector/SIMD instructions), just fork this repository and rewrite the primitive operators (in `src/steadytensor/primops.nim`) with your own optimized implementations.

**TODO:** add instructions for installing forked library and using with SteadyNN

# Primitive Operators
1. Arithmetic Operations
   1. Element-wise Tensor Addition
   2. Element-wise Scalar Addition
   3. Element-wise Tensor Subtraction
   4. Element-wise Scalar Subtraction
   5. Element-wise Vector Multiplication
   6. Element-wise Scalar Multiplicaiton
   7. Dot Product
2. Activation Functions
   1. Element-wise Identity
   2. Element-wise ReLu
   3. Element-wise Sigmoid
   4. Element-wise Tanh
   5. Element-wise Leaky ReLu (?)
   6. Element-wise Swish (?)
   7. Element-wise GELU (?)
   8. Element-wise ELU (?)
3. Reduction Operations
   1. Sum Reduction
   2. Mean Reduction
   3. Max Reduction
   4. Min Reduction
   5. Argmax Reduction
   6. Argmin Reduction
4. Comparison Operations
   1. Element-wise Equal
   2. Element-wise Not Equal
   3. Element-wise Greater
   4. Element-wise Greater Equal
   5. Element-wise Less
   6. Element-wise Less Equal
5. Miscellaneous Functions
   1. Element-wise Negate
   2. Element-wise Reciprocal
   3. Element-wise Abs
   4. Element-wise Sqrt
   5. Element-wise Sq
   6. Element-wise Ln
   7. Element-wise Exp
   8. Element-wise Sin
   9. Element-wise Clamp