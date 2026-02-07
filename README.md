<!--
 Copyright (c) 2024 Garrett Kinman
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

An ultra-light, ultra-flexible tensor library written in pure Nim. Intended for microcontrollers.

# Concept
This library is part of the broader [SteadyPCN](https://github.com/garrettkinman/SteadyPCN) framework for predictive coding, which is built around a few key principles:
1. **No dynamic memory allocation.** This is to improve performance and to provide compile-time guarantees of memory usage. For many safety-critical applications, this is also considered a requirement / best practice.
2. **Configurable memory format.** Tensors can be configured to be stored in either row-major or column-major order at compile time (just pass `-d:colMajor` to the compiler for column-major, else it defaults to row-major), allowing optimization depending on the hardware available to you.
3. **Configurable data types.** Tensors can easily be configured to use any underlying data type, even custom ones. This aids in rapid prototyping of various low-bit architectures.
4. **Portable to new hardware.** Porting to new hardware (including dedicated accelerators) is as easy as implementing the kernels on the hardware.
5. **No dependencies.** Working with other TinyML frameworks can be a pain, as there are so many dependencies that can (and often do) give you problems. By avoiding dependencies, this framework is much easier to use, simpler to understand and debug, and less of a pain to set up and use.

# Hardware Acceleration
If you want to accelerate this library on custom hardware (e.g., vector/SIMD instructions), just fork this repository and rewrite the kernels (in `src/steadytensor/kernels.nim`) with your own optimized implementations.

**TODO:** add instructions for installing forked library and using with SteadyPCN