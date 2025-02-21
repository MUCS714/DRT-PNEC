
# DRT-PNEC
We propose a new VIO initialization method by extending the [rotation-translation-decoupled framework](https://ieeexplore.ieee.org/document/10205123) with the addition of uncertainty parameters and optimization modules. This code is the implementation of our proposed method, which runs on **Linux**.


## 1. Prerequisites
1.1 **Ubuntu** 
* Ubuntu 18.04

1.2. **Dependency**

* C++14 or C++17 Compiler
* Eigen 3.3.7
* OpenCV 4.0.0
* Boost 1.58.0
* opengv 1.0.0
* Cere-solver 1.14.0 [Ceres Installation](http://ceres-solver.org/installation.html)
* basalt [gitlab](https://gitlab.com/VladyslavUsenko/basalt/-/tree/master/)
  * clone into ```thirdparty/basalt``` with git submodules \
    ```git clone --recursive https://gitlab.com/VladyslavUsenko/basalt.git```
  * follow their instructions to install  

## 2. Build Project with Cmake
For instructions on how to build and run the project, please refer to [drt-vio-initialization](https://github.com/boxuLibrary/drt-vio-init.git).

