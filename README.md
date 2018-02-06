# The Frameworks for Hierarchical-Pooled Deep-Convolutional Features

Here we provide the code for the frameworks for Hierarchical-Pooled Deep-Convolutional Features
 (HPDCF), from the following paper:

    Fengqian Pang, Heng Li, Yonggang Shi, and Zhiwen Liu. Computional Analysis of Cell Dynamics in 
    Videos with Hierarchical-Pooled Deep-Convolutional Features. Submitted to Journal of Computional Biology

This code is tested based on Windows7, Caffe Toolbox and MATLAB2014a. 

## OpenCV 2.4.11
We provide Matlab mex file for Dense Trajectory and Dense Flow, but they still need to install [OpenCV Lib](https://opencv.org/releases.html)

## Pretrained Deep Convolutional Networks
We only provide Caffe Toolbox based on CPU. If GPU mode is needed, please set featextr.use_gpu 1 in HPDCF_setup.m and recompile Caffe Toolbox with GPU [Caffe Toolkit](https://github.com/BVLC/caffe/tree/windows).

## Demo
A matlab demo code is provided:

*Step 1: Some external libraries<br>
You need download some [external libraries](https://drive.google.com/open?id=1r1ROZ261vAUW5987He5LOP0eblZL5WRc).<br>
*Step 2: OpenCV Lib and Caffe Toolbox<br>
You need download OpenCV Lib and install in your system. You could use the provided Caffe Toolbox based on CPU, or recompile it in GPU mode.<br>
*Step 3: Parameters Defination<br>
You could set some important parameters in HPDCF_setup.m, while other parameters could also be changed in the corrsponding Class Defination. Simply, you can just use the default parameters.<br>
*Step 4: Input Samples<br>
You can test different samples by changing vid_path in demo.m<br>
*Step 5: HPFCF<br>
Now you can run the matlab file "script_demo.m" to extract TDD features.<br>
