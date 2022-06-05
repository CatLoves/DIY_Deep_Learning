# DIY_Deep_Learning

<img src="./data/Figs/logo5.jpg" alt="drawing"/>


DIY(Do it youself) Deep Learning is a python package for our own deep learning framework using pure Numpy, highlights are:
- **Education focus**: Main purpose of this package is obtaining deep understanding of modern deep learning frameworks like Tensorflow and PyTorch, so we will implement most of deep learning functions on our own, like convolution, batch normalization, dropout, etc
- **High speed**: GPU acceleration is used to speed up our code, [GPU Python](https://developer.nvidia.com/how-to-cuda-python) and [CuPy](https://github.com/cupy/cupy) make our package comparablely fast as  PyTorch
- **High reliability**: All important functions come with a unit test, corner cases are considered to ensure high reliability
- **High availability**: Out of box example code is provided for all important functions for easy usage
- **High scalability**: Everyone can contribute his/her awesome code to this repository, scaling up the ability of this package. Further questions and communication can also be sent to email: 940417022@qq.com
- **High productivity**: Life is short, I use python! Get away from time-consuming and tedious C/C++ code for high productivity.
- **Complete documentation**: Professional document tool [pdoc](https://pdoc.dev/docs/pdoc.html) is massively used to help users master every important function

Limitations are:
- **Non-distributed**: Until now, multi-GPU traning is not supported.
  
Table of contents:
- [DIY_Deep_Learning](#diy_deep_learning)
- [Why should we DIY deep learning ?](#why-should-we-diy-deep-learning-)
- [How to get started ?](#how-to-get-started-)
  - [Prerequisites](#prerequisites)
  - [Overview on technology roadmap](#overview-on-technology-roadmap)
- [Development plan](#development-plan)
- [Tutorials and references](#tutorials-and-references)

# Why should we DIY deep learning ?

There are two important reasons:

1. **PyTorch/Tensorflow blocks you from mastering deep learning**

> “What I Cannot Create, I Do Not Understand”  
--Richard Feynman, Nobel prize winning Physicist

If you merely use PyTorch/Tensorflow for deep learning, it's unlikely that you will master deep learning quite well. For reasons, let's consider two cases:
(1) You only use PyTorch/Tensorflow API and don't like reading API's source code, in this case you may just know something about API's math or principle, but you don't know how it actually works;
(2) You use PyTorch/Tensorflow API and often read API's source code, in this case, you will find that the source code is really hard to read and understand, because most source code is written in C/C++, for example:

- Source code of Batch normalization in pytorch is [here](https://github.com/pytorch/pytorch/blob/420b37f3c67950ed93cd8aa7a12e673fcfc5567b/aten/src/ATen/native/Normalization.cpp#L61-L126):
```c
template<typename scalar_t>
void batch_norm_cpu_inference_contiguous(Tensor& output, const Tensor& input,
    const Tensor& weight /* optional */, const Tensor& bias /* optional */,
    const Tensor& mean, const Tensor& variance, double eps) {
  int64_t n_batch = input.size(0);
  int64_t n_channel = input.size(1);
  int64_t image_size = input.numel() / n_batch / n_channel;

  scalar_t* output_data = output.data_ptr<scalar_t>();
  const scalar_t* input_data = input.data_ptr<scalar_t>();
  const scalar_t* weight_data = weight.defined() ? weight.data_ptr<scalar_t>() : nullptr;
  const scalar_t* bias_data = bias.defined() ? bias.data_ptr<scalar_t>() : nullptr;
  const scalar_t* mean_data = mean.data_ptr<scalar_t>();
  const scalar_t* var_data = variance.data_ptr<scalar_t>();

  Tensor alpha = at::empty_like(mean);
  Tensor beta = at::empty_like(mean);
  scalar_t* alpha_data = alpha.data_ptr<scalar_t>();
  scalar_t* beta_data = beta.data_ptr<scalar_t>();
  for (int64_t c = 0; c < n_channel; c++) {
    scalar_t inv_var = 1 / std::sqrt(var_data[c] + static_cast<scalar_t>(eps));
    scalar_t weight_v = weight_data ? weight_data[c] : 1;
    scalar_t bias_v = bias_data ? bias_data[c] : 0;
    alpha_data[c] = inv_var * weight_v;
    beta_data[c] = bias_v - mean_data[c] * inv_var * weight_v;
  }
  ...
```

it is **really frustrating** for people not familiar with C++, mkldnn, cudnn, torch c lib etc. The time cost to understand these c++ code is **prohibitive** for most of us.

2. **It's not easy to customize Pytorch/Tensorflow for your specific purpose**

Suppose you have to deploy your deep learning models on AIot devices, only a subset and customized version of Pytorch/Tensorflow is needed, it would be time-consuming and risky to achieve your goal. 


# How to get started ?
    
## Prerequisites

1. **Passion for deep learning**: The tour to build a customized and reliable deep learning package is time-consuming and challenging, we will not get success without the passion for deep learning.

2. **Basic knowledge about deep learning**: You need to know the basics of deep learning, like convolution, batch normalization etc. Andrew Ng's deep learning courses are highly recommended: [here](https://github.com/ashishpatel26/Andrew-NG-Notes)


## Overview on technology roadmap

Building a customized and reliable deep learning package is a huge workload, which is comprehensive, complex and challenging. Therefore, survey was performed on all popular open-source deep learning frameworks, important metrics are:
- Focus: main focus of the framework, like speed or clean code;
- Popularity: we use Github star number for the sake of simplicity, on 2022-5-25;
- Backprop: how the backpropagation of networks was implemented

|    Framework    | Focus | Popularity | Backprop
| ---------- | --- | --- | ---
| [Tensorflow](https://www.tensorflow.org/) |  Industrial usage | 165 K | [Jax](https://github.com/google/jax)
| [PyTorch](https://pytorch.org/)       |  Research and industrial usage | 56.2 K | [C++](https://github.com/pytorch/pytorch/tree/9d3ffed32715896e5c8f358d2bc7cbf233093a27/torch/csrc/autograd)
| [MXNet](https://github.com/apache/incubator-mxnet)       |  Portable, efficient and scalable | 20 K | C++
| [Deeplearning4j](https://github.com/eclipse/deeplearning4j)       |  JVM support | 12.5 K | C++, JAVA
| [ONNX](https://github.com/onnx/onnx)       | Unified format of ML model | 12.5 K | -
| [Chainer(not maintained)](https://github.com/chainer/chainer)       | Flexibility | 5.7 K | [Python and C++](https://github.com/chainer/chainer/blob/f53e57434089fa6f8dfe93a1306ba394cbabf8ad/chainer/_backprop.py)
| [Trax](https://github.com/google/trax)       | Clear code and speed |  5.7 K | [Jax](https://github.com/google/jax)  
| [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch) | Pure Numpy, education focus | 21.1 K | [Pure numpy](https://github.com/eriklindernoren/ML-From-Scratch/blob/a2806c6732eee8d27762edd6d864e0c179d8e9e8/mlfromscratch/deep_learning/layers.py#L29)  
| [NumpyDL](https://github.com/chaoming0625/NumpyDL) | Pure Numpy, education focus | 203 | [Pure numpy](https://github.com/chaoming0625/NumpyDL/blob/master/npdl/layers/convolution.py#L88)
| [deepnet](https://github.com/parasdahal/deepnet) | Pure Numpy, education focus | 308 | [Pure numpy](https://github.com/parasdahal/deepnet/blob/master/deepnet/layers.py#L41)
| [MyDeepLearning](https://github.com/nick6918/MyDeepLearning) | Pure Numpy, education focus | 291 | [Pure numpy](https://github.com/nick6918/MyDeepLearning/blob/master/lib/layers/layers.py#L347)
| [xshinnosuke](https://github.com/E1eveNn/xshinnosuke) | Pure Numpy, Dynamic Graph, GPU usage | 266 | [Pure numpy](https://github.com/E1eveNn/xshinnosuke/blob/master/xs/core/autograd.py)
| [TensorLayer](https://github.com/tensorlayer/TensorLayer) | Pure Python | 7K | backends like Tensorflow

Based on the survey above, we plan to develop our package based on five packages:
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [NumpyDL](https://github.com/chaoming0625/NumpyDL)
- [deepnet](https://github.com/parasdahal/deepnet)
- [MyDeepLearning](https://github.com/nick6918/MyDeepLearning)
- [xshinnosuke](https://github.com/E1eveNn/xshinnosuke) 

Detailed technology roadmap is [here](https://github.com/CatLoves/DIY_Deep_Learning/blob/main/doc/this_project/roadmap.md)

# Development plan

- [ ] 2022-6-2 to 2022-6-8: output the fisrt version V0.0.1 which support network of liner layers, cupy acceleration,  pdoc documentation 
- [x] 2022-5-26 to 2022-6-1: get familar with existing reference frameworks 
- [x] 2022-5-23 to 2022-5-25: conduct a survey on existing open-source deep learning frameworks 
- [x] 2022-5-22: initial readme.md including: Why should we DIY deep learning and Features and principles of this package



# Tutorials and references

1. Deep learning courses by Andrew Ng: [here](https://github.com/ashishpatel26/Andrew-NG-Notes)

2. Machine learning using numpy: [here](https://github.com/ddbourgin/numpy-ml/tree/master/numpy_ml/neural_nets)