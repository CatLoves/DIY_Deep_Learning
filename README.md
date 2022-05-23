# DIY_Deep_Learning

<img src="./data/Figs/logo5.jpg" alt="drawing"/>


DIY(Do it youself) Deep Learning is a python package for our own deep learning framework using pure Numpy, with the following highlights:
- **Education focus**: Main purpose of this package is obtaining deep understanding of modern deep learning frameworks like Tensorflow and PyTorch, so we will implement most of deep learning functions on our own, like convolution, batch normalization, dropout, etc
- **High speed**: GPU acceleration is used to speed up our code, [GPU Python](https://developer.nvidia.com/how-to-cuda-python) and [CuPy](https://github.com/cupy/cupy) make our package comparablely fast as  PyTorch
- **High reliability**: All important functions come with a unit test, corner cases are considered to ensure high reliability
- **High availability**: Out of box example code is provided for all important functions for easy usage
- **High scalability**: Everyone can contribute his/her awesome code to this repository, scaling up the ability of this package. Further questions and communication can also be sent to email: 940417022@qq.com
- **High productivity**: Life is short, I use python! Get away from time-consuming and tedious C/C++ code for high productivity.
- **Complete documentation**: Professional document tool [pdoc](https://pdoc.dev/docs/pdoc.html) is massively used to help users master every important function




# Why should we DIY deep learning ?

There are two important reasons:

1. **PyTorch/Tensorflow blocks you from mastering deep learning**

> “What I Cannot Create, I Do Not Understand”  
--Richard Feynman, Nobel prize winning Physicist

If you merely use PyTorch/Tensorflow for deep learning, it's unlikely that you will master deep learning quite well. For reasons, let's consider two cases:
(1) You only use PyTorch/Tensorflow API and don't like reading API's source code, in this case you may just know something about API's math or principle, but you don't know how it actually works;
(2) You use PyTorch/Tensorflow API and often read API's source code, in this case, you will find that the source code is really hard to read and understand, because most source code is written in C/C++, for example:
- Source code of convolution in pytorch is [here](https://github.com/pytorch/pytorch/blob/c780610f2d8358297cb4e4460692d496e124d64d/aten/src/ATen/native/Convolution.cpp#L481):
```c
at::Tensor _convolution(
    const Tensor& input_r, const Tensor& weight_r, const Tensor& bias_r,
    IntArrayRef stride_, IntArrayRef padding_, IntArrayRef dilation_,
    bool transposed_, IntArrayRef output_padding_, int64_t groups_,
    bool benchmark, bool deterministic, bool cudnn_enabled) {

  const bool input_is_mkldnn = input_r.is_mkldnn();
  auto input = input_r;
  auto weight = weight_r;
  auto bias = bias_r;
  auto k = weight.ndimension();
  // mkldnn conv2d weights could have been re-ordered to 5d by
  // mkldnn_reorder_conv2d_weight
  if (input_is_mkldnn && (k == input.ndimension() + 1)) {
    k = input.ndimension();
  }
  int64_t dim = k - 2;

  TORCH_CHECK(dim > 0, "weight should have at least three dimensions");

  ConvParams params;
  params.stride = expand_param_if_needed(stride_, "stride", dim);
  params.padding = expand_param_if_needed(padding_, "padding", dim);
  params.dilation = expand_param_if_needed(dilation_, "dilation", dim);
  params.transposed = transposed_;
  params.output_padding = expand_param_if_needed(output_padding_, "output_padding", dim);
  params.groups = groups_;
  params.benchmark = benchmark;
  params.deterministic = deterministic;
  params.cudnn_enabled = cudnn_enabled;
  ...
```
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


# Development plan

- [ ] 2022-5-23 to 2022-5-29: output technology roadmap,  
- [x] 2022-5-22: initial readme.md including: Why should we DIY deep learning and Features and principles of this package



# Tutorials and references

1. Deep learning courses by Andrew Ng: [here](https://github.com/ashishpatel26/Andrew-NG-Notes)

2. Machine learning using numpy: [here](https://github.com/ddbourgin/numpy-ml/tree/master/numpy_ml/neural_nets)