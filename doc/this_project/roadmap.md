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