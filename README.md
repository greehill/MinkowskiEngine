[pypi-image]: https://badge.fury.io/py/MinkowskiEngine.svg
[pypi-url]: https://pypi.org/project/MinkowskiEngine/
[pypi-download]: https://img.shields.io/pypi/dm/MinkowskiEngine
[slack-badge]: https://img.shields.io/badge/slack-join%20chats-brightgreen
[slack-url]: https://join.slack.com/t/minkowskiengine/shared_invite/zt-piq2x02a-31dOPocLt6bRqOGY3U_9Sw

# Minkowski Engine

[![PyPI Version][pypi-image]][pypi-url] [![pypi monthly download][pypi-download]][pypi-url] [![slack chat][slack-badge]][slack-url]

The Minkowski Engine is an auto-differentiation library for sparse tensors. It supports all standard neural network layers such as convolution, pooling, unpooling, and broadcasting operations for sparse tensors. For more information, please visit [the documentation page](http://nvidia.github.io/MinkowskiEngine/overview.html).

## News

- 2026-03-08 Compatibility refresh: Python `3.10` to `3.14`, PyTorch `2.5` to `2.10`, `uv`-first builds, and modern PEP 517 packaging
- 2026-03-08 CPU validation now targets Linux and macOS, with representative Linux CUDA CI on official PyTorch wheel channels
- 2026-03-08 Point-cloud fixtures are now tracked locally instead of being downloaded during test import

## Example Networks

The Minkowski Engine supports various functions that can be built on a sparse tensor. We list a few popular network architectures and applications here. To run the examples, please install the package and run the command in the package root directory.

| Examples              | Networks and Commands                                                                                                                                                           |
|:---------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| Semantic Segmentation | <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/segmentation.png" width="256"> <br /> `python -m examples.indoor` |
| Classification        | ![](https://nvidia.github.io/MinkowskiEngine/_images/classification_3d_net.png) <br /> `python -m examples.classification_modelnet40`                                                          |
| Reconstruction        | <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_net.png"> <br /> <img src="https://nvidia.github.io/MinkowskiEngine/_images/generative_3d_results.gif" width="256"> <br /> `python -m examples.reconstruction` |
| Completion            | <img src="https://nvidia.github.io/MinkowskiEngine/_images/completion_3d_net.png"> <br /> `python -m examples.completion`                                                       |
| Detection             | <img src="https://nvidia.github.io/MinkowskiEngine/_images/detection_3d_net.png">                                                                                               |


## Sparse Tensor Networks: Neural Networks for Spatially Sparse Tensors

Compressing a neural network to speedup inference and minimize memory footprint has been studied widely. One of the popular techniques for model compression is pruning the weights in convnets, is also known as [*sparse convolutional networks*](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liu_Sparse_Convolutional_Neural_2015_CVPR_paper.pdf). Such parameter-space sparsity used for model compression compresses networks that operate on dense tensors and all intermediate activations of these networks are also dense tensors.

However, in this work, we focus on [*spatially* sparse data](https://arxiv.org/abs/1409.6070), in particular, spatially sparse high-dimensional inputs and 3D data and convolution on the surface of 3D objects, first proposed in [Siggraph'17](https://wang-ps.github.io/O-CNN.html). We can also represent these data as sparse tensors, and these sparse tensors are commonplace in high-dimensional problems such as 3D perception, registration, and statistical data. We define neural networks specialized for these inputs as *sparse tensor networks*  and these sparse tensor networks process and generate sparse tensors as outputs. To construct a sparse tensor network, we build all standard neural network layers such as MLPs, non-linearities, convolution, normalizations, pooling operations as the same way we define them on a dense tensor and implemented in the Minkowski Engine.

We visualized a sparse tensor network operation on a sparse tensor, convolution, below. The convolution layer on a sparse tensor works similarly to that on a dense tensor. However, on a sparse tensor, we compute convolution outputs on a few specified points which we can control in the [generalized convolution](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html). For more information, please visit [the documentation page on sparse tensor networks](https://nvidia.github.io/MinkowskiEngine/sparse_tensor_network.html) and [the terminology page](https://nvidia.github.io/MinkowskiEngine/terminology.html).

| Dense Tensor                                                                | Sparse Tensor                                                                |
|:---------------------------------------------------------------------------:|:----------------------------------------------------------------------------:|
| <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_dense.gif"> | <img src="https://nvidia.github.io/MinkowskiEngine/_images/conv_sparse.gif"> |

--------------------------------------------------------------------------------

## Features

- Unlimited high-dimensional sparse tensor support
- All standard neural network layers (Convolution, Pooling, Broadcast, etc.)
- Dynamic computation graph
- Custom kernel shapes
- Multi-GPU training
- Multi-threaded kernel map
- Multi-threaded compilation
- Highly-optimized GPU kernels


## Compatibility

Validated source-build targets in this repository:

- Python `3.10` to `3.14`
- PyTorch `2.5` to `2.10`
- Linux `x86_64` CUDA builds against official PyTorch wheel channels `cu124`, `cu126`, `cu128`, and `cu130`
- Linux and macOS CPU-only builds

Current limits:

- Python `3.14` is validated with PyTorch `2.9` and `2.10`
- Windows is out of scope in this pass
- CUDA `13.1` is not claimed yet. Wait for an official PyTorch `cu131` wheel channel before treating it as supported

## Requirements

- A matching PyTorch install must exist before building MinkowskiEngine
- `ninja`
- A BLAS implementation, typically `openblas`
- Linux: a C++17 compiler toolchain and the CUDA toolkit when building with GPU support
- macOS: CPU-only builds, with Homebrew `openblas` and `libomp`

## Installation

`uv` is the supported install and test workflow for this repository. Install PyTorch first, then build MinkowskiEngine from source with `uv pip install --no-build-isolation -v .`.

### CPU-only build with `uv`

Linux:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev
```

macOS:

```bash
brew install openblas libomp
```

Shared steps:

```bash
git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy ninja

MINKOWSKI_CPU_ONLY=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

### CUDA build with `uv`

Use Linux `x86_64`, install a CUDA toolkit that matches the PyTorch wheel channel you selected, and point `CUDA_HOME` at that toolkit.

Example for PyTorch `2.10.0` with CUDA `13.0` wheels:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libopenblas-dev

git clone https://github.com/NVIDIA/MinkowskiEngine.git
cd MinkowskiEngine

uv venv .venv --python 3.13
source .venv/bin/activate

uv pip install --python .venv/bin/python "setuptools>=69" wheel packaging
uv pip install --python .venv/bin/python "torch==2.10.0" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python numpy ninja

export CUDA_HOME=/usr/local/cuda-13.0
MINKOWSKI_FORCE_CUDA=1 MINKOWSKI_BLAS=openblas \
  uv pip install --python .venv/bin/python --no-build-isolation -v .
```

PyTorch release notes determine which wheel channel is valid for a given release. The repository CI validates representative official combinations:

- `torch 2.5` on `cu124`
- `torch 2.6` on `cu126`
- `torch 2.7` on `cu128`
- `torch 2.9` on `cu130`
- `torch 2.10` on `cu130`

The GPU workflow uses self-hosted NVIDIA runners. On push and pull request runs it is
opt-in: set repository variable `MINKOWSKI_GPU_CI_ENABLED=true` and define
`MINKOWSKI_GPU_RUNNER_LABELS` as a comma-separated label list such as
`cuda-12-8,cuda-13-0`. Without that configuration, GPU CI skips cleanly instead of
remaining queued. Manual `workflow_dispatch` runs can pass `runner_labels` directly.

### Build environment variables

The build now uses environment variables instead of `setup.py install` flags:

- `MINKOWSKI_CPU_ONLY=1` forces a CPU-only build
- `MINKOWSKI_FORCE_CUDA=1` requires a CUDA build and fails if torch or `CUDA_HOME` are not CUDA-capable
- `MINKOWSKI_BLAS=openblas|mkl|atlas|flexiblas|blas` chooses the BLAS backend
- `MINKOWSKI_BLAS_INCLUDE_DIRS=/path/one,/path/two` overrides BLAS header discovery
- `MINKOWSKI_BLAS_LIBRARY_DIRS=/path/one,/path/two` overrides BLAS library discovery
- Existing toolchain variables such as `CUDA_HOME`, `CXX`, `MAX_JOBS`, `USE_NINJA`, and `TORCH_CUDA_ARCH_LIST` are still honored
- On macOS, the runtime defaults `OMP_NUM_THREADS=1` unless you override it explicitly

### API compatibility notes

- `MinkowskiConvolutionTranspose` and `MinkowskiGenerativeConvolutionTranspose` still accept `generate_new_coords`, but it is now a deprecated alias for `expand_coordinates`
- Point-cloud tests use the tracked fixture at `tests/data/1.ply`; they no longer download data at import time

### C++ test extensions

The C++ backend test build also uses the shared build helper. See `tests/cpp/README.md` for the updated `uv` workflow.


## Quick Start

To use the Minkowski Engine, you first would need to import the engine.
Then, you would need to define the network. If the data you have is not
quantized, you would need to voxelize or quantize the (spatial) data into a
sparse tensor.  Fortunately, the Minkowski Engine provides the quantization
function (`MinkowskiEngine.utils.sparse_quantize`).


### Creating a Network

```python
import torch.nn as nn
import MinkowskiEngine as ME

class ExampleNetwork(ME.MinkowskiNetwork):

    def __init__(self, in_feat, out_feat, D):
        super(ExampleNetwork, self).__init__(D)
        self.conv1 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_feat,
                out_channels=64,
                kernel_size=3,
                stride=2,
                dilation=1,
                bias=False,
                dimension=D),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU())
        self.conv2 = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=2,
                dimension=D),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU())
        self.pooling = ME.MinkowskiGlobalPooling()
        self.linear = ME.MinkowskiLinear(128, out_feat)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pooling(out)
        return self.linear(out)
```

### Forward and backward using the custom network

```python
    # loss and network
    criterion = nn.CrossEntropyLoss()
    net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    input = ME.SparseTensor(feat, coordinates=coords)
    # Forward
    output = net(input)

    # Loss
    loss = criterion(output.F, label)
```

## Discussion and Documentation

For discussion and questions, please use `minkowskiengine@googlegroups.com`.
For API and general usage, please refer to the [MinkowskiEngine documentation
page](http://nvidia.github.io/MinkowskiEngine/) for more detail.

For issues not listed on the API and feature requests, feel free to submit
an issue on the [github issue
page](https://github.com/NVIDIA/MinkowskiEngine/issues).


## Known Issues

### Specifying CUDA architecture list

In some cases, you need to explicitly specify which compute capability your GPU uses. The default list might not contain your architecture.

```bash
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
MINKOWSKI_FORCE_CUDA=1 uv pip install --python .venv/bin/python --no-build-isolation -v .
```

### Too much GPU memory usage or Frequent Out of Memory

There are a few causes for this error.

1. Out of memory during a long running training

MinkowskiEngine is a specialized library that can handle different number of points or different number of non-zero elements at every iteration during training, which is common in point cloud data.
However, pytorch is implemented assuming that the number of point, or size of the activations do not change at every iteration. Thus, the GPU memory caching used by pytorch can result in unnecessarily large memory consumption.

Specifically, pytorch caches chunks of memory spaces to speed up allocation used in every tensor creation. If it fails to find the memory space, it splits an existing cached memory or allocate new space if there's no cached memory large enough for the requested size. Thus, every time we use different number of point (number of non-zero elements) with pytorch, it either split existing cache or reserve new memory. If the cache is too fragmented and allocated all GPU space, it will raise out of memory error.

**To prevent this, you must clear the cache at regular interval with `torch.cuda.empty_cache()`.**

### Matching CUDA and PyTorch

Make sure the installed PyTorch wheel channel and the local CUDA toolkit match. For example, a `cu130` torch install should build against a CUDA `13.0` toolkit exposed through `CUDA_HOME`.

### Running the MinkowskiEngine on nodes with a large number of CPUs

The MinkowskiEngine uses OpenMP to parallelize the kernel map generation. However, when the number of threads used for parallelization is too large (e.g. OMP_NUM_THREADS=80), the efficiency drops rapidly as all threads simply wait for multithread locks to be released.
In such cases, set the number of threads used for OpenMP. Usually, any number below 24 would be fine, but search for the optimal setup on your system.

```
export OMP_NUM_THREADS=<number of threads to use>; python <your_program.py>
```

## Citing Minkowski Engine

If you use the Minkowski Engine, please cite:

- [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR'19](https://arxiv.org/abs/1904.08755), [[pdf]](https://arxiv.org/pdf/1904.08755.pdf)

```
@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}
```

For multi-threaded kernel map generation, please cite:

```
@inproceedings{choy2019fully,
  title={Fully Convolutional Geometric Features},
  author={Choy, Christopher and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8958--8966},
  year={2019}
}
```

For strided pooling layers for high-dimensional convolutions, please cite:

```
@inproceedings{choy2020high,
  title={High-dimensional Convolutional Networks for Geometric Pattern Recognition},
  author={Choy, Christopher and Lee, Junha and Ranftl, Rene and Park, Jaesik and Koltun, Vladlen},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```

For generative transposed convolution, please cite:

```
@inproceedings{gwak2020gsdn,
  title={Generative Sparse Detection Networks for 3D Single-shot Object Detection},
  author={Gwak, JunYoung and Choy, Christopher B and Savarese, Silvio},
  booktitle={European conference on computer vision},
  year={2020}
}
```


## Unittest

For unittests and gradcheck, use torch >= 1.7

## Projects using Minkowski Engine

Please feel free to update [the wiki page](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage) to add your projects!

- [Projects using MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine/wiki/Usage)

- Segmentation: [3D and 4D Spatio-Temporal Semantic Segmentation, CVPR'19](https://github.com/chrischoy/SpatioTemporalSegmentation)
- Representation Learning: [Fully Convolutional Geometric Features, ICCV'19](https://github.com/chrischoy/FCGF)
- 3D Registration: [Learning multiview 3D point cloud registration, CVPR'20](https://arxiv.org/abs/2001.05119)
- 3D Registration: [Deep Global Registration, CVPR'20](https://arxiv.org/abs/2004.11540)
- Pattern Recognition: [High-Dimensional Convolutional Networks for Geometric Pattern Recognition, CVPR'20](https://arxiv.org/abs/2005.08144)
- Detection: [Generative Sparse Detection Networks for 3D Single-shot Object Detection, ECCV'20](https://arxiv.org/abs/2006.12356)
- Image matching: [Sparse Neighbourhood Consensus Networks, ECCV'20](https://www.di.ens.fr/willow/research/sparse-ncnet/)
