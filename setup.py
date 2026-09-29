# Copyright (c) POSTECH / ETH Zurich / Google.
# Redistributes portions of Meta's SAM 2 under the Apache 2.0 License (see LICENSE).

import os
from setuptools import find_packages, setup

NAME = "moga-sam2"
VERSION = "0.1.0"
DESCRIPTION = (
    "Robust Promptable Video Object Segmentation with MoGA — "
    "Memory-object-conditioned Gated-rank Adaptation of SAM 2."
)
URL = "https://sohyun-l.github.io/RobustPVOS_project_page/"
LICENSE = "Apache 2.0"

with open("README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

REQUIRED_PACKAGES = [
    "torch>=2.5.1",
    "torchvision>=0.20.1",
    "numpy>=1.24.4",
    "tqdm>=4.66.1",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "Pillow>=9.4.0",
]

# The connected-components CUDA extension is optional; SAM 2 auto-falls back to
# CPU postprocessing when it is not built, so keep the build off by default.
BUILD_CUDA = os.environ.get("MOGA_BUILD_CUDA_EXT", "0") == "1"
ext_modules = []
if BUILD_CUDA:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    ext_modules = [
        CUDAExtension(
            "sam2._C",
            sources=["sam2/csrc/connected_components.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ]

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=find_packages(exclude=("eval", "scripts")),
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=REQUIRED_PACKAGES,
    ext_modules=ext_modules,
    cmdclass=(
        {"build_ext": __import__("torch").utils.cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True)}
        if BUILD_CUDA
        else {}
    ),
)
