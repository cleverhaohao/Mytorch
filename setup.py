import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
__version__ = '0.0.1'
sources = ['src/pybind.cpp','src/Tensor.cu']
setup(
 name='mytensor',
 version=__version__,
 author='hqh',
 author_email='hanqinghao@stu.pku.edu.cn',
 packages=find_packages(),
 zip_safe=False,
 install_requires=['torch'],
 python_requires='>=3.8',
 license='MIT',
 ext_modules=[
CUDAExtension(
name='mytensor',
sources=sources,
extra_compile_args={
    'nvcc':['-O2','-lcublas','-lcurand']
},
libraries=['cublas', 'curand'],
),
],
 cmdclass={
    'build_ext': BuildExtension
    },
 classifiers=[
    'License :: OSI Approved :: MIT License',
    ],
)