from setuptools import  setup
import torch
import os,glob
from torch.utils.cpp_extension import (CUDAExtension, CppExtension, BuildExtension)


def get_extensions():
    extensions = []
    ext_name = 'deform3dattn_custom_cn' 
    # prevent ninja from using too many resources
    # os.environ.setdefault('MAX_JOBS', '4')
    try:
        import psutil
        num_cpu = len(psutil.Process().cpu_affinity())
        cpu_use = max(4, num_cpu - 1)
    except (ModuleNotFoundError, AttributeError):
        cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
    define_macros = []

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        define_macros += [('WITH_CUDA', None)]
        # op_files = glob.glob('./csrc/*')
        op_files = glob.glob('./csrc/*.cu')
        extension = CUDAExtension 
    else:
        print(f'Compiling {ext_name} without CUDA')
        op_files = glob.glob('./csrc/*.cpp')
        extension = CppExtension

    include_path = os.path.abspath('./csrc')
    ext_ops = extension( 
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path],
        define_macros=define_macros)
    extensions.append(ext_ops)
    return extensions 

setup(
    name='extension_example',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension}, 
    )
