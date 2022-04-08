# TensorFlow (GPU) Setup on Windows

This document shows how to install TensorFlow on Windows. Windows 10 was used here.

> **Note**: This document was not tested on Windows 11

## Table of contents

- [TensorFlow (GPU) Setup on Windows](#tensorflow-gpu-setup-on-windows)
    - [Table of contents](#table-of-contents)
    - [Hardware Requirements](#hardware-requirements)
    - [Installation on System Interpreter](#installation-on-system-interpreter)
        - [Step 1: CUDA Installation on Windows](#step-1-cuda-installation-on-windows)
            - [Optional Step 1.2: Compiling a CUDA Program](#optional-step-12-compiling-a-cuda-program)
        - [Step 2: cuDNN Installation (on Windows CUDA)](#step-2-cudnn-installation-on-windows-cuda)
        - [Step 3: Install and test TensorFlow](#step-3-install-and-test-tensorflow)
    - [Installation on Anaconda environment](#installation-on-anaconda-environment)

## Hardware Requirements

As shown on [TensorFlow website](https://www.tensorflow.org/install/gpu#hardware_requirements), an NVIDIA GPU is needed for hardware acceleration on TensorFlow.

- CUDA Enabled GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Search for your device (GPU in your system) and check the compute capability. Higher is better.

## Installation on System Interpreter

This will install everything on a system interpreter. It's best to install Python 3.9 from [Microsoft Store](https://www.microsoft.com/en-us/p/python-39/9p7qfqmjrfp7?activetab=pivot:overviewtab) and use that as the Python interpreter. This is recommended if the system is a development machine.

Microsoft Visual Studio 2019 was also used. You can get the free community version from [here](https://visualstudio.microsoft.com/vs/). This is needed for the C++ compiler.

### Step 1: CUDA Installation on Windows

Reference from [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html). CUDA is for **parallel computation** on NVIDIA GPUs

Download and Install CUDA Toolkit from [here](https://developer.nvidia.com/cuda-downloads). CUDA Toolkit 11.5, Windows 10, x86_64, local install was tested. Let the installer use the `AppData` folder to extract everything (temporary files).

- Usually the CUDA Toolkit is saved in `%ProgramFiles%\NVIDIA GPU Computing Toolkit\CUDA\v#.#` (like `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5`).
- The samples included with CUDA are stored in `%ProgramData%\NVIDIA Corporation\CUDA Samples\v#.#` (like `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5`)
- Nsight extension for Visual Studio can be found in `%ProgramFiles(x86)%\NVIDIA Corporation\Nsight Visual Studio Edition #.#` (like `C:\Program Files (x86)\NVIDIA Corporation\Nsight Visual Studio Edition 2021.3`).

The above paths are listed in the installation agreement (at the top of document). Make sure that the `Path` environment variable contains the above paths before proceeding (the installer should normally do that on its own).

#### Optional Step 1.2: Compiling a CUDA Program

This step can be skipped. More information about this section can be found on [NVIDIA docs](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#compiling-cuda-programs).

Windows uses Microsoft Visual Studio for development. The path `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5\` contains samples that can be compiled. Some of these programs are useful utilities for NVIDIA CUDA. Here, `1_Utilities\deviceQuery` is built. This utility gives the details of the device.

1. Go to `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5\1_Utilities\deviceQuery`
2. Run `deviceQuery_vs2019.vcxproj`: open the project in Visual Studio 2019. Change the dropdown from `Debug` to `Release`.
3. Go to `Build` -> `Build Solution`. The resulting `exe` must be saved in `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5\bin\win64\Release`.

You can now run

```pwsh
cd C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5\bin\win64\Release
.\deviceQuery.exe
```

This will give the details about your GPU (the compute capability). You can also add `C:\ProgramData\NVIDIA Corporation\CUDA Samples\v11.5\bin\win64\Release` to `Path` environment variable (if you want to run these commands without `cd`).

First CUDA Project built! :tada::smile:

You can try building other projects the same way and get the hang of CUDA.

### Step 2: cuDNN Installation (on Windows CUDA)

Reference from [NVIDIA docs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). The cuDNN is a library for **Deep Learning** on top of CUDA. More about it [here](https://developer.nvidia.com/cudnn).

First, you must install [zlib](https://www.zlib.net/) which is a compression library.

1. Download ZLIB zip from [here](http://www.winimage.com/zLibDll/zlib123dllx64.zip).
2. Extract the contents to `C:\tools\zlib` (create folder `tools` for generic tools that are to be used system wide, if it doesn't exist).
3. Add `C:\tools\zlib\dll_x64` to the `Path` environment variable. This is the location where `zlibwapi.dll` can be found.

Now that everything is ready, we can proceed with cuDNN installation

1. Download cuDNN from [here](https://developer.nvidia.com/rdp/cudnn-download). You will need an NVIDIA Developer Program Membership (which is free), just create an account if you don't have one already. It'll be a ZIP file containing everything pre-built.
2. Extract everything to `C:\tools\cuda`, such that the 'bin', 'include' and 'lib' folder are here.
3. Copy all `cudnn*.dll` files (around 7 items) from `C:\tools\cuda\bin` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin`
4. Copy all `all cudnn*.h` files (around 9 items) from `C:\tools\cuda\include` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\include`
5. Copy all `cudnn*.lib` files (around 14 items) from `C:\tools\cuda\lib\x64` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\lib\x64`

The last three steps from above add cuDNN to the list of CUDA libraries on the system.

Add `C:\tools\cuda\bin` path to `Path` environment variable. TensorFlow will need this in order to access the `cudnn64_8.dll` library.

### Step 3: Install and test TensorFlow

Install tensorflow using pip. The following command can be run on PowerShell

```pwsh
pip install tensorflow
```

That's it! :confetti_ball:

Just to test that everything works, run the following

```py
import tensorflow as tf
tf.config.list_physical_devices()
tf.config.list_physical_devices('GPU')
sys_details = tf.sysconfig.get_build_info()
for k in sys_details:
    print(f"{k}: {sys_details[k]}")
```

This should list `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')` as a physical device and should also show the CUDA properties.

In case you get an error saying that the `*.dll` files cannot be found, something like [this](https://github.com/tensorflow/tensorflow/issues/48868), then you can first add the dll directory. Simply, try running

```py
import os
# Whatever is the path containing the 'cudart64_110.dll' file (note '/', not '\')
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
import tensorflow as tf
tf.config.list_physical_devices()
tf.config.list_physical_devices('GPU')
sys_details = tf.sysconfig.get_build_info()
for k in sys_details:
    print(f"{k}: {sys_details[k]}")
```

This should fix the problem.

## Installation on Anaconda environment

This is if you have [Anaconda](https://anaconda.org/) installed. Do the Steps 1 and 2 (installing CUDA and cuDNN) from above. Then, do the following

1. Create (and activate) the Anaconda environment

    ```pwsh
    conda create -n "tf-gpu"
    conda activate tf-gpu
    ```

2. Install Python 3.9 into it (Python 3.9 was tested here, you can try other versions)

    ```pwsh
    conda install python=3.9
    ```

3. Install CUDA for the Anaconda environment

    ```pwsh
    conda install cuda -c nvidia
    ```

4. Install tensorflow using pip

    ```pwsh
    pip install tensorflow
    ```

That's it! :tada:

Verify the installation by running

```py
import tensorflow as tf
tf.config.list_physical_devices()
tf.config.list_physical_devices('GPU')
sys_details = tf.sysconfig.get_build_info()
for k in sys_details:
    print(f"{k}: {sys_details[k]}")
```

This should show the physical graphics cards being detected by TensorFlow and the build information.
