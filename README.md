# LT_DSE

If you failed to install and run this tracker, please email me (<dkn2014@mail.dlut.edu.cn>)

# Introduction

### :star2:The Winner Tracker of the VOT-2019 LT Challenge.:star2: 

# Prerequisites

* python 3.6
* ubuntu 16.04
* cuda-9.0
* gcc 5.4
* g++ 5.4
# Installation
1. Clone the GIT repository:
```
 $ git clone https://github.com/Daikenan/LT_DSE.git
```
2. Clone the submodules.  
   In the repository directory, run the commands:
```
   $ git submodule init  
   $ git submodule update
```
3. Run the install script. 
Usage example:
```
bash install.sh ~/anaconda3 votenvs
```
The first parameter `~/anaconda3` indicates the path of anaconda and the second indicates the virtual environment used for this project. 

4. modify ``local_path.py``:

``toolkit_path`` is not needed if you don't test on VOT toolkit.

5. Run the demo script to test the tracker:
```
source activate votenvs
python LT_DSE_Demo.py
```

# Integrate into VOT-2019LT

## VOT-toolkit
Before running the toolkit, please change the environment path to use the python in the conda environment "votenvs".
For example, in my computer, I add  `export PATH=/home/daikenan/anaconda3/envs/votenvs/bin:$PATH` to the `~/.bashrc` file.  

The interface for integrating the tracker into the vot evaluation tool kit is implemented in the module `tracker_vot.py`. The script `tracker_LT_DSE.m` is needed to be copied to vot-tookit. 

Since the vot-toolkit may be not compatible with pytorch-0.4.1, I always change the line  `command = sprintf('%s %s -c "%s"', python_executable, argument_string, python_script);` to `command = sprintf('env -i %s %s -c "%s"', python_executable, argument_string, python_script);` in `generate_python_command.m`. 
