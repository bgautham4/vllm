# Requirements
1. CUDA (runtime drivers) >= 12.1

# Anaconda installation
Download miniconda installer from [Here](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh).
```bash
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Follow through the installation and prompt *yes* when asked for automatic initialization of conda by modifying your shell profile.

Now source your shell profile (.bashrc .zshrc etc..) which will bring `conda` into `PATH`.

```bash
source ~/.bashrc #If bash is your shell
```

## Disable base environment
Conda by default will init base environment. To stop this:
```bash
conda config --set auto_activate_base false
#Source shell profile again
source ~/.bashrc # For bash
```

# Creating base environment and installing vllm
## Create and activate env
```bash
conda create -n <env_name> python=3.12
#Follow through installation
conda activate <env_name>
```

## Install base vllm (conda env should be active)
This is needed for the CUDA binaries and remainaing shared objects required by VLLM.
```bash
export COMMIT_HASH=3cb07a36a20f9af11346650559470d685e9dc711 #Commit hash for this version of vllm
pip install https://vllm-wheels.s3.us-west-2.amazonaws.com/${COMMIT_HASH}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
```

## Final step (conda env should be active)
Clone my version of VLLM and symlink the python files in this repo to the pip installed version.
```bash
git clone https://github.com/bgautham4/vllm.git my_vllm_source
cd my_vllm_source
python python_only_dev.py #will create symlinks
```
In case you wish to remove symlinks and go back to pip installed version of vllm then:


```python python_only_dev.py -q```

# Test working
```bash
cd my_vllm_source/examples
```
Edit `offline_inference.py` line 13
```python
llm = LLM(model="facebook.opt-125m")
#To
llm = LLM(model="ibm-granite/granite-3.1-8b-instruct", max_model_len=4096)
```
Run the file, this will not only test working of vllm installation, it will also cache model weights of IBM-granite which is needed for experiment.

# Run experiment (6-8 hrs)
## One time setup if possible
If you have access to GPU and super user (i.e can change GPU frequency etc), then configure your non root user to be able to execute nvidia-smi without password authentication by adding tthe following line to the `/etc/sudoers` file using the `visudo` program: (Exclude the < and > characters, they are used to indicate placeholders!)
```sudoers
#Add this line near the end
<your-user> ALL=(root) NOPASSWD:<absolute path to nvidia-smi (probably /usr/bin/nvidia-smi)>
```
Test if you are able to execute the following without getting prompted for password:
```bash
sudo nvidia-smi --lock-gpu-clocks=1380,1380
```
And edit line 47 in `my_vllm_source/benchmarks/combined_exp.sh`:
```bash
sudo nvidia-smi --lock-gpu-clocks=870,870
#To
sudo nvidia-smi --lock-gpu-clocks=1380,1380
```

## If above is not possible, then:
Simply comment out line 47

## Run
```bash
cd my_vllm_source/benchmarks
./combined_exp.sh --model ibm-granite/granite-3.1-8b-instruct --max-seq-len 4096
```
After run logs and metrics file are stored in `results` directory. You can create a tarball and compress it so you can share the results.
