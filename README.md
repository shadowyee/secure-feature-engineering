# 1. Environment
Python=3.7
Pysyft=0.2.x
torch=1.4.0
...

# 2. Installation
* Install Pysyft 0.2.x-fix-training package:
```bash
wget https://github.com/OpenMined/PySyft/archive/refs/heads/ryffel/0.2.x-fix-training.zip
unzip 0.2.x-fix-training.zip
```
* Install conda: 
  1. [miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
  2. [anaconda](https://www.anaconda.com/download#downloads)
* Create conda env:
```bash
conda create -n <env_name> python=3.7
conda activate <env_name>
```
* Install dependency:
```bash
cd 0.2.x-fix-training
pip install -e .
```
If you encountered some errors when installing dependency, try to comment out dict object '''extras_require''' in ```setup.py```.
Maybe `pip install protobuf~=3.19.0` is also required.

* Pysyft 0.2.x bug fix:
Because 0.2.x version is not supported anymore, you may encounter some bug when using it.
If you want to run this project correctly, try to modify **0.2.x-fix-training/syft/frameworks/torch/tensors/interpreters/gradients.py line 40**:
```python
<<< old code 
if not isinstance(self.other.child, int):
>>> new code
if torch.is_tensor(self.other) and not isinstance(self.other.child, int):
```

