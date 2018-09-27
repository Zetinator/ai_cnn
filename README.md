# ai_cnn
This is the design and implementation of a neural network able to drive a car autonomously.

## How to build
In order to build this **CNN** the following additional packages are required:
- **tensorflow** (with the optional **gpu** support recommended)
- **opencv**
- **matplotlib**
- **keras**

> All of them available within the conda packages list

### suggested set-up
use the following commands to create a virtual environment using **Miniconda3**:

```bash
conda create -n tensorflow_env python=3.5
source activate tensorflow_env
conda install tensorflow-gpu opencv matplotlib pydot keras
```

## How to train
To train this CNN you need to go to the folder where your images are... and execute the python-script called `to_npz.py`, in order to compress the images into an `.npz` compressed format, and the move this `.npz` file into the folder called `dataset/<the_name_of_your_dataset>` in the root of this project (where this `README.md` is).

The execute the following commands:

```bash
source activate tensorflow_env
python3 interface.py --dataset=<the_name_of_your_dataset>
```
