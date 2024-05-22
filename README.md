# RMR
This repository contains the code of the submitted paper "Randomized Minor-value Rectification: A Novel Matrix Sparsification Technique for Solving Constrained Optimizations in Cancer Radiation Therapy".
## Dependencies
* `portpy`
## Usage
You need to download influence matrix of patients from this [link](https://github.com/PortPy-Project/PortPy?tab=readme-ov-file#data-).

Here are some samples of how to use:

```
python main.py --method Naive --patient Lung_Patient_1 --threshold 0.008
python main.py --method AHK06 --patient Lung_Patient_1 --threshold 0.05
python main.py --method AKL13 --patient Lung_Patient_1 --threshold 1000000
python main.py --method DZ11 --patient Lung_Patient_1 --threshold 50
python main.py --method RMR --patient Lung_Patient_1 --threshold 0.05
```

If you have MOSEK license it is recommended to run the code with MOSEK solver:
```
python main.py --method Naive --patient Lung_Patient_1 --threshold 0.008 --solver MOSEK
```
## Demo
You can try the project with [demo](https://mybinder.org/v2/gh/anonymouswee23/RMR/HEAD?labpath=demo.ipynb).
