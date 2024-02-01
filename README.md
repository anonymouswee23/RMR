# RMR
This repository contains the code of the submitted paper "Efficient Matrix Sparsification for Optimimization Applications: The Randomized Minor-value Rectification (RMR) Algorithm".
## Dependencies
* `portpy`
## Usage
You need to download influence matrix of patients from this [link](https://github.com/PortPy-Project/PortPy?tab=readme-ov-file#data-).

Here are some samples of how to use:

```
python main.py --method Naive --patient Lung_Patient_1 --threshold 0.008
python main.py --method AHK06 --patient Lung_Patient_1 --threshold 0.02
python main.py --method AKL13 --patient Lung_Patient_1 --threshold 1000000
python main.py --method DZ11 --patient Lung_Patient_1 --threshold 10
python main.py --method RMR --patient Lung_Patient_1 --threshold 0.02
```
## Demo
You can try the project with colab demo `demo.ipynb`.
