# DeepGraphCut
This repository contains the implementation for the paper titled "A Deep Graph Cut Model For 3D Brain Tumor Segmentation" published in the 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). THe link for the paper is given below -
https://ieeexplore.ieee.org/document/9871685/

Requirements:
pytorch (see https://pytorch.org/get-started/locally/)
monai (see https://docs.monai.io/en/stable/installation.html)
cython (see https://cython.readthedocs.io/en/latest/src/quickstart/install.html)
pygco (see https://github.com/amueller/gco_python)
tqdm (pip install tqdm or conda install tqdm)
simpleitk (pip install SimpleITK or conda install -c simpleitk simpleitk)
tensorboard (pip install tensorboard or conda install -c conda-forge tensorboard)

To run the code,
1.  Train the model by running train.py
2.  Run the graphcut.py code to get the deep graphcut output. The output images will be saved in the folder "DGC_output"
