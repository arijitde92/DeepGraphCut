# DeepGraphCut
This repository contains the implementation for the paper titled "A Deep Graph Cut Model For 3D Brain Tumor Segmentation" published in the 2022 44th Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). THe link for the paper is given below -
https://ieeexplore.ieee.org/document/9871685/

Requirements:
- pytorch (see https://pytorch.org/get-started/locally/)
- monai (see https://docs.monai.io/en/stable/installation.html)
- cython (see https://cython.readthedocs.io/en/latest/src/quickstart/install.html)
- pygco (see https://github.com/amueller/gco_python)
- tqdm (pip install tqdm or conda install tqdm)
- simpleitk (pip install SimpleITK or conda install -c simpleitk simpleitk)
- tensorboard (pip install tensorboard or conda install -c conda-forge tensorboard)

Preprocessed data is available at: https://drive.google.com/drive/folders/1Laad-KpxvySlCpyKMrXb_4nH97lNiyR0?usp=share_link

To run the code,
1.  First download the data and store in a folder called "Data". The directory structure should be like below-

    Data
    
    |------train
    
            |-------image
            
            |-------mask
            
    |------test
    
            |-------image
            
            |-------mask
    
    The data should be in compressed NifTi format (.nii.gz).
    
    Create three folders "saved_images", "trained_models" and "DGC_output" which will be used in the next steps.
2.  Train the model by running train.py
3.	Evaluate the model by running eval.py. Predicted outputs, probability maps fill be generated in "saved_images" folder.
4.  Run the graphcut.py code to get the deep graphcut output. The output images will be saved in the folder "DGC_output".
