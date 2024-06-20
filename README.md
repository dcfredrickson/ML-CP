# ML-CP
Machine-Learned Chemical Pressure Analysis

This repository is divided into two directories. In the "ML-CP training" directory, you will find the code and data used to produce the two models discussed in the manuscript [MANUSCRIPT INFORMATION]. The models are exported as .pickle files. With these pickle files, you can use the code and data contained in "trained ML-CP model" directory to make ML-CP schemes. 

All code is written in python, and may require python version 3.8 and SciKit Learn version 1.3.2 to run. At the very least, the training and utility code will need the same version of python and SciKit Learn. 

To run the machine_learning.py code to produce the classifier and regressor models, download the files contact_data_by_cp.xlsx, element_data.xlsx, and machine_learning.py. With these three files in the same directory, run the command 
`python3.8 machine_learning.py`
