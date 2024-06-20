# ML-CP
Machine-Learned Chemical Pressure Analysis

This repository is divided into two directories. In the "ML-CP training" directory, you will find the code and data used to produce the two models discussed in the manuscript [MANUSCRIPT INFORMATION]. The models are exported as .pickle files. With these pickle files, you can use the code and data contained in "trained ML-CP model" directory to make ML-CP schemes. 

All code is written in python, and may require python version 3.8 and SciKit Learn version 1.3.2 to run. At the very least, the training and utility code will need the same version of python and SciKit Learn. 

To run the machine_learning.py code to produce the classifier and regressor models, download the files contact_data_by_cp.xlsx, element_data.xlsx, and machine_learning.py from the "ML-CP training" GitHub directory. With these three files in the same local directory, run the command

`python3.8 machine_learning.py`

This operation should take 1-2 minutes, and produces two new files, rfc.pickle and rfr.pickle. Now to produce ML-CP data, copy the resulting .pickle files to a new local directory containing the element_data.txt and mlcp.py files from the "trained ML-CP model" GitHub directory. You'll also need to include in this local directory a Crystallographic Information File (CIF file) for the structure you're interested in producing ML-CP data for. With the two .pickle files, the element_data.txt file, the mlcp.py file, and the CIF file in the same directory, run the command

`python3.8 mlcp.py [CIF filename]`

Optionally, you can also provide a scale factor and template. Scale factors can be used to produce a ML-CP scheme with a total CP closer to 0 by artificially scalling the a, b, and c cell lengths. subcell templates can be used to produce ML-CP data for subsets of the unit cell. This functionallity allows the user to produce ML-CP data for regions of a unit cell with lower requirements for memory allocation and time. The mlcp.py program currently only accepts XYZ files, which can be produced using VESTA or other structure viewing programs. To run the mlcp.py program with a scale factor, run the command

`python3.8 mlcp.py [CIF filename] [scale factor]`

To run the mlcp.py program with a scale factor and template or just a template, run the commands

`python3.8 mlcp.py [CIF filename] [scale factor] [template filename]`

or

`python3.8 mlcp.py [CIF filename] 1 [template filename]`

respecively. In the latter case, a scale factor is still required so 1 is used as a placeholder without modifiying the scale. 
