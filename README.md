# MscProject

## Preparation
### Python
* Using Python 3.7.
* Install Numpy Library.
* Install Pandas Library.
* Install Sklearng Library.
* Install Matplotlib Library
  
### PyTorch
* Install [PyTorch](http://pytorch.org/)

### OpenCV
* Built up OpenCV-Python by following [this official site](https://docs.opencv.org/4.4.0/da/df6/tutorial_py_table_of_contents_setup.html)

### EgoGesture
* Download videos by following [the official site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
* Rename the folder 'EgoGesture' and out it under the ./datasets/ . The folder should look like './datasets/EgoGesture/Subject01/Scene1/Color/rgb1/'

## Path
* The following commands are running under the root folder of this project.
* please cd into the root folder of project.

## Training and Testing
### Training
* Modify the root path in 'my_run.sh' into your path.
* Using the commands in 'my_test.sh' to training different models.
* Change the sample duration in 'my_run.sh'.
* Enable support of CUDA in 'my_run.sh' by deleting the line '--no_cuda'.
* The result will be put under ./result/

### Testing
* using my_test_single.py to testing on centre clip of testing set. Change the 'dir' into your directory where put the testing set.
* using my_test.py to testing on all clips of testing set with the system. Change the 'dir' into your directory where put the testing set.
* This will generate CSV files in './TestGesture/single/' and './TestGesture/continuous/', which you may need to change the path for successfully running.
### Analysis
* using 'statistic.py' in ./analysis/ to get the statistic data.
* change the paths in list variable named 'csv' in 'statistic.py' to get data from different group results.
* using 'vs.py' to get the visualised results of loss and precision.

### GUI
* using 'Python ui.py' to open the GUI.
  
### Reference 
The code used is modified basing the code provided in [this web site](https://github.com/ahmetgunduz/Real-time-GesRec).