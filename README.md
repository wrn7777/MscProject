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
* Built up OpenCV by following [this official site](https://docs.opencv.org/4.4.0/da/df6/tutorial_py_table_of_contents_setup.html)

### EgoGesture
* Download videos by following [the official site](http://www.nlpr.ia.ac.cn/iva/yfzhang/datasets/egogesture.html).
* Rename the folder 'EgoGesture' and out it under the ./datasets/ . The folder should look like './datasets/EgoGesture/Subject01/Scene1/Color/rgb1/'

## Training and Testing
### Training
* Modify the root path in 'my_run.sh'.
* Using the commands in 'my_test.sh' to training different models.
* Change the sample duration in 'my_run.sh'.
* The result will be put under ./result/

### Testing
* using my_test_single.py to testing on centre clip of testing set. Change the 'dir' into your directory where put the testing set.
* using my_test.py to testing on all clips of testing set with the system. Change the 'dir' into your directory where put the testing set.

### GUI
* using 'Python ui.py' to open the GUI.
### Reference 
The code used is modified basing the code provided in [this web site](https://github.com/ahmetgunduz/Real-time-GesRec).