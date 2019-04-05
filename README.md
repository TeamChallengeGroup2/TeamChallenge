# TeamChallenge (Team 2)

04-04-2019

Automatic segmentation of the end-systole and end-distole MR images. The 
segmentations are used to calculate the Ejection Fraction.

The main.py file should be runned. In this file, functions from other files are 
imported. 

In the INPUT section (lines 21-32) the user can change the parameters of the model.

In the second section (line 36) the data will be loaded using the loadData() 
function from the Data.py file. It uses as input the directory to your database. 
The output is a list of lists with the patient number, number of slices per frame, 
the ED-frame, the ED ground truth-frame, the ES-frame and the ES ground truth-frame.
After loading the data, the data is shuffled in order to take a random set for 
training later on. Also the info with the patient number, number of slices and 
spacing is saved.

In the third section (line 57-59) the data is cropped using the cropImage() function 
from the Cropping.py file. Looping trough the data the ED-frame and ES-frame are 
extracted and cropped by the cropROI(). The cropROI uses as input a frame and a 
number indicating the amount of slices that should be taken into account during 
searching for the ROI. All slices from the same frame are cropped around the 
same coordinates.

In the fourth section (line 62-105) the data is split into a set with 50% for 
training, 25% for validation and 25% for testing. Note that the validation set
is included in the training set in the code. In addition, the ground truth frames
were converted to binary masks with a threshold of 2.5 as the left ventricle (LV)
is labeled with 3. 

In the fifth section (line 108-110) some data will be augmented and added to the 
original dataset. This is done using the create_Augmented_data() function from the 
DataAugmentation.py file. It uses as input the images to train and their corresponding
ground truth, the number of batches to augment for and the batch size. Augmentation 
is performed by rotation in a range of 90 degrees, horizontal flip and vertical
flip. As output, two arrays for the images and the ground truth used for 
training including the augmented data is given.

In the sixth section (line 114-136) the training of the network is performed. As 
mentioned before, 50% of the data is used for training. As network 
the fcn network is used, is build calling fcn_model(). 

In the seventh section (line 139-158), the test images (25% of the data) are predicted and
validated using some metrics like the Dice coefficient, accuracy, specificity and
sensitivity computed with the function metrics() with as input the predicted mask 
and their corresponding ground truth. 

Finally, in section eight (line 161-201) the ejection fraction for each testing
patient is calculated twice. Once based upon the segmenting done by the model 
and once based upon the ground truth manually made by experts.

![Results][../figures/ef_test.png]
