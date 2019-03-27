# TeamChallenge

27-03-2019

Automatic segmentation of the end-systole and end-distole MR images. The segmentations are used to calculate the Ejection Fraction.

The main.py file should be runned. In this file, functions from other files are imported. 

In the INPUT section (line 26) you can change the parameters of the model.

In the second section (line 38) the data will be loaded using the loadData() function form the Data.py file.
It uses as input the directory to your database. The output is a list of lists with patient number, number of slices per frame, the ED-frame, the ED ground truth-frame, the ES-frame and the ES ground truth-frame.

In the third section (line 43) some data will be augmented and added to the original dataset. This is done using the augmentation() function from the Augmentation.py file.
It uses as input the list with your loaded data and a number for the amount of patients who's data will be augmented.
The output is a list containg the original data and the augmented data. This list has the same structure as your original data list.

In the fourth section (line 49) the data is cropped using the cropROI() function from the Cropping.py file. 
Looping trough the data the ED-frame and ES-frame are extracted and cropped by the cropROI(). 
The cropROI uses as input a frame and a number indicating the amount of slices that should be taken into account during searching for the ROI. All slices from the same frame are cropped around the same coordinates.

In the fifth section (line 79) the training of the network will be done. The data is split up in 50% trainig, 25% validation and 25% testing set. As network the fcn network is used, is build calling fcn_model(). This function uses the shape of the patches as input.

In the sixth section (line 188), validation is done if validation is set to true in the INPUT section. The left ventricles of the validation data set is segmented and the Dice, Accuracy, Sensitivity and Specificity scores are calculated and saved in their own lists.

As last, in section seven (line 252) the ejection fraction for each validation patient is calculated twice. Once based upon the segmenting done by the model and once based upon the ground truth manually made by experts.
