#To use the complete slices instead of patches:
#I define the loading from output list step as a function:
def load_train_data(output):
    EDframes = []
    EDground = [] 
    ESframes = []
    ESground = []

    for i in range(len(output)):
        EDframes.append(output[i][2])
        EDground.append(output[i][3])
        ESframes.append(output[i][4])
        ESground.append(output[i][5])

    # Take the ES frames and ED frames 
    frames = ESframes+EDframes
    groundtruth = ESground+EDground

    frames=np.array(frames)
    groundtruth=np.array(groundtruth)
    return frames, groundtruth

#I create a function to divide into train set and test set:
def loading_data(data):
    
    s=len(data)
    t=int(0.75*s)
    arr=random.sample(data,s)
    imgs_train, imgs_mask_train = load_train_data(arr[:t])
    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 3.  # scale masks to [0, 1]
    imgs_mask_train[imgs_mask_train<=0.8]=0
    
    imgs_test, imgs_id_test = load_train_data(arr[t:])
    imgs_id_test = imgs_id_test.astype('float32')
    imgs_id_test /= 3.
    imgs_test = imgs_test.astype('float32')
    imgs_id_test[imgs_id_test<=0.8]=0
    
    return imgs_train, imgs_mask_train, imgs_test, imgs_id_test
    
#Therefore, the sets are created from 'output' just using the function:
imgs_train, imgs_mask_train, imgs_test, imgs_id_test=loading_data(output)

#Building the model with input size= (128, 128, 1):
cnn  = fcn_model((128,128,1),2,weights=None)

#To compile it:
hist=cnn.fit(imgs_train[:,:,:,np.newaxis], imgs_mask_train[:,:,:,np.newaxis],batch_size=5, epochs=10, verbose=1, shuffle=True,
              validation_split=0.25)
#Here the hist.history() object tracks the loss and accuracy during training

#And the predictions are very straight forward too:
imgs_mask_test = cnn.predict(imgs_test[:,:,:,np.newaxis], verbose=1)

#So imgs_mask_test are the predictions and imgs_id_test their ground truths
#As the prediction have the channels dimension (3th dimension per slice), to go back to 2 dimensions per slice:
imgs_mask_test=np.squeeze(imgs_mask_test)
