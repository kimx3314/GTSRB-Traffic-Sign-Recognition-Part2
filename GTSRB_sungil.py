# created by Sean Sungil Kim                  < https://github.com/kimx3314 >
# used for reading the ppm data utilizing the original data file/folder system format  (load_data)
#          converting GTSRB images to grayscale images                                 (gray_convrt)
#          obtaining average dimension size (1-D only)                                 (avg_size)
#          resizing images                                                             (resize_all)
#          reshaping images                                                            (clf_reshape)
#          converting images to edge detected images                                   (canny_edge_convrt)
#          computing histogram of gradients features                                   (hog_compute)
#          performing random under-sampling                                            (under_sample)


import numpy as np
import cv2
import glob
import time
from skimage.feature import hog



def load_data(num_classes):
    
    dir_loc_List = []
    labels = []
    
    # creating directory list to read data from each class folder
    for class_Number in range(0, num_classes):
        if class_Number < 10:
            dir_loc_List.append('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/0000' + str(class_Number) + '/*.ppm')
            labels.append(str(class_Number))
        else:
            dir_loc_List.append('GTSRB_Final_Training_Images/GTSRB/Final_Training/Images/000' + str(class_Number) + '/*.ppm')
            labels.append(str(class_Number))

    class_counter = 0
    for dir_loc in dir_loc_List:
        
        # reading from each class directory folder
        if class_counter == 0:
            filelist = glob.glob(dir_loc)
            data_GTSRB = np.array([cv2.imread(fname) for fname in filelist])
            data_class = np.full((len(data_GTSRB), 1), class_counter)
            class_counter += 1
        else:
            filelist = glob.glob(dir_loc)
            data_GTSRB = np.concatenate((data_GTSRB, np.array([cv2.imread(fname) for fname in filelist])))
            data_class = np.concatenate((data_class, (np.full((len(np.array([cv2.imread(fname) for fname in filelist])), 1), class_counter))))
            class_counter += 1

    return data_GTSRB, data_class, labels



def gray_convrt(input_data):

    # converting input images to grayscale
    data_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in input_data])

    return data_gray



def avg_size(x_data):
    
    size_List = []
    for data in x_data:
        # get the minimum pixel length (images are not squares)
        size_List.append(min(data.shape))
    
    return (int(np.average(size_List)), int(np.average(size_List)))



def resize_all(input_data, size = (50, 50)):

    # resizing input images
    data_resized = np.array([cv2.resize(img, size) for img in input_data])

    return data_resized



def clf_reshape(input_data):
    
    # image flattening, reshaping the data to the (samples, feature) matrix format
    n_samples = len(input_data)
    data_reshaped = input_data.reshape((n_samples, -1))
    
    return data_reshaped



def canny_edge_convrt(input_data):

    # converting input images to edge detected images
    data_edge = np.array([cv2.Canny(img, 100, 200) for img in input_data])
    
    return data_edge


    
def hog_compute(input_data):
    
    # computing HOG features of input images
    start_ts = time.time()
    hog_output = [hog(img, pixels_per_cell = (2, 2), visualize = True, multichannel = False) for img in input_data]
    data_hog = [hog_img for out, hog_img in hog_output]
    
    # flattening the output list
    #flat_hog = []
    #for hog_img in data_hog:
    #    flat_hog.append(np.array([pixel for img_row in hog_img for pixel in img_row]))
    #flat_hog = np.array(flat_hog)
    print("HOG feature computation runtime:", time.time()-start_ts)

    return data_hog



def under_sample(x_data, y_data):

    # obtaining class labels and their counts
    labels_arr, class_count = np.unique(y_data, return_counts = True)

    counter = 0
    
    # for each class label, performing random under-sampling using the indices
    for cls in labels_arr:
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        cls_idx = get_indexes(cls, y_data)
        
        # obtaining all image instances for the class label "cls"
        reshaped_subset = x_data[cls_idx]
        class_subset = y_data[cls_idx]
        
        # obtaining the indices for the purpose of random sampling without replacement
        idx = np.random.choice(np.arange(len(reshaped_subset)), min(class_count), replace = False)
        
        if counter == 0:
            # applying the randomly sampled indices on the subsets
            x_und_smpl_data = reshaped_subset[idx]
            y_und_smpl_data = class_subset[idx]
            counter += 1
        else:
            # applying the randomly sampled indices on the subsets
            x_und_smpl_data = np.concatenate((x_und_smpl_data, reshaped_subset[idx]))
            y_und_smpl_data = np.concatenate((y_und_smpl_data, class_subset[idx]))
            counter += 1

    return x_und_smpl_data, y_und_smpl_data




        
        
