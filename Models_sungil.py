# created by Sean Sungil Kim                  < https://github.com/kimx3314 >
# used for comparing different classifiers with default parameters (CV)            (compare_clf)
#          capturing x% of explained variance ratios of PCA                        (capture_var_PCA)
#          comparing PCA solvers to capture lowest number of principal components  (compare_PCA)
#          comparing image thresholding methods                                    (compare_thresh)
#          performing wrapper based feature selection using input classifier       (wrapper_based_fs)


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time



def compare_clf(data_np, target_np, cv, rf = 1, gb = 1, ab = 1, svm = 1, mlp = 1, xgb = 1):

    # accuracy list
    acc_Dict = {}
    
    # setup cross-validation classifier scorers
    scorers = {'Accuracy': 'accuracy'}

    print('Comparing Different Classifiers:\n-----------------------------------------------------------------')
    
    # sci-kit Random Forest
    if rf == 1:
        start_ts = time.time()
        clf = RandomForestClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                           
        print("Random Forest Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))   
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Random Forest Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]
    
    # sci-kit Gradient Boosting
    if gb == 1:
        start_ts = time.time()
        clf = GradientBoostingClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                           
        print("Gradient Boosting Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Gradient Boosting AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Gradient Boosting Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    # sci-kit Ada Boosting
    if ab == 1:
        start_ts = time.time()
        clf = AdaBoostClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                      
        print("Ada Boost Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Ada Boost AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Ada Boost Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    # sci-kit SVM
    if svm == 1:
        start_ts = time.time()
        clf = SVC()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                       
        print("SVM Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("SVM Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['SVM Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    # sci-kit Neural Network
    if mlp == 1:
        start_ts = time.time()
        clf = MLPClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                       
        print("Multi-layer Perceptron Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Multi-layer Perceptron Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Multi-layer Perceptron Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]
    
    # XGBoost
    if xgb == 1:
        start_ts = time.time()
        clf = XGBClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                       
        print("XGBoost Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("XGBoost Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['XGBoost Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    print('The classifier (with default parameters) with the highest accuracy is %s\nThe cross-validation accuracy is %0.3f (+/- %0.2f)'\
          % (max(acc_Dict, key = acc_Dict.get), acc_Dict[max(acc_Dict, key = acc_Dict.get)][0], \
             acc_Dict[max(acc_Dict, key = acc_Dict.get)][1]))



def compare_fin_clf(data_np, target_np, cv, rf = 1, svm = 1, mlp = 1):

    # accuracy list
    acc_Dict = {}
    
    # setup cross-validation classifier scorers
    scorers = {'Accuracy': 'accuracy'}

    print('Comparing Different Classifiers:\n-----------------------------------------------------------------')
    
    # sci-kit Random Forest
    if rf == 1:
        start_ts = time.time()
        clf = RandomForestClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                           
        print("Random Forest Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))   
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Random Forest AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Random Forest Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    # sci-kit SVM
    if svm == 1:
        start_ts = time.time()
        clf = SVC()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                       
        print("SVM Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("SVM Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['SVM Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    # sci-kit Neural Network
    if mlp == 1:
        start_ts = time.time()
        clf = MLPClassifier()
        scores = cross_validate(clf, data_np, target_np, scoring = scorers, cv = cv)
        scores_Acc = scores['test_Accuracy']
                                                                                                                                       
        print("Multi-layer Perceptron Classifier Acc: %0.3f (+/- %0.2f)" % (scores_Acc.mean(), scores_Acc.std() * 2))
        #scores_AUC= scores['test_roc_auc']     #Only works with binary classes, not multiclass
        #print("Multi-layer Perceptron Classifier AUC: %0.2f (+/- %0.2f)" % (scores_AUC.mean(), scores_AUC.std() * 2))
        print("CV Runtime:", time.time()-start_ts, '\n-----------------------------------------------------------------')
        acc_Dict['Multi-layer Perceptron Classifier'] = [scores_Acc.mean(), scores_Acc.std() * 2]

    print('The classifier (with default parameters) with the highest accuracy is %s\nThe cross-validation accuracy is %0.3f (+/- %0.2f)'\
          % (max(acc_Dict, key = acc_Dict.get), acc_Dict[max(acc_Dict, key = acc_Dict.get)][0], \
             acc_Dict[max(acc_Dict, key = acc_Dict.get)][1]))



def compare_thresh(img):
    
    fig = plt.figure(figsize = (15, 3))
    
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Adaptive Mean Thresholding
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Adaptive Gaussian Thresholding
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding
    ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret5, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', \
              "Otsu's Thresholding", "Otsu's Thresholding w/ Gaussian Filtering"]
    images = [th1, th2, th3, th4, th5]
    
    # plotting 5 different thresholding methods
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



def wrapper_based_fs(data_x, data_y, clf):
    
    feat_start = 0
    
    # to select only based on max_features, set to integer value and set threshold = -np.inf
    sel = SelectFromModel(clf, prefit = False, threshold = 'mean', max_features = None)  
    fit_mod = sel.fit(data_x, data_y)    
    sel_idx = fit_mod.get_support()

    header = []
    for i in range(len(data_x[0])):
        header.append([i])

    # getting lists of selected and non-selected features (names and indexes)
    temp = []
    temp_idx = []
    temp_del = []
    for i in range(len(data_x[0])):
        if sel_idx[i] == 1:                           # selected features and their indexes are added to the temp and temp_idx list
            temp.append(header[i + feat_start])
            temp_idx.append(i)
        else:                                         # indexes of non-selected features are added to the temp_del list
            temp_del.append(i)

    print ('Wrapper Select:\n')
    print('Selected:', temp)
    print('Features (total/selected): %i / %i' % (len(data_x[0]), len(temp)))

    header = header[0:feat_start]
    for field in temp:
        header.append(field)
    fs_data = np.delete(data_x, temp_del, axis = 1)          # deleting non-selected features
    
    return fs_data
    
    
    