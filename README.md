# Advanced Machine Learning

### GTSRB Traffic Sign Image Multi-class Classification Part II
### Final Report Part II coming soon
using Python

by Sean Sungil Kim

This project is a follow-up of the GTSRB Traffic Sign Recognition Project Part I. The objective of this project is the same as that of Part I. In this project, another feature extraction method named Histogram of Oriented Gradients will be explored. In addition, combinations of the feature extraction and feature selection/dimensionality reduction methods are analyzed. Random Forest and SVM classifiers fit on thresheld and feature selected (wrapper-based) images demonstrated significant testing f-score performance of 0.918973 and 0.920249 respectively. These performance scores were better than that of any models analyzed in Part I. The model fitting and the hyper-tuning steps also required extremely less time compared to Part I. SVM had the fastest runtime (during both hyptertuning and model fitting phases) with the highest f-score performance. Random Forest took longer in terms of runtimes, but it was still significantly faster compared to any of the models in Part I. Random Forest had only slightly less predictive power than SVM. The utilization of combinations of feature extraction and feature selection was proven extremely critical in this project.

The dataset can be downloaded here: http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
