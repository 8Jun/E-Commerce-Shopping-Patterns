# E-Commerce Shopping Patterns

## Contributors
* Alexander Qaddourah
* Junji Wiener
* Teddi Li
* Alex McLaughlin

## E-commerce shopping patterns analysis using the CRISP-DM framework:
* Business Understanding
* Data Understanding
* Data Prep
* Modeling
* Evaluation
* Deployment (Did not deploy model but can use Flask and Heroku to do so.)

## Objective
This project was started in the Spring of my Master's in Business Analytics program at CU Boulder. This project explores a data set using millions of rows from an unnamed E-Commerce website. Using R, the objective is to see if buying patterns can be found.

## Data Set
For this project, the dataset we used can be found [here.](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store) The dataset was unfortunately too large to load into this repository. 

## Overview
For this project, my group and I wrote a thorough report which I have uploaded into the repository. I will hit on the main highlights here in this markdown, but I urge you to read our report and code for further explanations. 

The primary goal of this project was to use Machine Learning methods in R and/or Python to help explain purchase behavior using an E-Commerce website.

## Code Preview
Libraries used: SKlearn, Pandas, Numpy, google.colab, matplotlib
``` python
for classifier in classifiers:
    clf.set_params(clf=classifier).fit(X_train, y_train)
    classifier_name = classifier.__class__.__name__
    print(str(classifier))

    y_score = clf.predict_proba(X_test)[:,1]

    y_pred = clf.predict(X_test)
    
    roc_auc = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_things.append((fpr, tpr, '{} AUC: {:.3f}'.format(classifier_name, roc_auc)))
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall, precision)
    precision_recall_things.append((recall, precision, thresholds, '{} AUC: {:.3f}'.format(classifier_name, pr_auc)))
    #plot_precision_recall_curve(clf, X_test, y_test)
    
    feature_names = get_transformer_feature_names(clf.named_steps['preprocessor'])
    
    try:
      importances = classifier.feature_importances_
      indices = np.argsort(importances)[::-1]
      print('~Feature Ranking:')

      for f in range (X_test.shape[1]):
        print ('{}. {} {} ({:.3f})'.format(f + 1, feature_names[indices[f]], indices[f], importances[indices[f]]))
    except:

      pass

    print('~Model Score: %.3f' % clf.score(X_test, y_test))

    scores = cross_val_score(clf, X, y, cv=5)
    print('~Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))

    print('~Confusion Matrix:''\n',
    confusion_matrix(y_test, y_pred))
    print('~Classification Report:''\n',
    classification_report(y_test, y_pred,labels=np.unique(y_pred)))
   
    print('~Average Precision Score: {:.3f}'.format(average_precision_score(y_test, y_score)))
    print('~roc_auc_score: {:.3f}'.format(roc_auc))
    print('~precision-recall AUC: {:.3f}'.format(pr_auc))
    print()
```

## [Link to Powerpoint Presentation](https://github.com/8Jun/E-Commerce-Shopping-Patterns/blob/master/eCommerce%20-%20Final-1.pdf) 
![alt text](https://github.com/8Jun/E-Commerce-Shopping-Patterns/blob/master/ppt%20image.png)
