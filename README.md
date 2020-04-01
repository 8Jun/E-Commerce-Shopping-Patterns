# E-Commerce Shopping Patterns

## Executive Summary:
In this project, my team and I wanted to practice our Machine Learning toolset in Python. We obtained an extremely large dataset from Kaggle containing millions of rows of user event data. Our team needed to cut down on the number of rows to narrow our scope for the project. We decided to look at user events for only smartphones, which gave us a much more managable dataset.

After subsetting on smartphones, we feature engineered several columns to have more predictors. First, we took each user event (view, cart, purchase) and created a binary column for each type. Next, we took the timestamp column and subsetted on only Saturdays, as this day had the most transactions in the dataset when compared to other days of the week. We binned the hour of the day into "Early Morning", "Morning", "Afternoon", and "Evening". Finally, we added some frequency variables in our data. First, the number of unique product IDs each user looked at. We then added two other features, the number of "cart" events and the number of "view" events per user ID per product ID.

Our target variable for this project was to predict if the "Purchase" event column would be a 1 or 0. Ultimately, will a user purchase a said project given the newly created variables in the dataset? Given our data and our target variable (categorical, not continuous), the models our team decided on were: Gradient Boosting (XGBoost), Logistic Regression, and Neural Network.

Main takeaways for our project:
* Our team should have leveraged a big data cluster to help with computing power. When loading data in, our computers had a tough time processing the millions of rows of data.
* Another problem our team faced was the dataset was not the best for predictions. It was structured as a descriptive dataset about purchase behavior and should have been limited to just descriptive statistics.
* The XGBoost model was one of the best models out of the three. Our top three features from this model were the number of times a user "viewed" a product, followed by the number of times the user "carted" the product. The third most important feature was the number of total unique products viewed by the user. We concluded that these results made sense for our business problem because a user's typical purchase journey follows a similar path. Most users spend more time viewing different products in the same category to compare, and then will place their final one or two products inside the cart. When a user increases the number products in their consideration set, the probability of a purchase will increase.

![Screen Shot 2020-03-31 at 3 44 24 PM](https://user-images.githubusercontent.com/56977428/78078413-60bc4b80-7367-11ea-93ad-126721ed9671.png)

### Contributors:
* Alexander Qaddourah
* Junji Wiener
* Teddi Li
* Alex McLaughlin

### E-commerce shopping patterns analysis using the CRISP-DM framework:
* Business Understanding
* Data Understanding
* Data Prep
* Modeling
* Evaluation
* Deployment (Did not deploy model but can use Flask and Heroku to do so.)

### Objective:
* This project was started in the Spring of my Master's in Business Analytics program at CU Boulder. This project explores a data set using millions of rows from an unnamed E-Commerce website. Using R, the objective is to see if buying patterns can be found.

### Data Set:
* For this project, the dataset we used can be found [here.](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store) The dataset was unfortunately too large to load into this repository. 

### Overview:
* For this project, my group and I wrote a thorough report which I have uploaded into the repository. I will hit on the main highlights here in this markdown, but I urge you to read our report and code for further explanations.

### Languages used:
* Python - Used for the actual ML pipeline and some feature engineering.
* R - Used for data cleaning and a portion of the preprocessing activities.

### Project Goal:
* The primary goal of this project was to use Machine Learning methods in R and/or Python to help explain purchase behavior using an E-Commerce website.

### Code Preview:
* Libraries used: SKlearn, Pandas, Numpy, google.colab, matplotlib
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

### [Link to Powerpoint Presentation](https://github.com/8Jun/E-Commerce-Shopping-Patterns/blob/master/eCommerce%20-%20Final-1.pdf)
![alt text](https://github.com/8Jun/E-Commerce-Shopping-Patterns/blob/master/ppt%20preview%20png/ppt%20image2.png)
