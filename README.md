# Identify Target Customers

### (Unsupervised Learning)

## Introduction
The real-life data in this project concerns a company that performs mail-order sales in Germany, provided by AZ Direct and Arvato Finance Solution.

The goal of this project is to identify facets of the population that are most likely to be purchasers of their products for a mailout campaign. Unsupervised learning techniques are used to organize the general population into clusters for checking which of them comprise the main user base for the company.

There are a few steps of data cleaning before the machine learning techniques have been applied.

### Data

(All of the datasets are deleted after the project has been completed due to CONFIDENTIAL reasons.)

The main data for this project consist of two files:

- `Udacity_AZDIAS_Subset.csv`: demographics data for the general population of
      Germany; 891211 persons (rows) x 85 features (columns)
- `Udacity_CUSTOMERS_Subset.csv`: demographics data for customers of a mail-order
      company; 191652 persons (rows) x 85 features (columns)

The **columns** in the *general* demographics file and *customers* data file are the
**same**.

Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their **household**, **building**, and **neighborhood**.

### Features
There are another two documents explaining features:

- `AZDIAS_Feature_Summary.csv`: Summary of  demographics data
       features; 85 features (rows) x 4 columns

- `Data_Dictionary.md`: Detailed information file about the
       features

(Note: Only some important codes are included here. To find all the codes, please see `.ipynb` document.)


## Installation
Python 3, Google Colab / Jupyter Notebook

```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from random import shuffle
```

## Step 1: Preprocessing

### Step 1.1 Missing Values
- All the missing values were converted to **NaNs**.
- **Columns** with more than 20% missing values were removed.
- **Rows** with more than 10 missing values were removed.

### Step 1.2: Select and Re-Encode Features

The unsupervised learning techniques only work on data that is encoded **numerically**.

- **Numeric** and **interval** features were kept without changes.
- Most of the variables in the dataset are **ordinal** in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature. Thus, they were kept without any changes).
- **Categorical** data was encoded the levels as dummy variables. Depending on the number of categories, one of the following was performed:
 - Binary (**two-level**) categoricals that take numeric values were kept without any changes.
 - Binary variables that take on** non-numeric** value were **re-encoded** as numbers (0 and 1).
 - For **multi-level** categoricals (three or more values), some were re-encoded as multiple **dummy** variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), while some with repetitive information of other features were removed.
- **Mixed** data (a feature inclueds more than one variables) were separated to some categorical variables, and then converted into dummy variables, or dropped.


## Step 2: Feature Transformation

### Step 2.1: Apply Feature Scaling

Feature scaling helps the principal component vectors are not influenced by the natural differences in scale for features.

1.  **Temporarily remove** data points with **missing values** to compute the scaling parameters, as Sklearn requires that data not have missing values.

2. **Normalising**: Perform [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) on the data WITHOUT missing-values (data from step1), to scale each feature to mean 0 and standard deviation 1.

3. [Impute](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Imputer.html) missing values of the original data by mode, as the majority of the features are categorical or ordinal.


4. Apply transformation that was previously fit  (scaler from step2) on the data with all data points and imputed values (data from step 3).


```
def transform_data(df, cols):
  '''
  impute Nan,
  Transform by StandardScaler: [0, 1]
  '''
  #1 remove all data points with missing values
  df_not_null = df.dropna()

  #2 perform feature scaling on the data WITHOUT missing-value data points
  ##  scaler.fit(data_without_missing_values)
  scaler = StandardScaler()
  model = scaler.fit(df_not_null)

  #3 impute missing values of the original data
  # Impute Nan by mode
  imputer = Imputer(strategy = 'most_frequent', axis = 0)
  data = imputer.fit_transform(df)

  #4 apply the transformation that was previously fit
  #  on the data with all data points and imputed values
  ##  scaler.transform(imputed_data)
  data = model.transform(data)

  # Convert data to df
  df = pd.DataFrame(data, columns = cols)

  return df, scaler, imputer
```


### Step 2.2: Perform Dimensionality Reduction

- Apply [principal component analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) on the data, thus finding the vectors of maximal variance in the data.

```
def do_pca(n_components, data):
    '''
    Transforms data using PCA to create n_components,
    and provides back the results of the transformation.

    INPUT: n_components - int - the number of principal components to create
           data - the data you would like to transform

    OUTPUT: pca - the pca object created after fitting the data
            X_pca - the transformed X matrix with new number of components
    '''
    #X = StandardScaler().fit_transform(data)

    pca = PCA(n_components)
    X_pca = pca.fit_transform(data)
    return pca, X_pca

pca, X_pca = do_pca(df.shape[1], df)
```

```
def calculate_pca_p_of_var(pca):
  '''
  Calculate (cumulative) percentage of variance

  Return
  1. p - percentage of variance of a single component
  2. cumulative_p - cumulative percentage of variance

  '''

  # calculate percentage of variance captured by each principal component
  p = pca.explained_variance_ratio_.tolist()

  cumulative_p = []

  for i in range(1, len(p)+1):
    p_sum = np.sum(p[:i])

    cumulative_p.append(p_sum)

  return p, cumulative_p

single_p, cumulative_p = calculate_pca_p_of_var(pca)
```

![cumulative_p_100](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/1.png)
![cumulative_p_100](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/2.png)

- Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Then 69 transformed features were selected, which explains 95% variance.
```
for i, p in enumerate(cumulative_p):
  if p >= 0.95:
    print(f'When there is {i+1} components, the cumulative percentage of variance reaches 95%, which is {round(p*100, 2)}.')
    break
```
OUTPUT: When there is 67 components, the cumulative percentage of variance reaches 95%, which is 95.09.

- Re-fit a PCA instance to perform the decided-on transformation.

```
pca, X_pca = do_pca(67, df)

X_pca = pd.DataFrame(X_pca)
X_pca.shape
```
OUTPUT: (719624, 67)

![cumulative_p_95](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/3.png)


### Step 2.3: Interpret Principal Components

- Check out the **weight** of each variable on the first few components

```
def create_weight_df(pca, cols):
  '''
  To build a dataframe containing weights & variance of pca

  cols - column names of df (general population)
  '''

  weights = pca.components_.tolist()
  p_variance = pca.explained_variance_ratio_.tolist()

  df = pd.DataFrame(weights, columns = cols)
  df['p_variance'] = p_variance

  return df
```

```
def get_top5_feat(df, i, n):
  '''
  To get sorted list of feature weights
  of a principle component.

  df - weight dataframe

  i - the [i] th principle component

  n - number of features

  '''

  negative = df.loc[i-1].sort_values()[:n]
  positive = df.loc[i-1].sort_values(ascending = False)[:n]

  variance = round(df.loc[i-1]['p_variance'], 4)

  return negative, positive, variance
```

#### Interpreting Principle Component #1

![Weight of 10 variables of Component 1](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/4.png)


`HH_EINKOMMEN_SCORE` with positive weight is on the opposite side of `FINANZ_MINIMALIST` with negative weight. When`HH_EINKOMMEN_SCORE` becomes lower, while `FINANZ_MINIMALIST`score higher, which means this person has higher income and lower financial interest.

 - `HH_EINKOMMEN_SCORE`: Estimated household net income (1-highest, 6-lowest)
 - `FINANZ_MINIMALIST`:  low financial interest (1-higher, 5-lower)

Both `LP_STATUS_GROB_1.0 ` and `HH_EINKOMMEN_SCORE` have positive weight. Thus, they vary in a same direction - when  `LP_STATUS_GROB_1.0 ` increases, `HH_EINKOMMEN_SCORE` will increases. When a person is rich, he/she will have a lower `HH_EINKOMMEN_SCORE` and low `LP_STATUS_GROB_1.0` score.  

 - `LP_STATUS_GROB_1.0` - low income earner (1- yes,  0 - no)
 - `HH_EINKOMMEN_SCORE`:  Estimated household net income (1-highest, 6-lowest)

The principle component #1 will increase, when the features with positive weights increase, or the features with negaive weights decrease. However, as the absolute value of all the features is less than 0.3, the correlation between the component and these features is very weak. Thus, as the scores of features change, the component might just have little change.


## Step 3: Clustering

### Step 3.1: Apply Clustering to General Population

Apply **K-Means clustering** to the dataset, and use the **average** within-cluster **distances** from each point to their assigned cluster's centroid to decide on a number of clusters to keep.

- Perform [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) clustering on the **PCA-transformed data**.

- Then, compute the **average difference** from each point to its assigned cluster's center by KMeans object's `.score()` method.

```
def get_kmeans_score(data, center):
    '''
    returns the kmeans score regarding SSE for points to centers
    INPUT:
        data - the dataset you want to fit kmeans to
        center - the number of centers you want (the k value)
    OUTPUT:
        score - the SSE score for the kmeans model fit to the data
    '''
    #instantiate kmeans
    kmeans = KMeans(n_clusters = center)

    # Then fit the model to your data using the fit method
    model = kmeans.fit(data)

    # Obtain a score related to the model fit
    score = np.abs(model.score(data))

    return score
```

- Perform the above two steps for a number of **different cluster counts**.

```
scores = []
centers = list(range(1,30,2))

for center in centers:
  print(center)
  score = get_kmeans_score(X_pca, center)
  scores.append(score)
```

![k-means centroids](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/5.png)


-  **Re-fit** a KMeans instance with the final chosen centroid number to perform the clustering operation.

```
kmeans = KMeans(n_clusters = 5)
general_predict = kmeans.fit_predict(X_pca)
```

### Step 3.2: Apply All Steps to the Customer Data

- Clean the customer data (assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)

- Use the sklearn objects from the general demographics data, and apply their **transformations** to the customers data -  feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

```
#Impute Nan by mode
df = imputer.transform(df)

#Apply feature scaling: StandardScaler()
df = scaler.transform(df)

#transform the customers data using pca object
df = pca.transform(df)

customer_predict = kmeans.predict(df)
```


### Step 3.3: Compare Customer Data to Demographics Data

Consider the proportion of persons in each cluster for the general population, and the proportions for the customers:

- If the company's customer base to be **universal**, then the cluster assignment proportions should be fairly **similar** between the two.

- If there is a **higher proportion** of persons in a cluster for the **customer** data compared to the general population, then that suggests the people in that cluster to be a **target** audience for the company (see Cluster 0 and 4)

- On the other hand, the proportion of the data in a cluster being **larger** in the **general** population than the customer data suggests that group of persons to be** outside of the target **demographics (see cluster 1, 2, 3).

![comparison](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/6.png)

Which cluster or clusters are **overrepresented** in the customer dataset compared to the general population?

 - use the `.inverse_transform()` method of the **PCA** objects to transform centroids back to the **scaled data** to interpret the values.

```
def inverse_cluster(i, cols, kmeans):
  '''
  To apply the inverse transform through the PCA transform,
  to get the scaled scores (not back to the original data)
  of all columns

  INPUT: i - cluster No.
         cols - all of column names of cleaned data
         kmeans - fitted kmeans object from step3.1

  OUTPUT: a df including scalced scores & column names(index)
  '''
  # inverse transform to scalered data
  cluster = kmeans.cluster_centers_[i]
  scaled_score = pca.inverse_transform(cluster)

  # to dataframe
  df = pd.DataFrame({'Scaled Score': scaled_score}, index = cols)

  return df
```

Since StandardScaler was used on the scores, we know that  
- strongly negative values suggest lower values on the original scale, and
- positive values suggest higher values on the original scale.


From the bar plot in step 3.31, We can cleary see that Cluster 0 and 4 are over-represented in the customer data compared to the general population.

![cluster0](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/7.png)

The following kinds of people are part of **Cluster 0**:
1. Low financial interest (FINANZ_MINIMALIST)
- low movement (MOBI_REGIO)
- more  buildings in the microcell (KBA05_GBZ)
- higher share of 1-2 family homes in the PLZ8 region(PLZ8_ANTG1)
- higher share of 1-2 family homes in the microcell (KBA05_ANTG1)
- higher income (HH_EINKOMMEN_SCORE)
- not low income earners(LP_STATUS_GROB_1.0)
- lower share of 10+ family homes in the PLZ8 region(PLZ8_ANTG4)
- lower share of 6-10 family homes in the PLZ8 region (PLZ8_ANTG3)
- smaller size of community (ORTSGR_KLS9)



![cluster4](https://github.com/yanglinjing/dsnd_p3_identify_customer_segments/blob/master/readme_img/8.png)

The following kinds of people are part of **Cluster 4**:

1. Lower financial interest (FINANZ_MINIMALIST)
- higher share of 1-2 family homes in the microcell (KBA05_ANTG1)
- lower movement(MOBI_REGIO)
- more buildings in the microcell(KBA05_GBZ)
- higher share of 1-2 family homes  in the PLZ8 region (PLZ8_ANTG1 )
- smaller size of community (ORTSGR_KLS9)
- lower share of 6-10 family homes  in the PLZ8 region (PLZ8_ANTG3 )
- lower or no share of 10+ family homes  in the PLZ8 region (PLZ8_ANTG4)
- not low income earners(LP_STATUS_GROB_1.0)
- higher income(HH_EINKOMMEN_SCORE)

We can find that the features of people in Cluster 0 and 4 are same. Thus, our** target customers** are this kind of persons:

- They might have lower financial interests, lower movement, and higher income.
- They might live in smaller size of community
- They might have higher share of 1-2 family homes in the microcell which have more buildings.

- In the PLZ8 region, they might have
 - higher share of 1-2 family homes,
 - lower share of 6-10 family homes, and
 - lower or no share of 10+ family homes
