# Customer Segmentation Analysis

## 📌 Project Overview

Customer Segmentation Analysis is a data analytics and machine learning project that groups customers based on their purchasing behaviour.

The project uses **RFM Analysis** — Recency, Frequency, and Monetary Value — to understand customer behaviour and identify meaningful customer segments.

After calculating RFM metrics, **K-Means Clustering** is applied to divide customers into different groups. These customer segments can then be used to create targeted marketing strategies and improve customer engagement.

---

## 🎯 Objective

The main objectives of this project are:

* Analyze customer purchasing behaviour.
* Calculate Recency, Frequency, and Monetary values.
* Identify different types of customers.
* Standardize customer behaviour features.
* Determine the optimal number of clusters using the Elbow Method.
* Apply K-Means Clustering.
* Visualize customer segments.
* Profile each customer cluster.
* Suggest suitable marketing strategies for each customer segment.

---

## 🛠️ Technology Stack

* **Python**
* **Jupyter Notebook**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**

### Machine Learning Algorithm

* K-Means Clustering

### Data Analysis Technique

* RFM Analysis

---

## 📂 Project Structure

```text
DataAnalytics-L1-CustomerSegmentation/
│
├── dataset/
│   └── Online Retail.xlsx
│
├── images/
│   ├── rfm_distribution.png
│   ├── elbow_method.png
│   ├── cluster_distribution.png
│   ├── customer_segments.png
│   └── cluster_analysis.png
│
├── notebook/
│   └── Customer_Segmentation.ipynb
│
└── README.md
```

---

## 📊 Dataset

The project uses the **Online Retail Dataset**.

The dataset contains transaction-level information about customer purchases.

### Important Columns

| Column      | Description                       |
| ----------- | --------------------------------- |
| InvoiceNo   | Unique invoice/transaction number |
| StockCode   | Product code                      |
| Description | Product description               |
| Quantity    | Number of products purchased      |
| InvoiceDate | Date and time of transaction      |
| UnitPrice   | Price per unit                    |
| CustomerID  | Unique customer identifier        |
| Country     | Customer's country                |

---

## 🔍 Project Workflow

The project follows these major steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Data Cleaning
   ↓
RFM Analysis
   ↓
Feature Selection
   ↓
Data Standardization
   ↓
Elbow Method
   ↓
K-Means Clustering
   ↓
Cluster Visualization
   ↓
Customer Profiling
   ↓
Marketing Recommendations
```

---

# 1. Import Required Libraries

The following Python libraries are used for data analysis, visualization, and machine learning.

```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

---

# 2. Load the Dataset

The Online Retail dataset is loaded into a Pandas DataFrame.

```python
df = pd.read_excel("../dataset/Online Retail.xlsx")

df.head()
```

---

# 3. Understand the Dataset

Basic information about the dataset is examined.

```python
df.shape
```

```python
df.info()
```

```python
df.describe()
```

The first few rows are also displayed to understand the structure of the data.

```python
df.head()
```

---

# 4. Check Missing Values

Missing values are checked before performing the analysis.

```python
df.isnull().sum()
```

Since `CustomerID` is required for customer-level segmentation, records without a CustomerID can be removed.

```python
df = df.dropna(subset=['CustomerID'])
```

Check again:

```python
df.isnull().sum()
```

---

# 5. Data Cleaning

Duplicate records are checked.

```python
df.duplicated().sum()
```

Duplicates can be removed:

```python
df = df.drop_duplicates()
```

Cancelled transactions can also be removed where required.

```python
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
```

Only valid transactions are retained for customer segmentation.

---

# 6. Create Total Amount

A new column called `TotalAmount` is created.

```python
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
```

Check the result:

```python
df[['Quantity', 'UnitPrice', 'TotalAmount']].head()
```

---

# 7. RFM Analysis

RFM stands for:

### Recency

How recently a customer made a purchase.

### Frequency

How frequently a customer makes purchases.

### Monetary

How much money a customer spends.

These three metrics help understand customer behaviour.

---

# 8. Calculate Recency

First, convert `InvoiceDate` into datetime format.

```python
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
```

Set the analysis date.

```python
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
```

Calculate Recency:

```python
recency = df.groupby('CustomerID')['InvoiceDate'].max().reset_index()

recency['Recency'] = (
    analysis_date - recency['InvoiceDate']
).dt.days

recency = recency[['CustomerID', 'Recency']]
```

---

# 9. Calculate Frequency

Frequency represents the number of transactions made by each customer.

```python
frequency = (
    df.groupby('CustomerID')['InvoiceNo']
    .nunique()
    .reset_index()
)

frequency.columns = ['CustomerID', 'Frequency']
```

---

# 10. Calculate Monetary Value

Monetary value represents the total amount spent by each customer.

```python
monetary = (
    df.groupby('CustomerID')['TotalAmount']
    .sum()
    .reset_index()
)

monetary.columns = ['CustomerID', 'Monetary']
```

---

# 11. Create RFM Table

Merge the three RFM metrics.

```python
rfm = recency.merge(
    frequency,
    on='CustomerID'
).merge(
    monetary,
    on='CustomerID'
)
```

Display the RFM table:

```python
rfm.head()
```

Check its structure:

```python
rfm.info()
```

---

# 12. RFM Data Visualization

Visualize the distribution of RFM variables.

```python
plt.figure(figsize=(8, 5))

sns.histplot(rfm['Recency'], bins=30, kde=True)

plt.title('Recency Distribution')
plt.xlabel('Recency')
plt.ylabel('Number of Customers')

plt.show()
```

Frequency:

```python
plt.figure(figsize=(8, 5))

sns.histplot(rfm['Frequency'], bins=30, kde=True)

plt.title('Frequency Distribution')
plt.xlabel('Frequency')
plt.ylabel('Number of Customers')

plt.show()
```

Monetary:

```python
plt.figure(figsize=(8, 5))

sns.histplot(rfm['Monetary'], bins=30, kde=True)

plt.title('Monetary Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('Number of Customers')

plt.show()
```

---

# 13. Select Features for Clustering

The RFM variables are selected for customer segmentation.

```python
features = rfm[['Recency', 'Frequency', 'Monetary']]
```

Check the selected data:

```python
features.head()
```

---

# 14. Standardize the Data

RFM variables may have different scales.

Therefore, `StandardScaler` is used.

```python
scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(features)
```

Convert the result back into a DataFrame:

```python
rfm_scaled = pd.DataFrame(
    rfm_scaled,
    columns=['Recency', 'Frequency', 'Monetary']
)

rfm_scaled.head()
```

---

# 15. Find Optimal Number of Clusters

The **Elbow Method** is used to determine a suitable number of clusters.

```python
inertia = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    
    kmeans.fit(rfm_scaled)
    
    inertia.append(kmeans.inertia_)
```

Plot the Elbow Method:

```python
plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker='o'
)

plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')

plt.show()
```

The point where the decrease in inertia starts slowing down can be selected as the optimal number of clusters.

---

# 16. Apply K-Means Clustering

After selecting the appropriate number of clusters, K-Means is applied.

Example:

```python
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
```

Check the result:

```python
rfm.head()
```

---

# 17. Count Customers in Each Cluster

```python
cluster_counts = rfm['Cluster'].value_counts().sort_index()

cluster_counts
```

---

# 18. Customer Count by Cluster

```python
plt.figure(figsize=(8, 5))

sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values
)

plt.title('Number of Customers in Each Cluster')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')

plt.show()
```

---

# 19. Cluster Profiling

Calculate the average RFM values for each customer segment.

```python
cluster_profile = (
    rfm.groupby('Cluster')
    [['Recency', 'Frequency', 'Monetary']]
    .mean()
    .round(2)
)

cluster_profile
```

This table helps us understand how each customer segment behaves.

---

# 20. Customer Segment Visualization

A scatter plot can be used to visualize customer segments.

```python
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=rfm,
    x='Frequency',
    y='Monetary',
    hue='Cluster',
    palette='viridis',
    s=80
)

plt.title('Customer Segments')
plt.xlabel('Frequency')
plt.ylabel('Monetary Value')

plt.show()
```

---

# 21. Recency vs Monetary Visualization

```python
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=rfm,
    x='Recency',
    y='Monetary',
    hue='Cluster',
    palette='viridis',
    s=80
)

plt.title('Customer Segments: Recency vs Monetary')
plt.xlabel('Recency')
plt.ylabel('Monetary Value')

plt.show()
```

---

# 22. Frequency vs Recency Visualization

```python
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=rfm,
    x='Recency',
    y='Frequency',
    hue='Cluster',
    palette='viridis',
    s=80
)

plt.title('Customer Segments: Recency vs Frequency')
plt.xlabel('Recency')
plt.ylabel('Frequency')

plt.show()
```

---

# 23. Customer Segment Interpretation

The clusters can be interpreted according to their RFM characteristics.

### 🏆 High-Value / Loyal Customers

Characteristics:

* Low Recency
* High Frequency
* High Monetary value

Marketing strategy:

* Loyalty rewards
* Exclusive offers
* Premium products
* VIP programs

### 💰 Potential Loyal Customers

Characteristics:

* Relatively recent purchases
* Moderate frequency
* Moderate/high spending

Marketing strategy:

* Membership programs
* Personalized recommendations
* Cross-selling

### ⚠️ At-Risk Customers

Characteristics:

* High Recency
* Previously active customers
* Moderate/high spending

Marketing strategy:

* Win-back campaigns
* Discount offers
* Personalized emails

### 💤 Low-Value / Inactive Customers

Characteristics:

* High Recency
* Low Frequency
* Low Monetary value

Marketing strategy:

* Low-cost promotional campaigns
* Re-engagement offers
* Targeted discounts

The exact interpretation should be based on the actual RFM averages obtained from the dataset.

---

# 24. Marketing Recommendations

Based on the customer segmentation results, businesses can develop different strategies for different customer groups.

| Customer Segment     | Suggested Strategy                        |
| -------------------- | ----------------------------------------- |
| High-Value Customers | VIP rewards and exclusive offers          |
| Loyal Customers      | Loyalty programs and personalized offers  |
| Potential Customers  | Cross-selling and product recommendations |
| At-Risk Customers    | Win-back campaigns                        |
| Low-Value Customers  | Re-engagement campaigns                   |

---

# 📈 Key Insights

The analysis helps identify:

* Customers who purchase frequently.
* Customers who spend the most.
* Recently active customers.
* Customers who may be at risk of leaving.
* Different customer groups based on purchasing behaviour.
* Appropriate marketing strategies for each segment.

---

# 💡 Business Applications

Customer segmentation can be used in:

* Targeted marketing
* Customer retention
* Loyalty programs
* Personalized recommendations
* Email marketing
* Discount campaigns
* Customer relationship management
* E-commerce analytics

---

# ✅ Conclusion

This project demonstrates how data analytics and machine learning can be used to understand customer purchasing behaviour.

RFM analysis provides useful customer-level features, while K-Means clustering groups customers with similar behaviours.

The resulting customer segments can help businesses make data-driven marketing decisions, improve customer retention, and develop personalized marketing strategies.

---

# 🚀 Future Improvements

The project can be extended by:

* Testing different clustering algorithms.
* Comparing K-Means with Hierarchical Clustering.
* Using PCA for dimensionality reduction.
* Creating an interactive dashboard using Power BI or Tableau.
* Building an automated customer segmentation pipeline.
* Using customer segments for recommendation systems.
* Applying advanced machine learning techniques.

---

# 📚 Skills Demonstrated

This project demonstrates knowledge of:

* Python
* Pandas
* NumPy
* Data Cleaning
* Exploratory Data Analysis
* Data Visualization
* Customer Segmentation
* Business Analytics
* Marketing Analytics


---

# 👨‍💻 Author

**Aniket Kolekar**

Data Analytics Intern

GitHub: **ANIKET7796473**

---

## ⭐ Project Summary

**Project:** Customer Segmentation Analysis

**Domain:** Data Analytics / Machine Learning

**Dataset:** Online Retail Dataset

**Tools:** Python, Pandas, NumPy, Matplotlib, Seaborn, Jupyter Notebook
