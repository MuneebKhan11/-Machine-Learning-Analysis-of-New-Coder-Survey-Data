import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, BaggingClassifier, BaggingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# Read the survey data
survey = pd.read_csv("/kaggle/input/2021-new-coder-survey/2021 New Coder Survey.csv", low_memory=False)

# Create a new DataFrame with custom column names
df = pd.DataFrame(columns=['Area', 'Age', 'Education_Level', 'Hours_Learning', 'Income', 'Degree', 'Interest'])
df['Area'] = survey.iloc[:, 26]
df['Age'] = survey.iloc[:, 23]
df['Education_Level'] = survey.iloc[:, 32]
df['Hours_Learning'] = survey.iloc[:, 7]
df['Income'] = survey.iloc[:, 22]
df['Degree'] = survey.iloc[:, 33]
df['Interest'] = survey.iloc[:, 35]
df['Months_Programming'] = survey.iloc[:, 8]
resources = survey['3. Which online learning resources have you found helpful? Please select all that apply.'].str.split('; ').explode()
resources_counts = resources.value_counts().sort_values(ascending=False)


# Cleaning the data
def clean_income(value):
    if isinstance(value, str):
        value = value.replace(',', '').replace('$', '').replace('Under ', '').replace('Over ', '')
        if 'to' in value:
            lower, upper = value.split(' to ')
            try:
                return (float(lower) + float(upper)) / 2
            except ValueError:
                return None
        try:
            return float(value)
        except ValueError:
            return None
    return value

# Clean the 'Income' column
df['Income'] = df['Income'].apply(clean_income)

# Handle missing and incorrect values for other columns
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Hours_Learning'] = pd.to_numeric(df['Hours_Learning'], errors='coerce')

# Clip 'Age' and 'Hours_Learning' values to make the data more realistic
df['Age'] = df['Age'].clip(upper=100)
df['Hours_Learning'] = df['Hours_Learning'].clip(upper=100)

# Fill missing values with the median value of each column
df.fillna(df.median(), inplace=True)

# EDA
print(df.describe())



# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df, diag_kind='kde')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x='Area', data=df)
plt.xticks(rotation=45)
plt.title('Number of New Coders by Area')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Area', y='Income', data=df)
plt.xticks(rotation=45)
plt.title('Income by Area')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Education_Level', y='Income', data=df)
plt.xticks(rotation=45)
plt.title('Income by Education Level')
plt.show()

# Three-variable scatter plot example
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Age', y='Hours_Learning', hue='Income', data=df)
plt.title('Age vs Hours Learning with Income as Hue')
plt.show()

# Clustering
# Prepare data for clustering
df_cluster = df.drop(columns=['Education_Level', 'Degree', 'Interest'])
df_cluster = df_cluster.dropna()

# OneHot encode the 'Area' column
encoder = OneHotEncoder()
encoded_area = encoder.fit_transform(df_cluster[['Area']]).toarray()
area_columns = encoder.get_feature_names(['Area'])
encoded_area_df = pd.DataFrame(encoded_area, columns=area_columns, index=df_cluster.index)
df_cluster = pd.concat([df_cluster.drop(columns=['Area']), encoded_area_df], axis=1)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_cluster)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Create clusters using KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
df_cluster['Cluster'] = clusters

print(df_cluster.groupby('Cluster').mean())

# Classification
# Create a binary target variable for high income
df['High_Income'] = df['Income'].apply(lambda x: 1 if x >= 30000 else 0)

# Prepare data for classification
X = df.drop(columns=['Area', 'Education_Level', 'Degree', 'Interest', 'Income', 'High_Income'])
X = pd.get_dummies(X)
X.fillna(X.median(), inplace=True)  # Fill missing values with the median value of each column
y = df['High_Income']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)





                
##### Train multiple classifiers
classifiers = [
    RandomForestClassifier(random_state=42),
    MLPClassifier(random_state=42),
    LogisticRegression(random_state=42),
    DecisionTreeClassifier(random_state=42),
    BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42), n_estimators=100, random_state=42)
]

classifier_names = [
    'Random Forest',
    'Multilayer Perceptron',
    'Logistic Regression',
    'Decision Tree',
    'Decision Tree w/ Bagging'
]

regressor_names = [
    'Random Forest Regression',
    'Linear Regression',
    'MLP Model',
    'Random Forest Regression w/ Bagging',
    'Linear Regression w/ Bagging',
    'MLP Model w/ Bagging'
]

regressors = [
    RandomForestRegressor(random_state=42),
    LinearRegression(),
    MLPRegressor(random_state=42),
    BaggingRegressor(base_estimator=RandomForestRegressor(random_state=42), n_estimators=100, random_state=42),
    BaggingRegressor(base_estimator=LinearRegression(), n_estimators=100, random_state=42),
    BaggingRegressor(base_estimator=MLPRegressor(random_state=42), n_estimators=100, random_state=42)
]

# Train and evaluate the classifiers
accuracies = []
performance_metrics = [] 
cv = 3

for clf, name in zip(classifiers, classifier_names):
    print(f"\n{name}:\n")

    # Grid search for hyperparameters
    if name == 'Random Forest':
        param_grid = {'n_estimators': [10, 50, 100, 200]}
    elif name == 'Multilayer Perceptron':
        param_grid = {'hidden_layer_sizes': [(100,), (50, 50)], 'alpha': [0.0001, 0.01]} 
    elif name == 'Logistic Regression':
        param_grid = {'C': [0.1, 1, 10, 100], 'solver': ['newton-cg', 'lbfgs', 'liblinear']}
    elif name == 'Decision Tree':
        param_grid = {'max_depth': [2, 4, 6, 8, 10]}
    elif name == 'Decision Tree w/ Bagging':
        param_grid = {'base_estimator__max_depth': [2, 4, 6, 8, 10], 'n_estimators': [10, 50, 100]}

    grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    print(classification_report(y_test, y_pred))
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    cv = 3  
    cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=cv)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation score: {np.mean(cv_scores)}")

# Collect performance metrics
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    performance_metrics.append([precision, recall, f1_score, accuracy])

# Train and evaluate the regressors
mse_values = []

for reg, name in zip(regressors, regressor_names):
    print(f"\n{name}:\n")

    # Grid search for hyperparameters
    if name == 'Random Forest Regression':
        param_grid = {'n_estimators': [10, 50, 100, 200]}
    elif name == 'Linear Regression':
        param_grid = {}
    elif name == 'MLP Model':
        param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (25, 50, 25)], 'alpha': [0.0001, 0.001, 0.01, 0.1]}
    elif name == 'Random Forest Regression w/ Bagging':
        param_grid = {'base_estimator__n_estimators': [10, 50, 100], 'n_estimators': [10, 50, 100]}
    elif name == 'Linear Regression w/ Bagging':
        param_grid = {'n_estimators': [10, 50, 100]}
    elif name == 'MLP Model w/ Bagging':
        param_grid = {'base_estimator__hidden_layer_sizes': [(100,), (50, 50)], 'base_estimator__alpha': [0.0001, 0.01], 'n_estimators': [10, 50, 100]}

    grid_search = GridSearchCV(reg, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)  
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mse_values.append(mse)
    print("Mean Squared Error:", mse)
    cv = 3  
    cv_scores = cross_val_score(grid_search.best_estimator_, X, y, cv=cv, scoring='neg_mean_squared_error')
    print(f"Cross-validation scores: {-cv_scores}")
    print(f"Mean cross-validation score: {-np.mean(cv_scores)}")

# Plot classifier performance metrics
plot_performance_metrics(classifier_names, performance_metrics, ['Precision', 'Recall', 'F1-Score', 'Accuracy'], 'Performance Metrics of Different Classifiers')

# Plot regressor performance metrics
plt.figure(figsize=(10, 5))
sns.barplot(x=regressor_names, y=mse_values)
plt.xlabel('Regressor')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error of Different Regressors')
plt.show()
resources_counts = resources.value_counts().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.barplot(x=resources_counts.values, y=resources_counts.index, palette='rocket')
plt.title('Most Used Online Learning Resources')
plt.xlabel('Count')
plt.show()
plt.figure(figsize=(10, 7))
plt.pie(resources_counts, labels=resources_counts.index, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Most Used Online Learning Resources')
plt.show()
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.boxplot(x='Hours_Learning', data=df)
plt.title('Number of Hours Spent Learning Each Week')
plt.xlabel('Hours')
plt.show()
# Boxplot for Months_Programming
plt.figure(figsize=(10, 7))
df['Months_Programming'] = pd.to_numeric(df['Months_Programming'], errors='coerce')
sns.boxplot(x='Months_Programming', data=df)
plt.title('Months Programming')
plt.xlabel('Months')
plt.show()

plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
region_counts = df['Area'].value_counts()
plt.pie(region_counts.values, labels=region_counts.index, autopct='%1.1f%%')
plt.title('Number of Developers by Region')
plt.show()
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.boxplot(y='Age', data=df)
plt.title('Age Distribution of Developers')
plt.ylabel('Age')
plt.show()
plt.figure(figsize=(10, 7))
sns.histplot(data=df, x="Age", kde=True, color='skyblue')
plt.axvline(df['Age'].mean(), color='red', label='Mean')
plt.axvline(df['Age'].median(), color='green', label='Median')
plt.legend()
plt.title('Age Distribution of Developers')
plt.xlabel('Age')
plt.show()
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.histplot(df[df['Months_Programming'] > 0]['Months_Programming'], bins=20, kde=False, alpha=0.8)
plt.title('Months Programming Distribution of Developers')
plt.xlabel('Months Programming')
plt.ylabel('Count')
plt.show()
# Create a new column with the number of years programming
df['Years_Programming'] = df['Months_Programming'] // 12

# Count the number of developers in each range of years programming
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10']
df['Years_Range'] = pd.cut(df['Years_Programming'], bins=bins, labels=labels)
counts = df['Years_Range'].value_counts()

# Calculate the percentages for each range
percentages = [(count/sum(counts))*100 for count in counts]

# Create a pie chart
plt.figure(figsize=(10, 7))
plt.pie(percentages, labels=labels, autopct='%1.1f%%')
plt.title('Percentage of Developers by Years Programming')
plt.show()
# Calculate the most frequent age value
mode_age = df['Age'].mode()[0]

# Calculate the smallest age value
min_age = df['Age'].min()

# Calculate the largest age value
max_age = df['Age'].max()

# Calculate the 4 smallest age values
smallest_ages = df['Age'].nsmallest(4)

# Calculate the 4 largest age values
largest_ages = df['Age'].nlargest(4)

# Print the results
print("Most frequent age value:", mode_age)
print("Smallest age value:", min_age)
print("Largest age value:", max_age)
print("4 smallest age values:", smallest_ages.values)
print("4 largest age values:", largest_ages.values)
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.boxplot(x='Hours_Learning', data=df)
plt.title('Number of Hours Spent Learning Each Week')
plt.xlabel('Hours')
plt.show()
# Create a new DataFrame with the filtered values
restricted_df = df[df['Hours_Learning'] <= 70]

# Draw the boxplot for hours worked learning per week with restrictions
plt.figure(figsize=(10, 7))
sns.set_style('whitegrid')
sns.boxplot(x='Hours_Learning', data=restricted_df)
plt.title('Number of Hours Spent Learning Each Week (Restricted)')
plt.xlabel('Hours')
plt.show()
# Define the ranges for hours studied per week
bins = [0, 19, 29, 39, 49, 59, 100]
labels = ['0-18', '19-28', '29-38', '39-48', '49-58', '59+']
df['Hours_Range'] = pd.cut(df['Hours_Learning'], bins=bins, labels=labels)
counts = df['Hours_Range'].value_counts()

# Calculate the percentages for each range
percentages = [(count/sum(counts))*100 for count in counts]

# Create a pie chart
plt.figure(figsize=(10, 7))
plt.pie(percentages, labels=labels, autopct='%1.1f%%')
plt.title('Percentage of Hours Studied per Week')
plt.show()

plt.figure(figsize=(10, 7))
sns.scatterplot(x='Age', y='Income', data=df, color='b', alpha=0.5, s=80)
sns.regplot(x='Age', y='Income', data=df, scatter=False, color='r')
plt.title('Age vs Income', fontsize=14)
plt.xlabel('Age', fontsize=12)
plt.ylabel('Income', fontsize=12)
plt.tick_params(labelsize=10)
plt.show()


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Extract the 'Income' column for clustering
X = df[['Income']]

# Instantiate the KMeans estimator and fit it to the data
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,11), metric='distortion', timings=False)
visualizer.fit(X)

# Display the visualizer
visualizer.show()
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Extract the 'Income' column for clustering
X = df[['Income']]

# Instantiate the KMeans estimator and fit it to the data
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 11), metric='silhouette', timings=False)
visualizer.fit(X)

# Display the visualizer
visualizer.show()
# Extract the 'Hours_Learning' column for clustering
X = df[['Hours_Learning']]

# Instantiate the KMeans estimator and fit it to the data
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# Add the predicted cluster labels to the DataFrame
df['Cluster'] = kmeans.predict(X)

# Create a scatter plot with different colors for each cluster
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Age', y='Income', data=df, hue='Cluster', palette='viridis')
plt.title('Hours Worked Learning per Week by Age and Income')
plt.show()
# Extract the 'Age' and 'Hours_Learning' columns for clustering
X = df[['Age', 'Hours_Learning']]

# Instantiate the KMeans estimator and fit it to the data
model = KMeans(n_clusters=3, random_state=42)
model.fit(X)

# Add a new column to the DataFrame with the cluster labels
df['Cluster'] = model.labels_

# Create a scatter plot of Age vs Hours_Learning, colored by cluster label
plt.figure(figsize=(10, 7))
sns.scatterplot(x='Age', y='Hours_Learning', hue='Cluster', data=df, palette='viridis')
plt.title('Age vs Hours Worked per Week (Clustered)')
plt.show()
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from yellowbrick.cluster import KElbowVisualizer


# Extract the 'Education_Level' column for clustering
X = df[['Education_Level']]

# Convert education level strings to numerical labels
le = LabelEncoder()
X['Education_Level'] = le.fit_transform(X['Education_Level'])

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate the KMeans estimator and fit it to the data
model = KMeans(n_clusters=3)
model.fit(X_scaled)

# Get the cluster labels and count of each label
labels, counts = np.unique(model.labels_, return_counts=True)

# Create a new DataFrame with the cluster labels and counts
cluster_df = pd.DataFrame({'Cluster': labels, 'Count': counts})

# Create a bar plot of the cluster counts
plt.figure(figsize=(10, 7))
sns.barplot(x='Count', y='Cluster', data=cluster_df, orient='h')
plt.xlabel('Count')
plt.ylabel('Cluster')
plt.title('Normalized Cluster Counts by Education Level')
plt.yticks(range(len(labels)), le.inverse_transform(labels))
plt.show()
# Extract the 'Region' and 'Age' columns for clustering
X = df[['Area', 'Age']]

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Instantiate the KMeans estimator and fit it to the data
model = KMeans(n_clusters=3)
model.fit(X_scaled)

# Add the cluster labels to the original DataFrame
df['Cluster'] = model.labels_

# Create a pivot table of region counts by cluster
region_counts = pd.pivot_table(df, index='Area', columns='Cluster', aggfunc='count', fill_value=0)

# Normalize the region counts by cluster
region_counts_norm = region_counts.div(region_counts.sum(axis=0), axis=1)

# Create a bar plot of the normalized region counts by cluster
plt.figure(figsize=(10, 7))
sns.barplot(data=region_counts_norm, orient='h')
plt.xlabel('Percentage of Developers')
plt.ylabel('Area')
plt.title('Normalized Region Counts by Cluster and Age')
plt.legend(labels=['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.show()
accuracy_table = pd.DataFrame({'Classifier': classifier_names, 'Accuracy': accuracies})
print(accuracy_table)
# Display F1 scores in tabular form
f1_scores = [round(f1 * 100, 2) for _, _, f1, _ in performance_metrics]

f1_score_table = pd.DataFrame({'Classifier': classifier_names, 'F1 Score (%)': f1_scores})
print("\nF1 Scores for Classifiers:")
print(f1_score_table)
