from sklearn.datasets import load_iris import pandas as pd iris = load_iris() iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) iris_df['target'] = iris.target print(iris_df.head(10))

num_instances = iris_df.shape[0] num_features = iris_df.shape[1] - 1 num_classes = len(iris.target_names) print(f"Number of instances: {num_instances}") print(f"Number of features: {num_features}") print(f"Number of target classes: {num_classes}") for col in iris_df.columns[:-1]: print(f"{col}: Type: {iris_df[col].dtype}, Min: {iris_df[col].min()}, Max: {iris_df[col].max()}")

import seaborn as sns import matplotlib.pyplot as plt correlation_matrix = iris_df.iloc[:, :-1].corr() sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') plt.title('Correlation Heatmap') plt.show()

from sklearn.model_selection import train_test_split X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression model = LogisticRegression() model.fit(X_train_scaled, y_train)

from sklearn.model_selection import cross_val_score best_score = 0 for c in [0.01, 0.1, 1, 10, 100]: model = LogisticRegression(C=c) scores = cross_val_score(model, X_train_scaled, y_train, cv=5) score = scores.mean() print(f"C={c}: Cross-Validation Accuracy={score}") if score > best_score: best_score = score best_c = c best_model = LogisticRegression(C=best_c) best_model.fit(X_train_scaled, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score y_pred = best_model.predict(X_test_scaled) print("Accuracy:", accuracy_score(y_test, y_pred)) print("Precision:", precision_score(y_test, y_pred, average='macro')) print("Recall:", recall_score(y_test, y_pred, average='macro')) print("F1 Score:", f1_score(y_test, y_pred, average='macro'))

