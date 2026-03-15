import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv(r"C:\Users\paviy\htmlprgs\customer_shopping_daata.csv")
print(" First 5 rows of dataset:")
print(df.head())
print("\n Missing values per column:")
print(df.isnull().sum())
df['CustomerID'] = df['CustomerID'].fillna('Unknown')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.drop_duplicates(inplace=True)
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
df['TotalAmount'] = df['Quantity'] * df['Price']
print("\nSummary Statistics:")
print(df.describe())
plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")
ax=sns.countplot(data=df,x='Category',hue='Category', order=df['Category'].value_counts().index, palette='Set2', legend=False)
ax.set_title("Purchases by Product Category", fontsize=14, fontweight='bold')
ax.set_xlabel("Category")
ax.set_ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
df['Month'] = df['InvoiceDate'].dt.month
monthly_sales = df.groupby('Month')['TotalAmount'].sum()
plt.figure(figsize=(8, 5))
monthly_sales.plot(kind='bar', color='teal', edgecolor='black')
plt.title("Monthly Sales Trend", fontsize=14, fontweight='bold')
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun','Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
sns.barplot( data=df, x='Category', y='TotalAmount', hue='Gender', estimator=np.mean, palette='coolwarm', errorbar=None)
plt.title("Average Spending by Gender Across Categories", fontsize=14, fontweight='bold')
plt.xlabel("Category")
plt.ylabel("Average Spending")
plt.xticks(rotation=30)
plt.legend(title='Gender')
X = df[['Quantity']]
y = df['TotalAmount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred = lin_model.predict(X_test)
threshold = df['TotalAmount'].median()
y_test_class = (y_test > threshold).astype(int)
y_pred_class = (y_pred > threshold).astype(int)
cm_lin = confusion_matrix(y_test_class, y_pred_class)
disp_lin = ConfusionMatrixDisplay(confusion_matrix=cm_lin, display_labels=["Low Value", "High Value"])
disp_lin.plot(cmap='Blues')
plt.title("Confusion Matrix - Linear Regression (Categorized)", fontsize=14, fontweight='bold')
plt.show()
acc_lin = accuracy_score(y_test_class, y_pred_class)
print(f"\nLinear Regression Accuracy (as classification): {acc_lin*100:.2f}%")
df['HighValue'] = (df['TotalAmount'] > df['TotalAmount'].median()).astype(int)
X_cls = df[['Quantity', 'Price']]
y_cls = df['HighValue']
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")
print(f"Random Forest Accuracy: {acc_rf*100:.2f}%")
cm_log = confusion_matrix(y_test, y_pred_log)
cm_rf = confusion_matrix(y_test, y_pred_rf)
disp_log = ConfusionMatrixDisplay(confusion_matrix=cm_log, display_labels=["Low Value", "High Value"])
disp_log.plot(cmap='Oranges')
plt.title("Confusion Matrix - Logistic Regression", fontsize=14, fontweight='bold')
plt.show()
disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=["Low Value", "High Value"])
disp_rf.plot(cmap='Greens')
plt.title("Confusion Matrix - Random Forest", fontsize=14, fontweight='bold')
plt.show()
sns.set(style="whitegrid")
models = ["Logistic Regression", "Random Forest", "Linear Regression"]
accuracies = [acc_log * 100, acc_rf * 100, acc_lin * 100]
plt.figure(figsize=(7, 5))
ax = sns.barplot(x=models, y=accuracies, hue=models, palette="viridis", legend=False)
for i, acc in enumerate(accuracies):
    ax.text(i, acc + 1, f"{acc:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='semibold')
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()









