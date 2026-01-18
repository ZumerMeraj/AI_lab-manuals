# Import libraries
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

# SVM
svm = SVC(kernel='rbf')
svm.fit(X_train, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test))

print("Random Forest Accuracy:", rf_acc)
print("SVM Accuracy:", svm_acc)

if rf_acc > svm_acc:
    print("✅ Random Forest performs better.")
else:
    print("✅ SVM performs better.")
