from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_tree(X, y):
    X = X.reshape(len(X), -1)  # Flatten images
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    print("Decision Tree Report:")
    print(classification_report(y_test, model.predict(X_test)))
    
    return model
