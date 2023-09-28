import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess():
    # Load Credit Card Fraud Detection dataset
    data = pd.read_csv('data/creditcard.csv')

    # Split data in features and labels
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Split data to training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    return X_train, y_train, X_test, y_test