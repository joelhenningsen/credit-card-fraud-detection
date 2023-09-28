from data_preprocessing import load_and_preprocess
from model import build_model

if __name__ == "__main__":
    # Load and preprocess data
    X_train, y_train, X_test, y_test = load_and_preprocess()

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=64,
                        validation_data=(X_test, y_test))
    
    # Save the trained model
    model.save('models/fraud_detection_model.h5')