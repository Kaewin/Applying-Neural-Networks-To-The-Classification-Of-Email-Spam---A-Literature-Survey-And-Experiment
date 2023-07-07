import streamlit as st
import tensorflow as tf

def save_model(model, filepath):
    tf.saved_model.save(model, filepath)

def load_model(filepath):
    model = tf.saved_model.load(filepath)

    return model

if __name__ == '__main__':

    filepath = 'model.savedmodel'

    model = load_model(filepath)

    st.write("The model has been loaded.")

    st.write("What would you like to do?")

    options = ["Make a prediction", "View the model's parameters"]

    choice = st.selectbox("Select an option", options)

    if choice == "Make a prediction":
        # Get the user's input
        features = st.slider("Enter the features", 0, 1, step=0.01)

        # Make a prediction
        prediction = model.predict([features])[0]

        # Display the prediction
        st.write(f"The prediction is: {prediction}")

    elif choice == "View the model's parameters":
        # View the model's parameters
        for name, value in model.get_config().items():
            st.write(f"{name}: {value}")

    else:
        pass
