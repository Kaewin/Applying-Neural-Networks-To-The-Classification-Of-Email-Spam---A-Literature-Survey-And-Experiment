import streamlit as st

# Create a text box
name = st.text_input("What is your name?")

# Create a button
button = st.button("Click me!")

# If the button is clicked, print the user's name
if button:
    st.write(f"Hello, {name}!")
