import streamlit as st
import cv2
from Eggsy import detect_objects
import tempfile

# Set the title and background color of the web app
st.set_page_config(page_title="White and Brown Egg Counter Web App", page_icon=":egg:", layout="centered")

# Define the main function that runs the app
def main():
    st.title("White and Brown Egg Counter Web App")

    # Add a file uploader widget
    uploaded_file = st.file_uploader("Upload file")

    # Check if a file has been uploaded
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Run the object detection function and display the output
        detection_results = detect_objects(tfile.name, st.empty())
        
        # Clear the output area
        st.empty()

# Run the main function
if __name__ == "__main__":
    main()


