import pandas as pd 
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from io import StringIO #to read data file names
import os #File navigation
import h5py

st.set_page_config(page_title="Ocular Disease Recognition",
        page_icon=":alien:",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items=None)


# Using classes can help with data security as it doesn't leak data with global variables
# that can be picked up in APIs and log tracking/tampering
class DataListObject():
    """
    Data object class for holding the User's information, also locks it in for when tabs 
    are changed.
    """
    def __init__(self, user_data=[]):
        """
        The constructor for DataObject class
        
        :param listdf: pandas dataframe object, defaults to None
        :type listdf: pandas.core.frame.DataFrame, optional
        """
        self.user_data = user_data

class Classifer():
    def predict(user_data):
        classifier_model = tf.keras.models.load_mode(r'C:\Users\David\OneDrive\Documents\GitHub\Capstone_Project\Fully_Trained_Model\CNN.h5')
        shape = ((128,128,3))
        tf.keras.Sequential*(hub[hub.KerasLayer(classifier_model, input_shape=shape)])
        #test_image = image.resize((256,256))
        preprocessing.image.img_to_array(user_data)
        user_data = user_data/255.0
        user_data = np.expand_dis(user_data, axis = 0)
        class_names = ['AMD',
                    'Cataract',
                    'Diabetes',
                    'Glaucoma',
                    'Hypertension',
                    'Myopia',
                    'Normal']
        predictions = model.predict(user_data)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        predict = 'The image uploaded is: {}'.format(image_class)
        return predict

# Interface class        
class Interface():
    """
    Interface class contains a file picker and a side bar. It also handles the import of a data object.
    """
    def __init__(self):
        """
        The constructor for Interface class.
        """
        pass
    
    def side_bar(self, user_data, model):
        """
        Sidebar configuration and file picker
        """
        upload = st.sidebar.file_uploader("Choose one file", type = ['jpg','png','jpeg'])
        if upload is not None:
            user_data = Image.open(upload)
            resize = ImageOps.contain(user_data, (128,128), method=3)
            figure = plt.figure()
            plt.imshow(resize)
            plt.axis('off')
            pred = model.predict(resize)
            st.write(pred)
            st.pyplot(figure)
          
            """
            # This is for when the UI is ready for multiple user images at once and will return a csv file.
            img_dir = st.sidebar.file_uploader("Upload",accept_multiple_files=True,type="jpg")
            if img_dir is not None: #file uploader selected a file      
                user_data = Image.open(img_dir)
                for uploaded_file in image_directory:
                    imgdir = .read() # Super stealthy pass!
            """ 
        tabmenu = ['Home','Patient Analysis','Explort CSV','Explore the Model']
        navigation = st.sidebar.selectbox(label="Menu Selection", options=tabmenu)

        # Tab Collection for side bar navigation

        if navigation == 'Home':
            with st.container():
                home_tab.home()
        elif navigation == None:
            with st.container():
                st.header("Ocular Disease Recognition")
                st.subheader("Welcome")
                st.write("""
                    This is a web-based app that is designed to help professionals within the medical field by
                    analyze singular or mass datasets of ocular fundus images, which are photos that peer inside 
                    the eye and displays the optical nerves. These images are then classified into 7 categories of
                    diagnosis.
                """)
             
                st.subheader("Beginning the Process")
                st.write("""
                - Be sure of the quality of the data you are providing. This model has some requirements for its use.
                    **Requirements:**
                        1. Make sure to provide a patient ID number at the start of each image file that ends with an _left.jpg or _right.jpg
                        2. It helps to have your fundus images trimmed to minimize runtime, with in the GitHub 
                        lick below there is a Datascrub.ipynb that can show you how.
                
                -       Upload your fundus images file into the top left corner of the sidebar, you are also provided some in a download link below.
             
                """)
             
                st.subheader("Classification Types")
                st.markdown("""
                This is the list of classifications within the csv file, following this exact order.
                **(0) Normal:** A Normal Fundus
                **(1) Diabetes:** Symptoms are mild and moderate non profliferative retinopathy
                **(2) Glaucoma:**
                **(3) Cataract:** The image is very blurred.
                **(4) Age-related Macular Degeneration (AMD):**
                **(5) Hypertension:**
                **(6) Pathological Myopia:** This condition normally inflicts both eyes
                """)
                
    st.write("")
    st.subheader("Test Samples for Users")
    st.markdown("""
                Here you can select and download an image file for user interface testing
                """)
    text_contents = '''
        Foo, Bar
        123, 456
        789, 000
        '''

    # Different ways to use the API

    st.download_button('Download CSV', text_contents, 'text/csv')
    st.download_button('Download CSV', text_contents)  # Defaults to 'text/plain'

    with open('myfile.csv') as f:
        st.download_button('Download CSV', f)  # Defaults to 'text/plain'

    # ---
    # Binary files

    binary_contents = b'whatever'

    # Different ways to use the API

    st.download_button('Download file', binary_contents)  # Defaults to 'application/octet-stream'

    with open('myfile.zip','rb') as f:
        st.download_button('Download Zip', f, file_name='archive.zip')  # Defaults to 'application/octet-stream'

    # You can also grab the return value of the button,
    # just like with any other button.

    if st.download_button(...):
        st.write('Thanks for downloading!')
                
    st.subheader("For more indepth information on this project take a look at the GitHub Source Code()")

    if os.path.isfile("user_data//user_data.jpg"):
        os.remove("user_data//*_left.jpg")
        os.remove("user_data//*_right.jpg")
                
def main():
    """
    Main and its Streamlit configuration
    """
    # Instantiating classes for security by utiliizing non-gobal variables      
    ui = Interface()
    user_data = DataListObject()
    model = Classifer()
    # Passing the an empty list between classes for discrete variable transfer and storage
    ui.side_bar(user_data,model)
    
    
# Run Main
if __name__ == '__main__':

    main()