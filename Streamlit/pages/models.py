import streamlit as st
from streamlit_option_menu import option_menu
import sys
from utils import white_bg
import io
from tensorflow.keras.applications import InceptionV3

st.title('Modeling')

st.markdown('LeukoVision leverages state-of-the-art convolutional neural networks (CNNs) ' \
'to classify different types of blood cells. These models—InceptionV3, ResNet50, and VGG16—have ' \
'been widely used in medical image analysis due to their ability to capture subtle patterns in ' \
'microscopy images.')

section = option_menu(
    menu_title=None,
    options=["InceptionV3", "ResNet50", "VGG16"],  
    icons=['🔬','🧬','🧪'],
    orientation="horizontal",
)

# section = st.sidebar.radio(
#     "Choose Section",
#     ["InceptionV3", "ResNet50", "VGG16"]
# )

if section == "InceptionV3":
    st.header("InceptionV3 🔬")
    st.image(white_bg('./Streamlit/pages/images/inceptionv3.png'), caption='Architecture diagram of InceptionV3',use_container_width=True)
    model = InceptionV3(weights='imagenet')
    with st.expander("See Full Model Summary"):
        stream = io.StringIO()
        model.summary(print_fn=lambda x: stream.write(x + "\n"))
        summary_string = stream.getvalue()
        st.text(summary_string)

elif section == "ResNet50":
    st.header("ResNet50 🧬")
    

elif section == "VGG16":
    st.header("VGG16 🧪")
