import streamlit as st 
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle

features_list = np.array(pickle.load(open('D:\CNN Project\embedding.pkl','rb')))
filenames = pickle.load(open('D:\CNN Project\\name.pkl','rb'))

detector = MTCNN()
model = VGGFace(model = 'resnet50',include_top = False,input_shape=(224,224,3),pooling='avg')
st.title('Which bollywood celebrity are you')

upload_img = st.file_uploader('Choose an Image')

def save_upload(upload_img):
    try:
        with open(os.path.join('upload',upload_img.name),'wb') as f:
            f.write(upload_img.getbuffer())
            return True
    except :
        return False

def extract_feature(img_path,model,detector):
    img = cv2.imread(img_path)
    result = detector.detect_faces(img)
    x,y,width,height = result[0]['box']
    face = img[y:y+height,x:x+width]

    #Extract image feature
    image = Image.fromarray(face)
    image = image.resize((224,224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expanded_image = np.expand_dims(face_array,axis = 0)
    preprocess_img = preprocess_input(expanded_image)
    result = model.predict(preprocess_img).flatten()
    return result

def recommend(features_list,feature):
    similarity =[] # ex = [0.22,.33,.32]
    for i in range(len(features_list)):
        similarity.append(cosine_similarity(feature.reshape(1,-1),features_list[i].reshape(1,-1))[0][0])
    index_position  = sorted(list(enumerate(similarity)),reverse=True,key= lambda x:x[1])[0][0]
    return index_position

if upload_img is not None:
    # save image in upload dir
    if save_upload(upload_img):
        # load img
        display_image = Image.open(upload_img)        
        # extract feature
        features = extract_feature(os.path.join('upload',upload_img.name),model,detector)
        # recommend
        index_pos = recommend(features_list,features)
        #st.image(filenames[index_pos],width = 400)
        # display
        col1,col2 = st.columns(2)
        with col1:
            st.header('Your uploaded image')
            st.image(display_image)

        with col2:
            st.header('Bollywood celebrity you looklike')
            st.image(filenames[index_pos],width = 400)



