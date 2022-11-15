from tensorflow import keras
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.utils import load_img, img_to_array 
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image

features_list  = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('name.pkl','rb'))

# Detact face from image
model = VGGFace(model = 'resnet50',include_top = False,input_shape=(224,224,3),pooling='avg')
detactor = MTCNN()
sample_img = cv2.imread('D:\CNN Project\Bollywood_celeb_face_localized\sample\saif_ali_khan_dupli.jpg')
result = detactor.detect_faces(sample_img)
x,y,width,height = result[0]['box']
face = sample_img[y:y+height,x:x+width]

#Extract image feature
image = Image.fromarray(face)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
expanded_image = np.expand_dims(face_array,axis = 0)
preprocess_img = preprocess_input(expanded_image)
result = model.predict(preprocess_img).flatten()


# find similarity
similarity =[] # ex = [0.22,.33,.32]
for i in range(len(features_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),features_list[i].reshape(1,-1))[0][0])
index_position  = sorted(list(enumerate(similarity)),reverse=True,key= lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index_position])
cv2.imshow('output',temp_img)
cv2.waitKey(0)