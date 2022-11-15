
#import os
#import pickle
#actors = os.listdir('D:\CNN Project\Bollywood_celeb_face_localized\Bollywood_celeb_faces0')
#filename = []
#
#for actor in actors:
#    for file in os.listdir(os.path.join('D:\CNN Project\Bollywood_celeb_face_localized\Bollywood_celeb_faces0',actor)):
#        filename.append(os.path.join('D:\CNN Project\Bollywood_celeb_face_localized\Bollywood_celeb_faces0',actor,file))
#
#
#pickle.dump(filename,open('filenames.pkl','wb'))


from tensorflow import keras
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.utils import load_img, img_to_array 
import numpy as np
import pickle
from tqdm import tqdm

filenames = pickle.load(open('name.pkl','rb'))

model = VGGFace(model = 'resnet50',include_top = False,input_shape=(224,224,3),pooling='avg')
def feature_extractor(img_path,model1):
    img = load_img(img_path,target_size = (224,224))
    img_array = img_to_array(img)
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocess_img = preprocess_input(expanded_img)
    result = model1.predict(preprocess_img).flatten()
    return result

features = []
for file in tqdm(filenames):
  features.append(feature_extractor(file,model))
pickle.dump(features,open('embedding.pkl','wb'))