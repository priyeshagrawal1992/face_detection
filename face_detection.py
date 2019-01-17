#################################################################################################
# Run Command- python face_dectection.py
#

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import os, time


# load the face embeddings
#print("Loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
#print("Encoding Labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])
#print(labels)

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("\nModel Training Started\n")
recognizer = SVC(C=1.5, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()


print("Done training of Model, Starting face identification\n")
time.sleep(2.0)
os.system("python final.py")