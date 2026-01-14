import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('data_BSL.pickle', 'rb'))

max_length = max(len(seq) for seq in data_dict['data'])

def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences

data = pad_sequences(data_dict['data'], max_length)
labels = np.asarray(data_dict['labels'])

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
score = accuracy_score(y_predict, y_test)
print(score * 100)

f = open('model_BSL.pkl', 'wb')
pickle.dump({'model': model}, f)
f.close()
