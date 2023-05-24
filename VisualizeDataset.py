import numpy as np
import pickle

#
# This file is not needed for the final application, however the exploration of data is very important
#

test_eeg = np.load('/Dataset/Honours/Dataset/SEED/output/SEED-V/EEG_DE_features/1_123.npz')

data = pickle.loads(test_eeg['data'])
label = pickle.loads(test_eeg['label'])

label_dict = {0: 'Disgust', 1: 'Fear', 2: 'Sad', 3: 'Neutral',
              4: 'Happy'}  # hard-coded obtained from dataset documentation

print(data)
for key in data:
    print(key)
    print('label', label[key].shape, label_dict[label[key][0]], label[key][0])
    print('data', data[key].shape)
    for echo in data[key]:
        print("    ", np.array([echo]).shape)
