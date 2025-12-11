
import numpy as np

data = [{'a': 1}, {'a': 2}]
np.save('test.npy', data)

loaded = np.load('test.npy', allow_pickle=True)
print(f"Type: {type(loaded)}")
print(f"Shape: {loaded.shape}")
try:
    print([x['a'] for x in loaded])
except:
    print("Direct iteration failed")
    
# If 0-d
# print([x['a'] for x in loaded.item()])
