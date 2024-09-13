import numpy as np

with open("float16.bin", "rb") as f:
    buf = f.read()
    d = np.ndarray((2, ), dtype=np.float16, buffer=buf)
print(d)
