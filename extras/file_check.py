import os
import numpy as np

path = input("Enter path : ")
if not os.path.exists(path):
	print("exists")
else:
	print("doesnt exist")


print(path.split('/')[-1])
