from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits.target)

fig = plt.figure(figsize=(4, 2))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(10):
    ax = fig.add_subplot(2, 5, i + 1, xticks = [], yticks = [])
    ax.imshow(digits.images[i], cmap = plt.cm.binary)
    ax.text(0, 7, str(digits.target[i]))
    print(digits.target[i])
plt.show()