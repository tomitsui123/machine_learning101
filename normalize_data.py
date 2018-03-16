from sklearn.preprocessing import scale
from sklearn.cross_validation import train_test_split
from sklearn import datasets

digits = datasets.load_digits()
data = scale(digits.data)
X_train, X_test, y_train, y_test, images_train, images_test = train_test_split(data, digits.target, digits.images, test_size=0.25, random_state=42)

n_samples, n_features = X_train.shape

print(n_samples, n_features)