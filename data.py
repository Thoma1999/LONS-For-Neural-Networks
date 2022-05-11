from sklearn.datasets import load_iris
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_iris_data(seed):
	iris_data = load_iris() # load the iris dataset
	x = iris_data.data
	y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

	# One Hot encode the class labels
	encoder = OneHotEncoder(sparse=False)
	y = encoder.fit_transform(y_)

	# Split the data for training and testing
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=seed)

	scaler_x = preprocessing.MinMaxScaler(feature_range=(-1, +1)).fit(x_train)
	x_train = scaler_x.transform(x_train)
	x_test = scaler_x.transform(x_test)

	return x_train, x_test, y_train, y_test