'''
Classification with keras
Input data (.csv format) should be seperated by ":" and last column called "class"
'''
# import libraries
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from PyQt5.QtWidgets import *
from sklearn.metrics import accuracy_score
from keras.models import load_model

# Define the NN model
def create_model(n_predictors):
    # Sequential model
    model = Sequential()
    # Add an input layer
    model.add(Dense(500, activation='relu', input_dim=n_predictors))  
    # Add hidden layers
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # Add an output layer
    model.add(Dense(2, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define the function to preprocess the input data
def preprocess_data(data):
    # Read the data into a Pandas dataframe
    df = pd.read_csv(data, delimiter=';')
    # Normalize the data by dividing each column (except 'class') by its maximum value
    predictors = list(set(list(df.columns))-set(['class']))
    df[predictors] = df[predictors]/df[predictors].max()
    # Extract the predictor variables
    X = df[predictors].values
    # Value of input_dim variable
    n_predictors = X.shape[1]
    # One-hot encode the target variable
    y = to_categorical(df['class'].values)
    return X, y, n_predictors


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Neural Network Classification'
        self.left = 100
        self.top = 100
        self.width = 450
        self.height = 250
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create a label to show the training accuracy
        self.training_label = QLabel(self)
        self.training_label.setText('Training Accuracy:')
        self.training_label.move(150, 25)

        # Create a label to show the test accuracy
        self.test_label = QLabel(self)
        self.test_label.setText('Test Accuracy:')
        self.test_label.move(150, 200)

        # Create a button to train the neural network
        self.train_button = QPushButton('Train', self)
        self.train_button.setToolTip('Train the neural network')
        self.train_button.move(25, 25)
        self.train_button.clicked.connect(lambda: self.train_model('trained_model.h5'))

        # Create a button to test the neural network
        self.test_button = QPushButton('Test', self)
        self.test_button.setToolTip('Test the neural network')
        self.test_button.move(25, 200)
        self.test_button.clicked.connect(self.test_model)

        # Create a label to show the input file path
        self.file_label = QLabel(self)
        self.file_label.setText('Input File:')
        self.file_label.move(150, 150)

        # Create a line edit to input the file path
        self.file_edit = QLineEdit(self)
        self.file_edit.move(250, 150)
        self.file_edit.resize(150, 25)

        # Create a button to select the input file
        self.file_button = QPushButton('Browse', self)
        self.file_button.setToolTip('Select the input file')
        self.file_button.move(25, 150)
        self.file_button.clicked.connect(self.browse_file)

        # Set the width of the training and test labels
        self.training_label.setFixedWidth(200)
        self.test_label.setFixedWidth(200)

        # Create a horizontal line
        self.line = QFrame(self)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)
        self.line.setGeometry(0, 100, 450, 5)

    def train_model(self, model_file):
        # Load the training data
        data_file = 'training_data.csv'
        X, y, n_predictors = preprocess_data(data_file)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
        # Create the neural network model
        model = create_model(n_predictors)
        # Train the model
        model.fit(X_train, y_train, epochs=20, verbose=1)
        # Save the trained model to the file
        model.save(model_file)
        # Evaluate the model on the training set
        scores = model.evaluate(X_train, y_train, verbose=0)
        # Show the training accuracy
        self.training_label.setText('Training Accuracy: {}%'.format(round(scores[1]*100, 2)))
        # Return the trained model
        return model

    def test_model(self):
        # Get the input file path
        file_path = self.file_edit.text()
        # Define the model file
        model_file = 'trained_model.h5'
        # Load the trained model from the file
        trained_model = load_model(model_file)
        # Load the test data
        X_test, y_test, n_predictors = preprocess_data(file_path)
        # Predict the outputs of the test data using the trained model
        y_pred = trained_model.predict(X_test)
        # Compute the binary accuracy of the predictions
        binary_accuracy = accuracy_score(y_test, y_pred.round())
        # Show the test accuracy
        self.test_label.setText('Test Accuracy: {}%'.format(round(binary_accuracy*100, 2)))


    def browse_file(self):
        # Open a file dialog to select the test data file
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, 'Select Test Data', '', 'CSV Files (*.csv)')
        # Insert the selected file path into the line edit widget
        self.file_edit.setText(file_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())