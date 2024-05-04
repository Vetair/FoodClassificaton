import os
from PIL import Image
import torch
import json
import torch.nn.functional as F
from torchvision import transforms
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from models import EnsembleModel, MobileNetV2Model, InceptionV3Model


# Set environment variables
os.environ['QT_ICC_PROFILE'] = 'sRGB'

class Model:
	def __init__(self, modeldir):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.modeldir = modeldir
		self.class_indict = self.read_class_json()
		self.model_ft = None
        
		self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
		self.load_weights()
    
	def read_class_json(self):
        # Read class indices
		with open("./food_classes.json", 'r') as json_file:
			class_indict = json.load(json_file)
		return class_indict

	def load_weights(self):
		self.model_ft = torch.load(self.modeldir, map_location=self.device)
		print('Loaded weights')
		self.model_ft = self.model_ft.to(self.device)
		self.model_ft.eval()

	def single_image_loader(self, loader, image_name):
        # Load image
		img = Image.open(image_name).convert('RGB')
		img = loader(img)
		img = torch.unsqueeze(img, dim=0)
		img = img.to(self.device)
		return img

	def predict(self, image_dir):
		with torch.no_grad():
			output = self.model_ft(self.single_image_loader(self.data_transform, image_dir))
        	# Obtain probability distribution
			probabilities = F.softmax(output, dim=1)
			# Argmax searches for the index corresponding to the maximum value
			predicted_class = torch.argmax(probabilities, dim=1)

		return probabilities, predicted_class.item()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        uic.loadUi("mainwindow.ui", self)

        self.model = None
        self.image = None
		# Initially disable the select image button
        self.selectImage.setEnabled(False)
		# Initially disable the prediction button and set it to red
        self.predict.setEnabled(False)
		# Get the imageLabel defined in the. ui file
        self.imageLabel = self.findChild(QtWidgets.QLabel, 'imageLabel')
        self.selectModel.clicked.connect(self.SelectModel)
        self.selectImage.clicked.connect(self.SelectImage)
        self.predict.clicked.connect(self.Predict)

    def SelectModel(self):
        modelname, _ = QtWidgets.QFileDialog.getOpenFileName()
        if modelname == "":
            return
        elif os.path.splitext(modelname)[1] != ".pth":
            self.outLabel.setText("Provide a valid model (.pth file)")
            return
        try:
            self.model = Model(modelname)
            if self.model.model_ft is None:
                raise Exception("Cannot load model")
            else:
                self.outLabel.setText(f"Successfully loaded model: \n{os.path.basename(modelname)}")
                self.selectImage.setEnabled(True)  # Enable the Select Image button
                self.predict.setStyleSheet("background-color: gray; font: white")  # Set as default color
        except Exception as e:
            self.outLabel.setText(str(e))

    def SelectImage(self):
        imagename, _ = QtWidgets.QFileDialog.getOpenFileName()
        if imagename == "":
            return
        elif os.path.splitext(imagename)[1].lower() not in [".jpg", ".png", ".jfif"]:
            self.outLabel.setText("Provide a valid image (.jpg or .png or .jfif file)")
            return
        self.image = imagename
        self.outLabel.setText(f"Successfully loaded image: \n{os.path.basename(imagename)}")
        if self.model is not None:
            self.predict.setEnabled(True)  # Enable prediction button
            self.predict.setStyleSheet("background-color: green; font: white")  # Turn green
			# Load and display images
            self.displayImage(imagename)
	
    def displayImage(self, image_path):
        # Create a QPixmap object and load the image
        pixmap = QtGui.QPixmap(image_path)
        # Scaling images to fit the size of QLabel
        scaled_pixmap = pixmap.scaled(self.imageLabel.size(), aspectRatioMode=QtCore.Qt.KeepAspectRatio)
        # Set the image displayed by QLabel
        self.imageLabel.setPixmap(scaled_pixmap)
        # Display image labels
        self.imageLabel.show()

    def Predict(self):
        if self.image is None:
            self.outLabel.setText("Provide a valid image")
            return
        if self.model is None:
            self.outLabel.setText("Provide a valid model")
            return
        probabilities, predicted_class = self.model.predict(self.image)
        out_str = "Class probabilities:\n"
        class_indices = [self.model.class_indict[str(i)] for i in range(len(self.model.class_indict))]
        max_length = len(max(class_indices, key=len))
        for i, prob in enumerate(probabilities[0]):
            line = f"{class_indices[i]:<{max_length}}:  {prob.item()*100:.2f}%\n"
            out_str += line
        self.outLabel.setText(out_str.strip())


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
