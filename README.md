# DLWORKSHOP2

## Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch

🖥️ Environment Setup

Python version: 3.9+
PyTorch version: 2.x
CUDA: Enabled if GPU available

🖼️ Data Preprocessing

Resize images to 224x224
Apply data augmentation for training: random rotation, horizontal flip
Normalize using ImageNet mean and std

🧠 Model

Pre-trained model: ResNet18 (ImageNet weights)
Freeze convolutional layers
Replace the fully connected layer

⚙️ Training

Loss function: CrossEntropyLoss
Optimizer: Adam, learning rate = 0.001
Epochs: 10

Batch size: 16

📊 Evaluation & Visualization

Compute test accuracy on the test dataset
Plot CrossEntropyLoss vs Epochs for training and validation
Plot confusion matrix for performance analysis
Use predict_and_show(image_path) for single-image prediction
Supports real images outside the dataset

Requirements
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
Pillow>=10.0.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
numpy>=1.25.0


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb


Prediction:

Place your images in any folder, e.g., Downloads/.
The notebook has a function predict_and_show(image_path) to display the image with predicted label.
Pre-trained ResNet18 ensures good predictions even with small datasets
Notes

Training on dummy images does not give meaningful results.

Predictions are correct for real images due to pre-trained ResNet18 weights.

The notebook includes GPU check and runs faster with CUDA enabled.


