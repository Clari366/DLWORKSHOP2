# DLWORKSHOP2

## Building an AI Classifier: Identifying Cats, Dogs & Pandas with PyTorch
Overview

This project uses PyTorch and transfer learning to classify images of cats, dogs, and pandas.
It leverages a pre-trained ResNet18 model to predict the class of any input image and display it with the predicted label.

The model can correctly identify:

Dogs (dog.jpg) → dog

Pandas (panda.png) → panda

Cats (any clear cat image) → cat

Features

Uses pre-trained ResNet18 for feature extraction.

Maps ImageNet predictions to 3 simplified classes: cat, dog, panda.

Displays the input image along with the predicted label.

GPU support with CUDA for faster inference.

Confusion matrix and accuracy reporting included.

Requirements
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
Pillow>=10.0.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
numpy>=1.25.0

Folder Structure
data/
    train/
        cat/
        dog/
        panda/
    test/
        cat/
        dog/
        panda/
notebooks/
    cat_dog_panda_transfer_learning.ipynb
requirements.txt
README.md

Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb


Prediction:

Place your images in any folder, e.g., Downloads/.

The notebook has a function predict_and_show(image_path) to display the image with predicted label.

predict_and_show("C:/Users/admin/Downloads/dog.jpg")     # outputs: dog
predict_and_show("C:/Users/admin/Downloads/panda.png")   # outputs: panda

Notes

Training on dummy images does not give meaningful results.

Predictions are correct for real images due to pre-trained ResNet18 weights.

The notebook includes GPU check and runs faster with CUDA enabled.

This README is simple, clear, and fully explains the project for GitHub.

If you want, I can also make a slightly enhanced version with badges, overview image, and code snippet preview so it looks professional on GitHub.

Do you want me to do that?

You said:
give me as a single text
ChatGPT said:

Sure! Here’s the entire README as a single text you can directly put in your GitHub repository:

Cat-Dog-Panda Image Classification using Pre-trained ResNet18
Overview

This project uses PyTorch and transfer learning to classify images of cats, dogs, and pandas. It leverages a pre-trained ResNet18 model to predict the class of any input image and display it with the predicted label. The model can correctly identify:

Dogs (dog.jpg) → dog

Pandas (panda.png) → panda

Cats (any clear cat image) → cat

Features

Uses pre-trained ResNet18 for feature extraction.

Maps ImageNet predictions to 3 simplified classes: cat, dog, panda.

Displays the input image along with the predicted label.

GPU support with CUDA for faster inference.

Confusion matrix and accuracy reporting included.

Requirements
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0
Pillow>=10.0.0
matplotlib>=3.8.0
scikit-learn>=1.3.0
numpy>=1.25.0

Folder Structure
data/
    train/
        cat/
        dog/
        panda/
    test/
        cat/
        dog/
        panda/
notebooks/
    cat_dog_panda_transfer_learning.ipynb
requirements.txt
README.md

Usage

Clone the repository:

git clone <your-repo-link>
cd <your-repo-folder>


Install dependencies:

pip install -r requirements.txt


Run the Jupyter Notebook:

jupyter notebook notebooks/cat_dog_panda_transfer_learning.ipynb


Prediction:

Place your images in any folder, e.g., Downloads/.

The notebook has a function predict_and_show(image_path) to display the image with predicted label:

predict_and_show("C:/Users/admin/Downloads/dog.jpg")    
predict_and_show("C:/Users/admin/Downloads/panda.png") 
