# Background
The objective of this project is to quantify the marginal effect of filters on age prediction of full-frontal facial images. Filters are used commonly within applications (Instagram, Camera+, etc.) and modify the original image by playing with gamma modulation, saturation, RGB contrast, and other photo features. I aim to use a pre-trained model for age detection of unfiltered photos (https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) to measure the bias of modified images on ages.

The model focus of this project is a Convolutional VGG16 Neural Network.

This project has leveraged various libraries and datasets including: 
- DEX: Deep EXpectation of apparent age from a single image - a pretrained age detection neural network
- https://github.com/acoomans/instagram-filters - a library for Instagram-like image filters
- https://github.com/MarcBS/keras - this repo includes tools for converting implementation in the Caffe framework to Keras
- Chicago Face Database (http://chicagofaces.org/) - Ma, Correll, & Wittenbrink (2015). The Chicago Face Database: A Free Stimulus Set of Faces and Norming Data. Behavior Research Methods, 47, 1122-1135.
- SC Face Database (http://www.scface.org/) - P. Tome, J. Fierrez, R. Vera-Rodriguez, D. Ramos, Identification using Face Regions:
Application and Assessment in Forensic Scenarios, Forensic Science International, Vol. 233, No. 1, pp. 75-83, 2013

# Pre-Trained Model
The LAP Face Challenge winner is DEX - a convolutional neural network designed in the VGG16 architecture. It was trained on 500K+ facial images with ages from IMDB and Wikipedia. The inputs to the model are JPEG images and outputs are probabilities for 101 classes corresponding to ages 0-100. 

Class

# Neural Network Training

# Processing Pipeline

# Results

# Conclusion
