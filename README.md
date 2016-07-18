# Background
The objective of this project is to quantify the marginal effect of filters on age prediction of full-frontal facial images. Filters are used commonly within applications ([Instagram](https://www.instagram.com), [Camera+](http://campl.us/), etc.) and modify the original image by playing with various photographic features: gamma modulation, saturation, RGB contrast, etc. I aim to use a pre-trained model for age detection of unfiltered photos to measure the bias of modified images on ages.

The project focus is a Convolutional VGG16 Neural Network.

This project has leveraged various libraries and datasets including: 
- [DEX: Deep EXpectation of apparent age from a single image](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) - a pretrained age detection neural network
@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {ICCV, ChaLearn Looking at People workshop},
  year = {2015},
  month = {December},
}
- [Instragram-like Filter API](https://github.com/acoomans/instagram-filters) - a library for Instagram-like image filters
- [CaffeToKeras Library](https://github.com/MarcBS/keras) - this repo includes tools for converting implementation in the Caffe framework to Keras
- [Chicago Face Database](http://chicagofaces.org/) - Ma, Correll, & Wittenbrink (2015). The Chicago Face Database: A Free Stimulus Set of Faces and Norming Data. Behavior Research Methods, 47, 1122-1135.
- [SC Face Database](http://www.scface.org/) - P. Tome, J. Fierrez, R. Vera-Rodriguez, D. Ramos, Identification using Face Regions:
Application and Assessment in Forensic Scenarios, Forensic Science International, Vol. 233, No. 1, pp. 75-83, 2013

# Pre-Trained Model
The pre-trained model I chose was the winner of the ChaLearn LAP 2015 Challenge winner, "DEX", a convolutional neural network designed in the VGG16 architecture. It was trained on 500K+ facial images with ages from IMDB and Wikipedia. The inputs to the model are JPEG images and outputs are probabilities for 101 classes corresponding to ages 0-100. 

[DEx]: https://github.com/cjl2183/FilterAgeChange/img/DEx_Img.png "DEx Architecture"

Initial Classification of 727 photos from the SCFace db and the Chicago Face DB resulted in 10% accuracy in correctly identifying the truncated age (as an integer).

[InitResults]: https://github.com/cjl2183/FilterAgeChange/img/InitResults.png "Initial Results"

# Neural Network Training
Due to the poor performance, I opted to train the neural network to reduce bias for ages classes 22, 24, and 28 and better predict ages in general.

# Processing Pipeline

# Results

# Conclusion
