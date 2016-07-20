# Background
The objective of this project is to quantify the marginal effect of filters on age prediction of full-frontal facial images. Filters are used commonly within applications ([Instagram](https://www.instagram.com), [Camera+](http://campl.us/), etc.) and modify the original image by playing with various photographic features: gamma modulation, saturation, RGB contrast, etc. I aim to use a pre-trained model for age detection of unfiltered photos to measure the bias of modified images on ages.

The project was focused on a Convolutional VGG16 Neural Network over a time span of 2+ weeks.

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
The pre-trained model I chose was the winner of the ChaLearn LAP 2015 Challenge winner, "DEX", a convolutional neural network designed in the VGG16 architecture. It was trained on 500K+ facial images with ages from IMDB and Wikipedia. The inputs to the model are JPEG images and outputs are probabilities for 101 classes corresponding to ages 0-100. I converted the model from Caffe to Keras and ran it on an AWS instance from AMI ID: ami-125b2c72 maintained by Stanford's [CS231N Class](http://cs231n.github.io/aws-tutorial/). It's a g2.2xlarge, CUDA 7.5 and CuDNN v3 enabled instance which comes preinstalled with Caffe, Torch7, Theano, Keras and Lasagne. Python wrappers for Caffe are also installed.

![DEx](/img/DEx_Img.png "DEx Architecture")

Due to incompatibility issues between the Keras version the conversion script was designed for and the new Keras version, the model architecture had to be re-specified. Once the model architecture was defined in Keras, the original weights were loaded from an .h5 file for the predict method.

Initial Classification of 727 photos collected from the SCFace db and the Chicago Face DB resulted in 10% accuracy in correctly identifying the age (identified as the rounded down age integer).

![InitResults](https://github.com/cjl2183/FilterAgeChange/tree/master/img/InitResults.png "Initial Results")

It's very likely that the training examples are skewing the data since youth is in high demand for the entertainment industry. Additionally it's practically a job requirement for actors to enhance themselves cosmetically. Thus, training a model to detect ages from actors' faces yields suboptimal results for the average person.

# Neural Network Training
Due to the less than optimal performance, I opted to train the neural network to reduce bias for ages classes 22, 24, and 28 and  better predict ages in general.

## Processing Pipeline
- Python's [dlib](http://dlib.net/python/) library was used to identify the facial features with (x,y) coordinates.
- A rectangular boundary enclosing the face was used to crop the image with a 20% margin remaining around the face.
- Each image was flipped left-to-right and saved. This doubled the sample size while still adding information (since faces are not exactly symmetrical).
- Using [SciPy miscellaneous methods](http://docs.scipy.org/doc/scipy/reference/misc.html), images were then converted to 4-dimensional numpy arrays, resized, and rearranged from RGB to GRB color-channels

## Retraining
Ages with decimal precision were included in the photo collections for each image subject. Then for each image, the neural network was trained to target a class representing the subject's age integer floor. 

The Keras model was compiled with the following options. 
- *Optimizer*: Stochastic Gradient Descent - The final layer of the neural network outputs probabilities for each age class via a softmax function, which can be turned into a negative log-likelihood loss functon by taking the logarithm of the output. This differentiation results in a convex equation that can be optimized via SGD. "Adam" is also a recommended loss function option to SGD+Nesterov available in Keras.
- *Learning Rate*: 0.001 -  The value that gave the best performance within the time constraint. This is the value prior to decay.
- *Decay*: 1e-6 - The amount to reduce the learning rate with every update
- *Momentum*: 0.9 - This is the typical value used for better convergence. It builds velocity in the direction with consistent gradients. 
- *Nesterov*: True - This calculates the "lookahead" gradient (the gradient of the future step).
- *Loss*: Categorical Cross-entropy - since the outputs are probabilities for 101 classes, this measures loss with a multi-class log-loss function. 
- *Metrics*: Accuracy - For classification, this is the only one available in Keras.
- *Dropout*: 0.5 - 50% of the values are randomly dropped from the layer to prevent overfitting.

The following parameters also tuned: 
- Batch Size: 32 - The maximum size (constrained by the memory of the server).
- Epochs: 50 - Defines the number of times each sample is seen during training. This was the limit for hyperparameter search given time constraints.

The model was trained on 727x2 images with the following splits
- 80% training
- 10% validation
- 10% testing

## Results
The accuracy of predicting ages was vastly improved to 52% accuracy with a 144 sample holdout (10% of the total dataset).

![TrainingResults](https://github.com/cjl2183/FilterAgeChange/tree/master/img/TrainingResults.png "Training Results")

Due to time constraints the epochs was set to 50, which wasn't enough for the validation accuracy to converge given the options above. However with more epochs, the accuracy could easily be improved beyond 52%. 

The difficulty of the parameter searching for neural networks has produced creative ways to shortcut the search - creating ensemble models from different "checkpoints", saving a "moving average" model, or averaging outputs from models with random sets of hyperparameters could bridge the gap to a model with the "optimal" hyperparameters. 

# Predicting Ages of Filtered Photos
Finally the following filters could be examined for their effect on the ages predicted by a model trained on unfiltered images. The following filters were used with a code snippet showing how the image is modified (note: these filters have since been deprecated in Instagram as of July, 2015):

- **Gotham**
```python
self.execute("convert {filename} -modulate 120,10,100 -fill '#222b6d' -colorize 20 -gamma 0.5 -contrast -contrast {filename}")
```
- **Kelvin**
```python
self.execute("convert \( {filename} -auto-gamma -modulate 120,50,100 \) \( -size {width}x{height} -fill 'rgba(255,153,0,0.5)' -draw 'rectangle 0,0 {width},{height}' \) -compose multiply {filename}");
```
- **Lomo**
```python
self.execute("convert {filename} -channel R -level 33% -channel G -level 33% {filename}")
```
[Filters]: https://github.com/cjl2183/FilterAgeChange/img/Filters.png "Filters"

The results are shown below with the histogram of ages from the original photos. This was converted into a kernel density estimation and overlayed onto the histogram of ages from the filtered photos. It appears that the *Gotham* filter shifted the histogram to the right, thereby "aging" the photos slightly. *Lomo* has a tendency to push ages into specific groups and redistribute from older age groups, thus reducing predicted ages overall. *Kelvin* has a mixed effect of pushing ages out of certain bins and spiking the histogram for certain classes. 

![Filter Ages](https://github.com/cjl2183/FilterAgeChange/tree/master/img/FilterAges.png "FilterAges")

# Conclusion
A clear outcome is that filters change the prediction of the model. An alternative approach could be to dive below the "filters" and see how the raw modifications affect age detection. For instance, by changing the "gamma" paramter, it shifts age distribution by x. By changing the red channel x%, the age distribution shifted by x. Then a heatmap of attributes to age modification could be created.

Ultimately a neural network could be trained to negate the effects from filters given enough observations. 

The possibilities are endless.
