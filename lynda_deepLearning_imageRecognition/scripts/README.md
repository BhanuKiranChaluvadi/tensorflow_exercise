Deep Learning tutorials by "Adam Geitgey" from Lynda.com


We have four different kinds of layers in this neural network. 
1. The convolutional layers add translational invariance, 
2. The max pooling layers down sample the data, 
3. And dropout forces the neural network to learn in a more robust way.
4. And then finally, the dense layer maps the output of the previous layers to the output layer so we can predict which class the image belongs to.

The first three layers work really well together, so we'll put them together into a block and we'll call the whole thing a convolutional block.


Tutorial - 03: Designing a Deep Neural Network for Image Recognition.
Tutorial - 04: Building and Training the Deep Neural Network.
Tutorial - 05: Fine-Tuning Pre-trained Neural Network - transfer learning.
Tutorial - 06: Using an Image Recognition API



The batch size is how many images we want to feed into the network at once during training. If we set the number too low, training will take a long time and might not ever finish. If we set the number too high, we'll run out of memory on our computer.
Typical batch sizes are between 32 and 128 images. let's use a batch size of 32, next, we need to decide how many times we wanna go through our training data set during the process. One full pass through the entire training data set is called an epoch. For this example, let's do 30 passes through the training data set.

The more passes through the data we do, the more chance the neural network has to learn; but the longer the training process will take. And eventually you'll hit a point where doing additional training doesn't help anymore. So, finding the right number takes some experimentation. In general, the larger your data set, the less training passes you'll do on it. For example, for extremely large data sets with millions of images you might only do five passes.

Keras expects these batches as a four dimensional array.
The first dimension is the list of images, and the other three dimensions are the image data itself. 
