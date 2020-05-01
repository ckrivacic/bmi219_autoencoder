# bmi219_autoencoder




> **Question 0.1) Why is it important to set the seed for the random number
generator?**

This is probably used to seed the weights. If we want to play around
with hyperparameters, it's best to compare networks that have been
seeded with the same rng. (?)

> **Q1.1) How many examples do the training set and test set have?**

Training: 60000
Test: 10000

> **Q1.2) What's the format of each input example? Can we directly put these into a fully-connected layer?**

Each input is a tuple of a 28x28 matrix of pixels and a label (0-9).


> **Q1.3) Why do we normalize the input data for neural networks?**

Large numbers can make the gradients initially very shallow during
back-propogation, making training very slow. We normalize to get data in a regime 
where activation by nonlinear functions like the sigmoid function won't immediately 
result in a shallow gradient regardless of the starting weights.


> **Q1.4) In this scenario, MNIST is already split into a training set 
and a test set. What is the purpose of dataset splitting (and specifically, 
the purpose of a test set)? For modern deep learning, a three-way split 
into training, validation, and test sets is usually preferred, why?**

It is possible for a neural net to memorize many of the features of the
training set. The test set ensures that the neural net works on data
that it hasn't ever seen before, ensuring generalizability.
While a test set is used to evaluate the model parameters once trained, a validation
set is used to evaluate the model hyperparameters during training. While the validation set 
does not directly influence the gradient, it can still bias the model since we 
will choose hyperparameters based on the model's performance on the validation set.


> **Q2.1) It's recommended to shuffle the training data over each epoch, 
but this isn't typically the case for the test set, why?**

Shuffling during training introduces stochasticity to the training
process, which is necessary because we are not guaranteed to find the
global minimum. This is unnecessary during the testing phase because we
are not changing the model parameters, just evaluating them.


> **Q2.2) What seems to be a good batch size for training? What happens if you train 
with a batch size of 1? What about a batch size equal to the total training set?**

A batch size of 1 trains much slower because it means you can't
parallelize the training, and its path through parameter space is 
much more erratic. Training on the whole dataset means you can
only learn about the averages of features and tends to converge on
sharp, non-generalizable minima. A good batch size for
training seems to be somewhere around 50-300.

> **Q2.3) The PyTorch DataLoader object is an iterator that generates batches as it's called. 
Try to pull a few images from the training set to see what these images look like. 
Does the DataLoader return only the images? What about the labels?**

It returns both the image (in 28x28 matrix/tensor form) and the label.
The labels are typically required for evaluating the model (though this
is not the case in an autoencoder).


> **Q3.1) What activation functions did you use, and why?**

Starting with ReLU because if the initial weights are significantly off, it has a
relatively strong gradient it can follow compared to the plateau of
other functions. Sigmoid didn't seem to work
quite as well (was probably just learning slower), but wasn't tested as extensively.

Sigmoid learning curve:
![Sigmoid](default_layers/lr_0.0001_sigmoid.png)

ReLU learning curve:
![ReLU](default_layers/lr_0.0001.png)

Note: These were done with a learning rate of 0.0001, so neither converged fully, but sigmoid 
appeared to be moving more slowly towards convergence.


> **Q5.1) What loss function is suited to this problem?**

Mean squared error, since this is a regression problem rather than a
classification one. I used MSE rather than mean error because MSE
penalizes outputs that are very wrong more harshly, resulting in steeper gradients.

> **Q5.2) Try a few optimizers, what seemed to work best?**

Note: All images below were generated with a learning rate of 0.0005, using the 
default architecture of 1000 -> 500 -> 250 -> 2 -> 250 -> 500 -> 1000.
Batch size was 128.

RProp was very slow, both in terms of convergence and the computational time 
it took to compute the gradients for each batch. Interestingly it was very smooth,
indicating that the momentum used was effective (though possibly more effective than is desirable).
![Rprop](default_layers/lr_0.0005_rprop.png)

RMSprop learned much faster, suggesting the way it calculates momentum may be 
stronger.
![RMSprop](default_layers/lr_0.0005_rmsprop.png)

SGD (mini-batch gradient descent) tended to move loss much more slowly. No image.
SGD with momentum moved loss much more quickly than SGD, but eventually around
epoch 43 the loss went to nan. Before this, loss looked pretty normal so
I'm not sure what's happening here. No image due to NAN loss.

Adam seems to work the best.
![Adam](default_layers/lr_0.0005.png)

> **Q5.3) What's the effect of choosing different batch sizes?**

Larger batches can cause overfitting due to converging on sharp local
minima, but train faster and follow a smoother path to the optimal
paramters than smaller batches.

Larger batch sizes also require a larger learning rate since the gradient 
is applied less often. For an extreme example, see below for batch size of 
128 vs. batch size of 60,000 with a learning rate of 0.0005.

Batch size=128:
![Batch size=128](default_layers/lr_0.0005.png)

Batch size = 60,000:
![Batch size=60,000](default_layers/lr_0.0005_fullbatch.png)

> **Q6.1)  How do you know when your network is done training?**

When the loss plateaus. Ideally, you would also be evaluating a
validation set, and use the parameters where the validation set is at
its lowest, since the test set can continue plateauing or improving while the model
overfits. The validation set helps us tell when the model is minimized
but still generalizable.


> **Q6.2) What does `torch.no_grad()` do?**

Turns off the gradient so that we are not wasting memory on it when we
only want to do feed-forward operations.


> ** Visualizing Training Progress **

Images tracking reconstruction for the test and train datasets every 10 epochs can be found in `[default_layers or mynet ]/[test_data_tracking or train_data_tracking]`

Here are some examples of test data from the "default" architecture (1000, 500, 250, 2, 250, 500, 1000).
Input data:
![input_data](default_layers/test_data_tracking/input_data.png)

After first epoch:
![epoch_0](default_layers/test_data_tracking/imshow_epoch_0.png)

After 50th epoch:
![epoch_50](default_layers/test_data_tracking/imshow_epoch_50.png)

After 100th epoch:
![epoch_100](default_layers/test_data_tracking/imshow_epoch_100.png)

After 140th epoch:
![epoch_140](default_layers/test_data_tracking/imshow_epoch_140.png)


> ** Visualizing Latent Space **

Latent space qualitatively correlates with convergence. The high learning rate, for instance, had a terrible looking latent space:


![lr_0.01](default_layers/latent_lr_0.01.png)

Similarly, using the full batch example, which didn't fully converge, we can see 
in the latent space that categories do not cleanly separate.

![full_batch](default_layers/latent_lr_0.0005_fullbatch.png)

The best model (lr=0.0005, converged, Adam optimizer, ReLU) had better separation but still had some trouble areas with numbers that are close to several others in apperance.

![best](default_layers/latent_lr_0.0005.png)


Note: If you run plot_latentspace(\*args), you can click on any spot and see what the decoder half of the model predicts for those two latent variables.


> ** My Neural Net **

I constructed a net with a smaller architecture of 500, 200, 2, 200, 500 with ReLU nonlinear terms. It trained much faster than the previous neural net with only slightly worse loss, though I was able to observe the beginning of the overfitting regime with this architecture.

![my_training](mynet/training.png)


Images from training are below.

Inputs:

![inputs](mynet/test_data_tracking/input_data.png)

Epoch 0:

![epoch_0](mynet/test_data_tracking/imshow_epoch_0.png)


Epoch 80:

![epoch_80](mynet/test_data_tracking/imshow_epoch_80.png)


Epoch 140:

![epoch_140](mynet/test_data_tracking/imshow_epoch_140.png)


Latent space:

![latent_space](mynet/latentspace.png)

Note that this simpler model actually has more easily identifiable classes than the default architecture, suggesting it may be more generalizable.
