import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# load mnist dataset
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train/255
x_test = x_test/255

image_width = 28
image_height = 28
num_channels = 1

x_train = x_train.reshape(x_train.shape[0], image_height, image_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], image_height, image_width, num_channels)
x_train = np.concatenate([x_train, x_test], axis = 0)
batch_size = 64

# Here we need to use tensorflow dataset object because we are training this without using model.fit
train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(10).batch(batch_size)

# First we build Generator model. Which takes input as noise and produce images
# We have to decide the diamention of noise
NOISE_DIM = 150

# Design a Generator model means we are upsampling layers
generator_input_layer = tf.keras.layers.Input(shape = (NOISE_DIM))
g1 = tf.keras.layers.Dense(7*7*256, activation = 'relu')(generator_input_layer)
g2 = tf.keras.layers.Reshape((7, 7, 256))(g1)
g3 = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation = 'relu', strides = (2, 2), padding = 'same')(g2)
g4 = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation = 'relu', strides = (2, 2), padding = 'same')(g3)
generator_output_layer = tf.keras.layers.Conv2DTranspose(1, (3, 3), activation = 'sigmoid', padding = 'same')(g4)

# Now we have to define and summerize Generator
generator = tf.keras.models.Model(generator_input_layer, generator_output_layer, name = 'generator')
print(generator.summary())

# Now we have to build Discriminator model. Which takes input as image and predict real or fake
# Design a Discriminator means we are downsampling layers
discriminator_input_layer = tf.keras.layers.Input(shape = (28, 28, 1))
d1 = tf.keras.layers.Conv2D(256, (3, 3), activation = 'LeakyReLU', strides = (2, 2), padding = 'same')(discriminator_input_layer)
d2 = tf.keras.layers.Conv2D(128, (3, 3), activation = 'LeakyReLU', strides = (2, 2), padding = 'same')(d1)
d3 = tf.keras.layers.Flatten()(d2)
d4 = tf.keras.layers.Dense(64, activation = 'relu')(d3)
d5 = tf.keras.layers.Dropout(0.2)(d4)
discriminator_output_layer = tf.keras.layers.Dense(1, activation = 'sigmoid')(d5)

# Now we have to define and summarize Discriminator 
discriminator = tf.keras.models.Model(discriminator_input_layer, discriminator_output_layer, name = 'discriminator')
print(discriminator.summary())

# Prepare optimizer, loss function,  for Generator and Discriminator
optimizerG = tf.keras.optimizers.Adam(learning_rate = 0.00001)
optimizerD = tf.keras.optimizers.Adam(learning_rate = 0.00003)

loss = tf.keras.losses.BinaryCrossentropy()

gAccMetric = tf.keras.metrics.BinaryAccuracy()
dAccMetric = tf.keras.metrics.BinaryAccuracy()

# Let's make everything together
# Let's first create function for Discriminator
def trainDstep(data):
    '''
    This function is for discriminator training
    It only focus on discriminator to train properly
    '''
    # batch size is 64, so extract the value
    batchSize = tf.shape(data)[0]
    # we have to create a noise vector as generator input sampled from Gaussian Random Normal
    noise = tf.random.normal(shape = (batchSize, NOISE_DIM))
    
    # Concatinate the real and fake labels for generator and discriminator
    # First batchSize portion is assign as real and we mark as 1
    # Second batchSize portion is assign as fake and we mark as 0
    y_true = tf.concat([
        # the original data is real, labeled with 1
        tf.ones(batchSize, 1),
        tf.zeros(batchSize, 1)
    ],
    axis = 0)
    
    # Now calculate the gradient using tf.GradientTape()
    # tf.GradientTape provides hooks that give the user control over what is or not whatched
    with tf.GradientTape() as tape:
        fake = generator(noise)
        fake = tf.cast(fake, tf.float32)
        data = tf.cast(data, tf.float32) 
        # It produces fake images from generator
        # Now we have to concatenate real and fake data to train discriminator
        # generator batch size = 32
        # discriminator batch size = 32 + 32 
        x = tf.concat([data, fake], axis = 0)
        y_pred = discriminator(x)
        # calculate discriminator loss
        discriminatorLoss = loss(y_true, y_pred)

    # Now we have to train our discriminator using our predefine optimizer
    grads = tape.gradient(discriminatorLoss, discriminator.trainable_weights)
    optimizerD.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Now accuracy
    dAccMetric.update_state(y_true, y_pred)

    # Now return accuracy and loss for visualization
    return {
        'discriminator_loss' : discriminatorLoss,
        'discriminator_accuracy' : dAccMetric.result()
    }

# Let's create the function for generator
def trainGstep(data):
    '''
    This function is for generator training
    It only focus on generator to train properly
    '''
    batchSize = tf.shape(data)[0]
    noise = tf.random.normal(shape = (batchSize, NOISE_DIM))
    # Here our aim is to build a model which can build fake images and classified as real
    y_true = tf.ones(batchSize, 1)
    
    # Now calculate the gradient using tf.GradientTape()
    with tf.GradientTape() as tape:
        y_pred = discriminator(generator(noise))
        generatorLoss = loss(y_true, y_pred)
    
    # Now we have to train generator using our predefine  optimizer
    grads = tape.gradient(generatorLoss, generator.trainable_weights)
    optimizerG.apply_gradients(zip(grads, generator.trainable_weights))

    # Now accuracy
    gAccMetric.update_state(y_true, y_pred)

    return {
        'generator_loss' : generatorLoss,
        'generator_accuracy' : gAccMetric.result()
    }

# Create checkpoint to save the model
checkpoint_prefix = 'gan_model'
checkpoint = tf.train.Checkpoint(optimizerG = optimizerG, optimizerD = optimizerD, generator = generator, discriminator = discriminator)

# Save images to understand how my model improving in every epochs

# Saving images to check whether generator is improving or not!
def plotImages(model, epoch):
    images = model(tf.random.normal(shape = (81, NOISE_DIM)))
    images = images * 255
    plt.figure(figsize = (9, 9))
    for i, image in enumerate(images):
        plt.subplot(9, 9, i+1)
        plt.imshow(np.squeeze(image, -1), cmap = "Greys_r")
        plt.axis('off')
    plt.savefig('./images_per_epochs/epoch_' + str(epoch) +'_.png')
# Here we don't use model.fit functyion since the original GANs paper trained the discriminator for 5
# steps and then generator for 1 step
# Here we train both for 1 step
# visualize image in every few epochs is important
# We assign 30 epochs

for epoch in range(100):
    # Accumulate the loss to calculate the average at the end of epoch
    dLossSum = 0
    gLossSum = 0
    dAccSum = 0
    gAccSum = 0
    count = 0
    
    # loop the dataset one batch at a time
    for batch in train_dataset:
        # Train the discriminator
        # in original paper discriminator is trained for 5 times and then generator trained for 1 times
        # We can do this by repeating next 2 lines of codes for k times
        dLoss = trainDstep(batch)
        dLossSum += dLoss['discriminator_loss']
        dAccSum += dLoss['discriminator_accuracy']

        # Train the generator
        gLoss = trainGstep(batch)
        gLossSum += gLoss['generator_loss']
        gAccSum += gLoss['generator_accuracy']

        count += 1

    # Now save the model
    if (epoch + 1) % 20 == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    print('Epochs: {}, Generator Loss: {}, Discriminator Loss: {}, Generator Accuracy: {}, Discriminator Accuracy: {}'.format(epoch, gLossSum/count, dLossSum/count, gAccSum/count, dAccSum/count))
    plotImages(generator, epoch)