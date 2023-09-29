# Alzheimer-s-Classification-Model-from-MRI-Images

# Introduction + Set-up:

    1. The project focuses on building a machine learning model for determining the dementia level of Alzheimer's patients based on MRI images.
    2. TensorFlow 2.3 was used, taking advantage of its new features, and a GPU accelerator was employed for efficient computation.
    3. The project aimed to achieve a high ROC AUC score, indicating the model's ability to distinguish between different dementia levels.

# Data Loading:

    1. The Kaggle Alzheimer's dataset was used, requiring structured file directory format for image loading.
    2. A training-validation split of 80:20 was set up.
    3. Class names were defined, and the number of classes (4) was specified.

# Visualize the Data:

    1. Data visualization was performed to understand and verify the loaded images.
    2. A 3x3 grid of sample images with their corresponding class names was displayed.

# Feature Engineering:

    1. To handle categorical data, one-hot encoding was applied to the labels.
    2. Data caching and prefetching were implemented for efficient data processing.

# Deciding a Metric:

    1. Due to the dataset's imbalance, ROC AUC was chosen as the evaluation metric. A lower ROC AUC indicates that the model cannot distinguish between different classes, while a higher score implies better class separation.

# Build the ML Model:

    1. A convolutional neural network (CNN) model was built using TensorFlow's tf.keras.
    2. The model architecture consists of convolutional blocks and dense blocks, with batch normalization and dropout layers to prevent overfitting.
    3. The model aimed to predict the dementia level across four classes.

# Training the Model:

    1. Callbacks were used for more efficient model training, including learning rate scheduling, model checkpointing, and early stopping.
    2. Learning rate scheduling helps in adjusting the learning rate during training, preventing convergence issues.
    3. Model checkpointing saves the best model based on validation performance.
    4. Early stopping helps prevent overfitting by stopping training when performance no longer improves.

# Visualize Model Metrics:

    1. Training and validation metrics, including ROC AUC and loss, were visualized after each epoch to monitor model performance.
    2. The graphs showed how the model learned over time and highlighted the validation dataset's performance.

# Evaluate the Model:

    1. The model's performance was evaluated on a separate testing dataset.
    2. The testing dataset was prepared similarly to the training and validation datasets.
    3. The model achieved a loss of approximately 1.53 and a ROC AUC score of about 0.836 on the testing dataset.

# Conclusion:

    1. The project successfully built and trained a CNN model for dementia level prediction using MRI images.
    2. By using ROC AUC as the evaluation metric, the model demonstrated its ability to distinguish between dementia levels.
    3. The project showcases the application of machine learning in the health sciences, with TensorFlow 2.3 and GPU acceleration providing efficient tools for model development and training.
