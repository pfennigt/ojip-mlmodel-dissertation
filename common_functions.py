import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Functions for model evaluation

# Plot the loss development over epochs
def plot_loss_development(history):
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_loss, label='Training Loss', color='blue')
    plt.plot(val_loss, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Function Development Over Epochs')
    plt.ylim(0)
    plt.legend()
    plt.grid()
    plt.show()

# Get the true/false positives/negatives if the data is one-hot encoded
def get_onehot_prediction_error(Y_pred, Y_test, prediction_threshold):
    # Set the values over the prediction_threshold as True
    Y_pred_bool = (Y_pred > prediction_threshold).astype(int)

    # Calculate the hits and errors
    Y_pred_err = (Y_test - Y_pred_bool * 2)
    Y_pred_err = Y_pred_err.apply(lambda x: x.value_counts()).astype(float)

    # Map the comparison values to their meaning
    error_map = {
        -2: "false positive",
        -1: "true positive",
        0: "true negative",
        1: "false negative"
    }
    Y_pred_err.index = pd.Index([error_map[x] for x in Y_pred_err.index ])
    return Y_pred_err

# Get the metrics Accuracy, Precision, Recall, and F1-Score
def get_model_metrics(Y_test, Y_pred, prediction_threshold):
    # Set the values over the prediction_threshold as True
    Y_pred_bool = (Y_pred > prediction_threshold).astype(int)

    model_metrics = pd.DataFrame(
        columns = ["Accuracy", "Precision", "Recall", "F1-Score"],
        index = pd.MultiIndex([[],[]], [[],[]])
    )

    for column in list(Y_test.columns.get_level_values(0)) + ["total"]:
        if column == "total":
            Y_test_long = Y_test.droplevel(1, axis=1).stack()
            Y_pred_bool_long = Y_pred_bool.droplevel(1, axis=1).stack()
        else:
            Y_test_long = Y_test.droplevel(1, axis=1)[column]
            Y_pred_bool_long = Y_pred_bool.droplevel(1, axis=1)[column]


        # Total scores
        accuracy = accuracy_score(Y_test_long, Y_pred_bool_long )

        # Calculate precision
        if Y_pred_bool_long.sum() == 0:
            precision = np.nan
        else:
            precision = precision_score(Y_test_long, Y_pred_bool_long )

        # Calculate recall (sensitivity)
        recall = recall_score(Y_test_long, Y_pred_bool_long )

        # Calculate F1-score
        f1 = f1_score(Y_test_long, Y_pred_bool_long )

        #idx[column, f"n={int((Y_pred_bool_long!= 0).sum())}"]

        model_metrics.loc[idx[column, f"n={int((Y_test_long!= 0).sum())}"], :] = (accuracy, precision, recall, f1)

    # # Add the number of non-zero samples
    # model_metrics.loc[:, "n"] = (Y_test != 0).sum().droplevel(1)

    return model_metrics

# Preprocessing data
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value.to_numpy()[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'str':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))