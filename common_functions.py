import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
idx = pd.IndexSlice

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Functions for model evaluation


# Plot the loss development over epochs
def plot_loss_development(history, info:str|None=None):
    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # Plot the loss curves
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(train_loss, label="Training Loss", color="blue")
    ax.plot(val_loss, label="Validation Loss", color="orange")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Function Development Over Epochs{'' if info is None else info}")
    ax.set_ylim(0)
    ax.legend()
    ax.grid()
    return fig, ax


# Get the true/false positives/negatives if the data is one-hot encoded
def get_onehot_prediction_error(Y_pred, Y_test, prediction_threshold):
    # Set the values over the prediction_threshold as True
    Y_pred_bool = (Y_pred > prediction_threshold).astype(int)

    # Calculate the hits and errors
    Y_pred_err = Y_test - Y_pred_bool * 2
    Y_pred_err = Y_pred_err.apply(lambda x: x.value_counts()).astype(float)

    # Map the comparison values to their meaning
    error_map = {
        -2: "false positive",
        -1: "true positive",
        0: "true negative",
        1: "false negative",
    }
    Y_pred_err.index = pd.Index([error_map[x] for x in Y_pred_err.index])
    return Y_pred_err


# Get the metrics Accuracy, Precision, Recall, and F1-Score
def get_model_metrics(Y_test, Y_pred, prediction_threshold):
    # Set the values over the prediction_threshold as True
    Y_pred_bool = (Y_pred > prediction_threshold).astype(int)

    model_metrics = pd.DataFrame(
        columns=["Accuracy", "Precision", "Recall", "F1-Score"],
        index=pd.MultiIndex([[], []], [[], []]),
    )

    for column in list(Y_test.columns.get_level_values(0)) + ["total"]:
        if column == "total":
            Y_test_long = Y_test.droplevel(1, axis=1).stack()
            Y_pred_bool_long = Y_pred_bool.droplevel(1, axis=1).stack()
        else:
            Y_test_long = Y_test.droplevel(1, axis=1)[column]
            Y_pred_bool_long = Y_pred_bool.droplevel(1, axis=1)[column]

        # Total scores
        accuracy = accuracy_score(Y_test_long, Y_pred_bool_long)

        # Calculate precision
        if Y_pred_bool_long.sum() == 0:
            precision = np.nan
        else:
            precision = precision_score(Y_test_long, Y_pred_bool_long)

        # Calculate recall (sensitivity)
        recall = recall_score(Y_test_long, Y_pred_bool_long)

        # Calculate F1-score
        f1 = f1_score(Y_test_long, Y_pred_bool_long)

        # idx[column, f"n={int((Y_pred_bool_long!= 0).sum())}"]

        model_metrics.loc[idx[column, f"n={int((Y_test_long != 0).sum())}"], :] = (
            accuracy,
            precision,
            recall,
            f1,
        )

    # # Add the number of non-zero samples
    # model_metrics.loc[:, "n"] = (Y_test != 0).sum().droplevel(1)

    return model_metrics


# Preprocessing data
# https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers
def df_to_dataset(dataframe, targets, shuffle=True, batch_size=32):
    df = dataframe.copy()

    # Get the data to each first-level column
    df_sets = df.columns.levels[0]
    all_dict = {key: df[key].to_numpy() for key in df_sets}

    # Get the targets
    targets_dict = {k:all_dict.pop(k) for k in targets}

    # Make into Dataset
    ds = tf.data.Dataset.from_tensor_slices((dict(all_dict), dict(targets_dict)))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def get_dataset_from_input_df(dataframe, all_inputs, batch_size=32):
    df = dataframe.copy()

    # Get the data to each first-level column
    all_dict = {key: df[key].to_numpy() for key in all_inputs}

    # Make into Dataset
    ds = tf.data.Dataset.from_tensor_slices(dict(all_dict))
    return ds.batch(batch_size)

def get_normalization_layer(name, dataset, normLayer_name=None):
    # Create a Normalization layer for the feature.
    normalizer = layers.Normalization(axis=None, name=normLayer_name)

    # Prepare a Dataset that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the statistics of the data.
    normalizer.adapt(feature_ds)

    return normalizer


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None, lookupLayer_name=None, encodingLayer_name=None):
    # Create a layer that turns strings into integer indices.
    if dtype == "string":
        index = layers.StringLookup(max_tokens=max_tokens, name=lookupLayer_name)
    # Otherwise, create a layer that turns integer values into integer indices.
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens, name=lookupLayer_name)

    # Prepare a `tf.data.Dataset` that only yields the feature.
    feature_ds = dataset.map(lambda x, y: x[name])

    # Learn the set of possible values and assign them a fixed integer index.
    index.adapt(feature_ds)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size(), name=encodingLayer_name)

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(index(feature))

def get_bucket_encoding_layer(name, bin_boundaries, dicretizationLayer_name=None, encodingLayer_name=None):
    
    discretization_layer = layers.Discretization(bin_boundaries=bin_boundaries, name=dicretizationLayer_name)

    # Encode the integer indices.
    encoder = layers.CategoryEncoding(num_tokens=len(bin_boundaries), name=encodingLayer_name)

    # Apply multi-hot encoding to the indices. The lambda function captures the
    # layer, so you can use them, or include them in the Keras Functional model later.
    return lambda feature: encoder(discretization_layer(feature))

# Time Series normalisation Layer
class TimeSeriesNormalization(layers.Layer):
    def __init__(self, epsilon=1e-6, name=None):
        super(TimeSeriesNormalization, self).__init__()
        self.epsilon = epsilon  # To prevent division by zero

        if name is not None:
            self.name=name

    def call(self, inputs):
        """
        Normalize each time series independently to zero mean and unit variance.

        Args:
            inputs: Tensor of shape (batch_size, time_steps, features)

        Returns:
            Normalized tensor of the same shape
        """
        mean = tf.reduce_mean(
            inputs, axis=1, keepdims=True
        )  # Compute mean along time axis
        std = tf.math.reduce_std(
            inputs, axis=1, keepdims=True
        )  # Compute std along time axis

        return (inputs - mean) / (std + self.epsilon)  # Normalize


class NormalizedTimeSeriesWithDerivatives(layers.Layer):
    def __init__(self, epsilon=1e-6, name=None, **kwargs):
        super(NormalizedTimeSeriesWithDerivatives, self).__init__(**kwargs)
        self.epsilon = epsilon  # To prevent division by zero
        
        if name is not None:
            self.name=name

    def call(self, inputs):
        """
        Compute first and second numerical derivatives using TensorFlow operations, normalize each component,
        and concatenate all features.

        Args:
            inputs: Tensor of shape (batch_size, time_steps, 1) representing the time series.

        Returns:
            Normalized tensor of shape (batch_size, time_steps, 3), where:
            - First channel: Normalized original time series
            - Second channel: Normalized first derivative
            - Third channel: Normalized second derivative
        """
        # print(inputs.shape)

        # First derivative: Finite difference (forward difference method)
        first_derivative = inputs[:, 1:, :] - inputs[:, :-1, :]
        first_derivative = tf.pad(
            first_derivative, [[0, 0], [1, 0], [0, 0]]
        )  # Pad to maintain shape

        # print(first_derivative.shape)

        # Second derivative: Finite difference of first derivative
        second_derivative = first_derivative[:, 1:, :] - first_derivative[:, :-1, :]
        second_derivative = tf.pad(
            second_derivative, [[0, 0], [1, 0], [0, 0]]
        )  # Pad to maintain shape

        # print(second_derivative.shape)

        # Concatenate original signal with derivatives
        combined = tf.concat([inputs, first_derivative, second_derivative], axis=-1)

        # print(combined)

        # Normalize each feature independently (along time axis)
        mean = tf.reduce_mean(
            combined, axis=1, keepdims=True
        )  # Compute mean per sample
        std = tf.math.reduce_std(
            combined, axis=1, keepdims=True
        )  # Compute std per sample

        normalized_output = (combined - mean) / (std + self.epsilon)  # Normalize

        # print(normalized_output)

        return normalized_output
    
class TimeSeriesDerivatives(layers.Layer):
    def __init__(self, epsilon=1e-6, name=None, **kwargs):
        super(TimeSeriesDerivatives, self).__init__(**kwargs)
        self.name=name
        
        if name is not None:
            self.name=name

    def call(self, inputs):
        """
        Compute first and second numerical derivatives using TensorFlow operations, normalize each component,
        and concatenate all features.

        Args:
            inputs: Tensor of shape (batch_size, time_steps, 1) representing the time series.

        Returns:
            Normalized tensor of shape (batch_size, time_steps, 3), where:
            - First channel: Normalized original time series
            - Second channel: Normalized first derivative
            - Third channel: Normalized second derivative
        """
        # print(inputs.shape)

        # First derivative: Finite difference (forward difference method)
        first_derivative = inputs[:, 1:, :] - inputs[:, :-1, :]
        first_derivative = tf.pad(
            first_derivative, [[0, 0], [1, 0], [0, 0]]
        )  # Pad to maintain shape

        # print(first_derivative.shape)

        # Second derivative: Finite difference of first derivative
        second_derivative = first_derivative[:, 1:, :] - first_derivative[:, :-1, :]
        second_derivative = tf.pad(
            second_derivative, [[0, 0], [1, 0], [0, 0]]
        )  # Pad to maintain shape

        # print(second_derivative.shape)

        # Concatenate original signal with derivatives
        combined = tf.concat([inputs, first_derivative, second_derivative], axis=-1)

        return combined
    
# Plot the final metrics of a model
def plot_model_metrics(metrics, dataset, ylim=(0,1)):
    # Get the targets from the dataset to count occurrences
    _targets_subsets = list(dataset.map(lambda x,y: y).as_numpy_iterator())
    _targets = [pd.DataFrame({k:v.flatten() for k,v in _targets_subset.items()}) for _targets_subset in _targets_subsets]
    _targets = pd.concat(_targets, axis=0)
    target_number = _targets.sum().sort_values(ascending=False)

    # Get all metrics and assign them to the targets
    plot_metrics = pd.Series(
        {tuple(k.split("_..")): v for k,v in metrics.items() if not k.endswith("loss")}
    )
    metrics_names = plot_metrics.index.levels[1]


    # Create the figure
    fig, axes = plt.subplots(
        len(metrics_names) + 1, 
        1,
        figsize=(7,7),
        sharex=True
    )

    # Plot the number of occurrences
    axes[0].bar(
        range(len(target_number)),
        target_number
    )
    axes[0].set_title("Number of samples")
    axes[0].set_ylabel("Number of samples")

    for i, metric in enumerate(metrics_names):
        axes[i+1].bar(
            range(len(target_number)),
            plot_metrics.loc[idx[target_number.index, metric]]
        )
        axes[i+1].set_title(metric)
        axes[i+1].set_ylabel(metric)
        axes[i+1].set_ylim(ylim)


    axes[-1].set_xticks(list(range(len(target_number))), target_number.index.to_numpy())
    fig.tight_layout()

    return fig, axes

# Predict a model output for an input dataframe and return a dataframe
def predict_model_from_df(model, df):
    pred = model.predict(
        get_dataset_from_input_df(df, model.input)
    ) # For a Dataframe
    # test = model.predict(test_ds.map(lambda x,y : x)) # For a DataSet
    res = pd.DataFrame({k:v.flatten() for k,v in pred.items()})
    res.index = df.index
    return res