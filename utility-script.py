# %% [code]
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def install_from_git(path):
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"git+{path}"])
    
install_from_git(r"https://github.com/keisen/tf-keras-vis.git")

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tf_keras_vis.saliency import Saliency
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.utils import num_of_gpus

from tf_keras_vis.activation_maximization import ActivationMaximization
from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D, Scale
from tf_keras_vis.activation_maximization.regularizers import Norm, TotalVariation2D

# %% [code]
def decode_categorical_label(label, class_names):
    return class_names[np.argmax(label)]

def decode_categorical_labels(labels, class_names):
    return [decode_categorical_label(label, class_names) for label in labels]

def decode_sparse_label(label, class_names):
    return class_names[label]

def decode_sparse_labels(labels, class_names):
    return [decode_sparse_label(label, class_names) for label in labels]

def plot_overlayed_images(images, maps, cmap, alpha, figsize, titles=None):
    n = len(images)
    fig, axs = plt.subplots(ncols=n, figsize=figsize)
    
    if n == 1:
        if titles is not None:
            axs.set_title(titles[0])
        axs.imshow(images[0] / 255)
        axs.imshow(maps[0], cmap=cmap, alpha=alpha)
    else:
        for i in range(n): 
            if titles is not None:
                axs[i].set_title(titles[i])
            axs[i].imshow(images[i] / 255)
            axs[i].imshow(maps[i], cmap=cmap, alpha=alpha)
    return fig, axs

def copy_model(model):
    copied_model = tf.keras.models.clone_model(model)
    copied_model.set_weights(model.get_weights())
    return copied_model

def plot_images_with_heatmaps(images, maps, cmap, figsize, titles=None):
    n = len(images)
    fig, axs = plt.subplots(nrows=2, ncols=n, figsize=figsize)
    if n == 1:
        if titles is not None:
            axs[0].set_title(titles[0])
        axs[0].imshow(images[0] / 255)
        axs[1].imshow(maps[0], cmap=cmap)
    else:    
        for i in range(n):
            if titles is not None:
                axs[0, i].set_title(titles[i])
            axs[0, i].imshow(images[i] / 255)
            axs[1, i].imshow(maps[i], cmap=cmap)
    return fig, axs

def dense_score_function(output):
    return (output[0][0], output[0][1])

def dense_model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

def plot_attentions(model, images, method, clone=True, labels=None, overlay=True, cmap="jet", figsize=(10,8), alpha=0.3, smooth=False, smooth_samples=10, smooth_noise=0.2):
    n = len(images)
    maps = []
    
    if method == "saliency":
        vis_model = Saliency(model, model_modifier=dense_model_modifier_function, clone=clone)
        for image in images:
            if smooth:
                maps.append(vis_model(dense_score_function, image, smooth_samples=smooth_samples, smooth_noise=smooth_noise)[0,:,:])
            else:
                maps.append(vis_model(dense_score_function, image)[0,:,:])
        
    elif method == "gradcam":
        vis_model = Gradcam(model, model_modifier=dense_model_modifier_function, clone=clone)
        for image in images:
            maps.append(vis_model(dense_score_function, image, penultimate_layer=-1)[0,:,:])
            
    elif method == "gradcam_pp":
        vis_model = GradcamPlusPlus(model, model_modifier=dense_model_modifier_function, clone=clone)
        for image in images:
            maps.append(vis_model(dense_score_function, image, penultimate_layer=-1)[0,:,:])
    
    if overlay:
        fig, axs = plot_overlayed_images(images, maps, cmap, alpha, figsize, titles=labels)
    else:
        fig, axs = plot_images_with_heatmaps(images, maps, cmap, figsize, titles=labels)
    
    return fig, axs

# %% [code]
def plot_activation_maximization_conv(model, layer_name, filters_numbers, clone=True, figsize=(12, 4)):
    
    def model_modifier_function(current_model):
        target_layer = current_model.get_layer(name=layer_name)
        target_layer.activation = tf.keras.activations.linear
        new_model = tf.keras.Model(inputs=current_model.inputs, outputs=target_layer.output)
        return new_model
    
    score = CategoricalScore(filters_numbers)
    
    activation_maximization = ActivationMaximization(model, model_modifier=model_modifier_function, clone=clone)

    n = 1
    
    if isinstance(filters_numbers, int):
        activations = activation_maximization(score, callbacks=[Progress()])
        filters_numbers = [filters_numbers]
    else:
        n = len(filters_numbers)
        seed_input = tf.random.uniform((n, *IMAGE_SHAPE), 0, 255)
        activations = activation_maximization(score, seed_input=seed_input, callbacks=[Progress()])

    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
    if n == 1:
        axs = [axs]
    for i, activation in enumerate(activations):
        axs[i].set_title(f'filter[{filters_numbers[i]}]', fontsize=16)
        axs[i].imshow(activation)
        axs[i].axis('off')
    plt.tight_layout()
    return fig, axs

def plot_activation_maximization_dense(model, class_idx, figsize=(12, 4), clone=True):
    
    score = CategoricalScore(class_idx)
    activation_maximization = ActivationMaximization(model, model_modifier=dense_model_modifier_function, clone=clone)
    
    n = 1
    if isinstance(class_idx, int):
        activations = activation_maximization(score, callbacks=[Progress()])
        class_idx = [class_idx]
    else:
        n = len(class_idx)
        seed_input = tf.random.uniform((n, *IMAGE_SHAPE), 0, 255)
        activations = activation_maximization(score, seed_input=seed_input, callbacks=[Progress()])
    
    
    fig, axs = plt.subplots(nrows=1, ncols=n, figsize=figsize)
    if n == 1:
        axs = [axs]
    for i, activation in enumerate(activations):
        axs[i].set_title("all" if class_idx[i] == 1 else "hem", fontsize=16)
        axs[i].imshow(activation)
        axs[i].axis('off')
    plt.tight_layout()
    return fig, axs