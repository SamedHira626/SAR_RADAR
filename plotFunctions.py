#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

def plot_images(images, headers, num):

    if num <= 3:
        rows = 1
        cols = num
    elif num <= 6:
        rows = 2
        cols = 3
    else:
        rows = 3
        cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()

    for i in range(num):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(headers[i])
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
