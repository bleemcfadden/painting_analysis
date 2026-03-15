#prediction.py, boni's

#libs
import os
import json
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

#models
MODEL_PATH = "model/painting_style_cnn.keras"

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"preprocess_input": preprocess_input}
)

dummy = np.zeros((1,224,224,3))
_ = model.predict(dummy)

last_conv_layer_name = "top_conv"

#classnames and threshold
with open("model/class_names.json") as f:
    class_names = json.load(f)

with open("model/threshold.json") as f:
    threshold = json.load(f)["threshold"]
    
#image preprocessing for uploaded files
def preprocess_image(img):

    img = cv2.resize(img, (224,224))
    img_arr = img.astype("float32")

    x = np.expand_dims(img_arr, axis=0)

    return img_arr, x

#prediction function
def predict_image(img_arr, x):

    preds = model.predict(x, verbose=0)[0]

    max_prob = float(np.max(preds))
    pred_idx = int(np.argmax(preds))

    if max_prob < threshold:
        pred_style = "Unknown"
    else:
        pred_style = class_names[pred_idx]

    return pred_style, max_prob, preds

#GRADCAM, please don't crash this time lol
#define the layers
preprocess_layer = model.layers[0]
augment_layer    = model.layers[1]
backbone         = model.layers[2]
head_layers      = model.layers[3:]

def call_layer(layer, x, training=False):
    try:
        return layer(x, training=training)
    except TypeError:
        return layer(x)

def make_gradcam_heatmap(img_array):
    with tf.GradientTape() as tape:
        # forward pass through preprocessing + augmentation
        x = call_layer(preprocess_layer, img_array, training=False)
        x = call_layer(augment_layer, x, training=False)

        # conv feature maps from backbone
        conv_outputs = call_layer(backbone, x, training=False)
        tape.watch(conv_outputs)

        # classifier head
        x = conv_outputs
        for layer in head_layers:
            x = call_layer(layer, x, training=False)

        preds = x
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None. Grad-CAM path is disconnected.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), int(pred_index.numpy()), preds.numpy()

#gradcam overlay
def overlay_gradcam(img, heatmap, alpha=0.35):
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = img.copy().astype("float32")
    overlay = overlay + alpha * heatmap
    overlay = overlay / np.max(overlay)
    overlay = np.uint8(255 * overlay)

    return overlay

#COLOR ANALYSIS please don't change format T_T
#kmeans for colors
color_ranges = {
    "Red-Yellow": (0, 90),
    "Green-Cyan": (91, 180),
    "Blue-Purple": (181, 270),
    "Magenta-Red": (271, 360)
}

def palette_by_hue_range(img, hue_min, hue_max, k=3):

    img = cv2.resize(img, (120,120)).astype("uint8")

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # convert OpenCV hue 0–179 → 0–360
    hue = hsv[:, :, 0].astype(float) * 2

    # non-circular hue segmentation
    mask = (hue >= hue_min) & (hue < hue_max)

    pixels = img[mask]

    if len(pixels) < 10:
        return None

    n_clusters = min(k, len(pixels))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    palette = centers[order]

    return palette[:k]

#for the 5 dom palette
def dominant_palette(img, k=5):

    img = cv2.resize(img, (120,120)).astype("uint8")

    pixels = img.reshape(-1,3)

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)

    labels = kmeans.fit_predict(pixels)

    centers = kmeans.cluster_centers_.astype(np.uint8)

    counts = np.bincount(labels)
    order = np.argsort(counts)[::-1]

    return centers[order]

#combine the palette so it can be seen as one picture. 
def build_palette_panel(img):
    row_height = 40
    label_width = 150
    palette_width = 300

    row_specs = [
        ("Dominant (5)", dominant_palette(img, k=5)),
        ("Red-Yellow (3)", palette_by_hue_range(img, 0, 90, k=3)),
        ("Green-Cyan (3)", palette_by_hue_range(img, 90, 180, k=3)),
        ("Blue-Purple (3)", palette_by_hue_range(img, 180, 270, k=3)),
        ("Magenta-Red (3)", palette_by_hue_range(img, 270, 360, k=3)),
    ]

    panel_rows = []

    for label, colors in row_specs:
        row = np.ones((row_height, label_width + palette_width, 3), dtype=np.uint8) * 255

        # write label
        cv2.putText(
            row,
            label,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

        # draw palette if exists
        if colors is not None and len(colors) > 0:
            n = len(colors)
            step = palette_width // n

            for i, c in enumerate(colors):
                x0 = label_width + i * step
                x1 = label_width + (i + 1) * step if i < n - 1 else label_width + palette_width
                row[:, x0:x1] = c

        panel_rows.append(row)

    panel = np.vstack(panel_rows)
    return panel

#radian histogram
def hue_radian_histogram(img, bins=60):
    img = cv2.resize(img, (200, 200))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)

    # keep colorful and visible pixels
    mask = (sat > 2) & (val > 20)

    hue = hue[mask]
    sat = sat[mask]
    val = val[mask]

    if len(hue) == 0:
        hue = (hsv[:, :, 0].astype(np.float32) * 2.0).flatten()
        weights = None
    else:
        weights = np.sqrt(sat/255) * (val/255)

    # rotate hue seam away from blue/cyan region
    hue = (hue + 180.0) % 360.0

    hue_rad = np.radians(hue)

    hist, edges = np.histogram(
        hue_rad,
        bins=bins,
        range=(0, 2 * np.pi),
        weights=weights
    )
    hist = hist / (hist.sum() + 1e-8)

    # circular smoothing
    hist_pad = np.r_[hist[-1], hist, hist[0]]
    hist = np.convolve(hist_pad, [0.25, 0.5, 0.25], mode="same")[1:-1]

    angles = edges[:-1]
    width = 2 * np.pi / bins

    # shift colors back so wheel labels/colors still look normal
    display_angles = (angles - np.pi) % (2 * np.pi)
    colors = plt.cm.hsv(display_angles / (2 * np.pi))

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="polar")

    ax.bar(
        angles,
        hist,
        width=width,
        bottom=0,
        color=colors,
        edgecolor="black",
        linewidth=0.3
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])
    ax.set_title("Hue Distribution")

    return fig
    # plt.show()
    
#boniistired_boniistired_boniistired_boniistired_boniistired_boniistired_boniistired
#main inference function
def predict_painting(img, artist=None, title=None):

    img_arr, x = preprocess_image(img)

    pred_style, confidence, preds = predict_image(img_arr, x)

    heatmap, _, _ = make_gradcam_heatmap(x)
    overlay = overlay_gradcam(img_arr, heatmap, alpha=0.25)

    palette_img = build_palette_panel(img_arr)

    hue_fig = hue_radian_histogram(img_arr)

    return {
        "prediction": pred_style,
        "confidence": confidence,
        "probabilities": preds,
        "overlay": overlay,
        "palette": palette_img,
        "hue_plot": hue_fig,
        "artist": artist,
        "title": title
    }

#multiple images added
def predict_multiple(images, metadata=None):

    results = []

    for i, img in enumerate(images):

        artist = None
        title = None

        if metadata:
            artist = metadata[i].get("artist")
            title = metadata[i].get("title")

        result = predict_painting(img, artist, title)

        results.append(result)

    return results

