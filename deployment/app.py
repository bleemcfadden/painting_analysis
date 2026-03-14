#app.py boni is tired and cranky. 
import gradio as gr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from prediction import predict_painting

MAX_RESULTS = 5

def run_inference(images, artist, title):

    outputs = []

    if images is None:
        return [None]*(MAX_RESULTS*4)

    for i in range(MAX_RESULTS):

        if i >= len(images):
            outputs.extend([None, None, None, ""])
            continue

        img_file = images[i]

        img = Image.open(img_file.name).convert("RGB")
        img = np.array(img)

        result = predict_painting(
            img,
            artist=artist if artist else None,
            title=title if title else None
        )

        fig = result["hue_plot"]
        fig.canvas.draw()
        hue_img = np.asarray(fig.canvas.buffer_rgba())[...,:3]
        plt.close(fig)

        text = f"""
Prediction {i+1}

Style: {result["prediction"]}
Confidence: {result["confidence"]:.3f}

Artist: {result["artist"] or "Unknown"}
Title: {result["title"] or "Unknown"}
"""

        outputs.extend([
            result["overlay"],
            result["palette"],
            hue_img,
            text
        ])

    return outputs

with gr.Blocks(css="""
#palette img {
    margin-top: 40px;
}
""") as demo:

    gr.Markdown("# 🦉 Painting Style Analyzer")

    images = gr.File(
        file_count="multiple",
        file_types=["image"],
        label="Upload Painting(s)"
    )

    artist = gr.Textbox(label="Artist (optional)")
    title = gr.Textbox(label="Painting Title (optional)")

    run_button = gr.Button("Analyze Painting")

    outputs = []

    for i in range(MAX_RESULTS):

        with gr.Column():

            with gr.Row():

                overlay = gr.Image(label="Painting + GradCAM")
                palette = gr.Image(label="Color Palette", elem_id="palette")
                hue = gr.Image(label="Hue Distribution")

            text = gr.Textbox(label="Prediction", lines=4)

            outputs.extend([overlay, palette, hue, text])

    run_button.click(
        fn=run_inference,
        inputs=[images, artist, title],
        outputs=outputs
    )

demo.launch()