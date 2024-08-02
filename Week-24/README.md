# Assignment
1. Select 5 different styles from community-created SD concepts library.
2. Apply these styles on prompt and show output.
3. Apply guidance loss on same prompt and show output.
4. Create a hugging face app.

# Introduction
The goal of this assignment is to use stable diffusion for image generation. Use different style
embeddings to generate output. Use gudance loss function to guide image generation in specific
direction.

## :golfing: Guidance loss

- Grayscale: To make some part of image black and white. 
- Bright: To increase bright colors in image.
- Contrast: To increase contrast colors in image.
- Symmetry: To make sure image is almost symmetric. 
- Saturation: To increase saturation of some colors in image.

## Gradio App
![Gradio-app](./images/gradio_app.png)  
Gradio App can be found [here](https://huggingface.co/spaces/AkashDataScience/SD_Textual_Inversion)

## Acknowledgments
This assignement is refering to code given in repo listed below
* [Stable Diffusion Deep Dive](https://github.com/fastai/diffusion-nbs)