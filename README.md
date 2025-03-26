> Download the model file and code here to run it locally. : https://huggingface.co/spaces/prithivMLmods/Magic-Eraser-Tool/tree/main


> Supported Streamlit Version [ pip install streamlit==1.34.0 ]

![App Screenshot](assets/magic-eraser.png)  <!-- Replace with actual screenshot path -->

# Magic Eraser Tool 🪄

Remove unwanted objects from your images with AI-powered inpainting!

## Features ✨

- Simple brush-based interface to mark areas for removal
- AI-powered inpainting technology
- Preserves original image quality
- Adjustable brush size for precise editing
- Download your edited images with one click

## How It Works 🛠️

1. Upload an image (JPG/PNG)
2. Use the brush tool to mark areas you want to remove
3. Click "Submit" and let AI work its magic
4. Download your cleaned image

## Installation 💻

To run locally:

```bash
git clone https://github.com/PRITHIVSAKTHIUR/Magic-Eraser-Tool.git
cd Magic-Eraser-Tool
pip install -r requirements.txt
streamlit run app.py
```

## Requirements 📋
- Streamlit==1.34.0
- NumPy
- Pillow
- pandas
- Other dependencies (see requirements.txt)
