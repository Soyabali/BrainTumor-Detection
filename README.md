# Brain Tumor Detection

A local Gradio interface for running the included YOLO checkpoint on brain scan
images. The app displays an annotated image and a table of detected classes,
confidence scores, and bounding-box coordinates.

## Requirements

- Python 3.10 or newer
- The checkpoint at `model/braintumor.pt` (included)

## Install and run

From this directory, run these commands in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

Open the local URL printed in the terminal (Gradio uses port 7860 by default
and selects another available port if needed), upload an image, adjust the
confidence and image-size settings if needed, and select **Run detection**. The
first inference loads the model and may take longer than subsequent inferences.

The app binds to localhost and does not expose the interface to other devices
by default. Ultralytics/PyTorch will use the available device automatically;
install a compatible PyTorch build separately if you need a specific CUDA
configuration.

## Medical-use notice

This application is for research and educational use only. Its output is not a
medical diagnosis and must not be used as a substitute for evaluation by a
qualified medical professional.
