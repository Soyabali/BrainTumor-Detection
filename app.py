from functools import lru_cache
from pathlib import Path

import gradio as gr
from PIL import Image
from ultralytics import YOLO


MODEL_PATH = Path(__file__).resolve().parent / "model" / "braintumor.pt"
TABLE_HEADERS = ["Class", "Confidence", "X1", "Y1", "X2", "Y2"]


@lru_cache(maxsize=1)
def load_model() -> YOLO:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"YOLO model checkpoint not found: {MODEL_PATH}")
    return YOLO(str(MODEL_PATH))


def predict(image: Image.Image | None, confidence: float, image_size: int):
    if image is None:
        raise gr.Error("Upload a brain scan image before running detection.")

    results = load_model().predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )
    if not results:
        raise RuntimeError("The YOLO model returned no inference results.")

    result = results[0]
    detections = []
    boxes = result.boxes
    if boxes is not None:
        coordinates = boxes.xyxy.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        for box, class_id, score in zip(coordinates, class_ids, confidences):
            class_name = result.names[int(class_id)]
            detections.append(
                [
                    str(class_name),
                    round(float(score), 4),
                    *(round(float(coordinate), 1) for coordinate in box),
                ]
            )

    annotated_image = Image.fromarray(result.plot()[..., ::-1].copy())
    if detections:
        status = f"Detection complete: {len(detections)} object(s) found."
    else:
        status = "Detection complete: no objects found above the confidence threshold."
    return annotated_image, detections, status


with gr.Blocks(title="Brain Tumor Detection") as demo:
    gr.Markdown(
        """
        # Brain Tumor Detection
        Upload a brain scan image to run object detection with the included YOLO model.

        **For research and educational use only. This tool is not a medical diagnosis.**
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                type="pil",
                sources=["upload"],
                label="Upload a brain scan",
            )
            confidence = gr.Slider(
                minimum=0.05,
                maximum=0.95,
                value=0.25,
                step=0.05,
                label="Confidence threshold",
            )
            image_size = gr.Slider(
                minimum=320,
                maximum=1280,
                value=640,
                step=32,
                label="Inference image size",
            )
            detect_button = gr.Button("Run detection", variant="primary")
        with gr.Column(scale=1):
            output_image = gr.Image(type="pil", label="Detection result")
            detection_table = gr.Dataframe(
                headers=TABLE_HEADERS,
                datatype=["str", "number", "number", "number", "number", "number"],
                label="Detected objects",
                interactive=False,
            )
            status = gr.Textbox(label="Status", interactive=False)

    detect_button.click(
        fn=predict,
        inputs=[input_image, confidence, image_size],
        outputs=[output_image, detection_table, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1")
