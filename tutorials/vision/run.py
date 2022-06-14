import argparse
import numpy as np
from typing import List, Union
from PIL import Image

# import triton_util from parent dir
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from triton_util import TritonRemoteModel  # noqa


# onnxruntime doesn't support python 3.10
try:
    from onnxruntime import InferenceSession
except ImportError:
    print("onnxruntime doesn't support python 3.10: use --triton, --local won't work")


def image_preprocess(imgs: List[Image.Image]) -> List[np.ndarray]:
    processed_imgs = []
    for img in imgs:
        mean_pixel = np.asarray([0.485, 0.456, 0.406], np.float32)
        std_pixel = np.asarray([0.229, 0.224, 0.225], np.float32)
        image = np.asarray(img.resize((224, 224)))
        scaled_fp32 = image.astype(np.float32) / 255.0
        normalized = (scaled_fp32 - mean_pixel) / std_pixel
        transposed = np.rollaxis(normalized, 2, 0)
        processed_imgs.append(np.expand_dims(transposed, axis=0))
    return processed_imgs


def interpret_cat_scores(prediction: List[np.ndarray]):
    parsed_result = prediction[0][0]
    cat_score, approach_score, prey_score, _ = parsed_result[0], parsed_result[1], parsed_result[2], parsed_result[3]
    print(f"Cat score: {format(cat_score, '.3f')}")
    print(f"Approach score: {format(approach_score, '.3f')}")
    print(f"Prey score: {format(prey_score, '.3f')}")
    print('--------------------------------------')


def run_local():
    # Preprocess input image
    image = Image.open("cat_input_images/prey1.jpeg")
    processed_img = image_preprocess([image])[0]

    # Initialize the model
    session = InferenceSession('critterblock.onnx')

    # Run inference
    pred = session.run([], {"image": processed_img})

    # Interpret predictions
    interpret_cat_scores(pred)


def run_triton(port: Union[str, None], hostname: str, protocol: str):
    if port is None:
        port = '8000' if protocol == 'http' else '8001'

    server_url = f'{hostname}:{port}'

    # Preprocess input image
    image = Image.open("cat_input_images/prey1.jpeg")

    processed_img = image_preprocess([image])[0]

    # Initialize the model
    model = TritonRemoteModel(server_url, "critterblock", protocol=protocol)

    # Run inference
    pred = model(processed_img)

    # Interpret predictions
    interpret_cat_scores(pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose mode to run inference in.')
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--triton", default=False, action="store_true")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--port", default=None)
    parser.add_argument("--protocol", default="grpc", choices=["grpc", "http"])
    args = parser.parse_args()

    if args.local:
        run_local()

    if args.triton:
        run_triton(args.port, args.hostname, args.protocol)
