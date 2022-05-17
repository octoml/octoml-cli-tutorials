import argparse
import numpy as np
from PIL import Image
from tritonclient.utils import InferenceServerException
import tritonclient.grpc as grpc_client
import tritonclient.http as httpclient
from ariel import function_from_model

# onnxruntime doesn't support python 3.10
try:
    from onnxruntime import InferenceSession
except ImportError:
    print("onnxruntime doesn't support python 3.10: use --triton, --local won't work")

def image_preprocess(imgs):
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


def interpret_cat_scores(prediction):
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


def run_triton(port, hostname="localhost"):
    # Preprocess input image
    image = Image.open("cat_input_images/prey1.jpeg")

    # Initialize the model 
    model = function_from_model("critterblock", preprocessing=image_preprocess, port=port, hostname=hostname)

    # Run inference
    pred = model(image)

    # Interpret predictions
    interpret_cat_scores(pred)

def run_triton_http(port, hostname="localhost"):
    # Prepare client
    url = "%s:%s" % (hostname, port)
    triton_client = httpclient.InferenceServerClient(url=url)

    # Preprocess input image
    image = Image.open("cat_input_images/prey1.jpeg")
    processed = image_preprocess([image])
    img = processed[0]


    # Initialize the model
    inputs = []
    # can look up inputs/outputs via:
    # curl http://localhost:8000/v2/models/critterblock/config
    inputs.append(httpclient.InferInput('image', [1,3,224,224], "FP32"))
    inputs[0].set_data_from_numpy(img, binary_data=False)

    # Run inference
    results = triton_client.infer("critterblock", inputs)


    # Interpret predictions
    pred = [results.as_numpy('cat_image_type')]
    interpret_cat_scores(pred)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose mode to run inference in.')
    parser.add_argument("--local", default=False, action="store_true")
    parser.add_argument("--triton", default=False, action="store_true")
    parser.add_argument("--hostname", default="localhost")
    parser.add_argument("--protocol", default="grpc")
    parser.add_argument("--port", default=8001)
    args = parser.parse_args()

    if args.local:
        run_local()

    if args.triton:
        if args.protocol == "grpc":
            run_triton(args.port, args.hostname)
        elif args.protocol == "http":
            # assume http to imply port 8000
            if args.port == 8001:
                http_port = 8000
            else:
                http_port = args.port
            run_triton_http(http_port, args.hostname)
        else:
            print("Unknown protocol given. Supported protocols are grpc and http")


