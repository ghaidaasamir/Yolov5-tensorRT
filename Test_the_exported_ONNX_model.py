import onnx
import onnxruntime as ort

# Load the ONNX model
model_path = "yolov5x.onnx"
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid!")

# Inference test
ort_session = ort.InferenceSession(model_path)
print("ONNX model loaded successfully for inference!")
