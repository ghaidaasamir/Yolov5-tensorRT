# Yolov5-tensorRT
Here’s how to convert a **YOLOv5 model** trained in PyTorch to the **ONNX format** and run the program:

### **Step 1: Install YOLOv5**
1. Clone the YOLOv5 repository:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### **Step 2: Train or Use a Pretrained YOLOv5 Model**
If you already have a trained YOLOv5 model (`best.pt`), you can skip training.

1. **Train the Model**:
   ```bash
   python train.py --img 640 --batch 16 --epochs 100 --data coco.yaml --weights yolov5s.pt
   ```
   - This command trains the model with the specified parameters and saves the weights as `best.pt` in the `runs/train/` directory.

2. **Download a Pretrained Model** (if you don’t want to train):
   ```bash
   wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
   ```
   - Use pretrained YOLOv5 models like `yolov5s.pt`, `yolov5m.pt`, or `yolov5l.pt` (small, medium, large).

---

### **Step 3: Export the Model to ONNX**
1. Use the provided export script in YOLOv5 to convert the PyTorch model (`.pt`) to ONNX format:
   ```bash
   python export.py --weights yolov5s.pt --img 640 --batch 1 --device 0 --simplify --include onnx
   ```

2. **Explanation of Arguments**:
   - `--weights yolov5s.pt`: Specifies the PyTorch weights to export.
   - `--img 640`: The input image size (YOLOv5 works with square images, e.g., 640x640).
   - `--batch 1`: Batch size for inference.
   - `--device 0`: Use GPU (0) for export; use `--device cpu` to export on the CPU.
   - `--simplify`: Simplifies the ONNX graph to remove redundant operations.
   - `--include onnx`: Specifies the format(s) to export (in this case, ONNX).

3. After running this command:
   - The ONNX model will be saved in the `runs/` directory, e.g., `runs/exp/yolov5s.onnx`.

---

### **Step 4: Verify the ONNX Model**
1. Install the ONNX runtime library:
   ```bash
   pip install onnxruntime
   ```

2. Test the exported ONNX model:
   ```python
   import onnx
   import onnxruntime as ort

   # Load the ONNX model
   model_path = "runs/exp/yolov5s.onnx"
   onnx_model = onnx.load(model_path)
   onnx.checker.check_model(onnx_model)
   print("ONNX model is valid!")

   # Inference test
   ort_session = ort.InferenceSession(model_path)
   print("ONNX model loaded successfully for inference!")
   ```

---

### **Step 5: (Optional) Visualize the ONNX Model**
You can visualize the ONNX graph to ensure the structure is correct.

1. Install **Netron**:
   ```bash
   pip install netron
   ```

2. Launch the Netron app to view the ONNX model:
   ```bash
   netron runs/exp/yolov5s.onnx
   ```

---

### **Step 6: Use the ONNX Model in TensorRT**
Once you have the YOLOv5 ONNX model, you can integrate it with TensorRT:
   ```bash
   /usr/src/tensorrt/bin/trtexec --onnx=/path/to/yolov5x.onnx --saveEngine=/path/to/yolov5x.engine  --minShapes=images:1x3x640x640 --optShapes=images:1x3x640x640 --maxShapes=images:1x3x640x640
   ```

### **Step 7: Build the Project Using CMake**
To compile the code for using the YOLOv5 ONNX model with TensorRT, you need to build the project using CMake.

1. **Create a build directory**:
   In your project directory (where the CMakeLists.txt file is located), create a `build` directory:
   ```bash
   mkdir build
   cd build
   ```

2. **Run CMake**:
   Generate the Makefiles by running the following command:
   ```bash
   cmake ..
   ```

3. **Build the project**:
   Once CMake has finished configuring the project, compile the code using:
   ```bash
   make
   ```

4. **Run the compiled program**:
   After the build is complete, you should have an executable.
   You can run the program:
   ```bash
   ./yolo-tensorrt /path/to/yolov5x.onnx /path/to/image.jpeg /path/to/yolov5x.engine
   ```
