# YoloV8.cpp

This C++ code is using the OpenCV library to perform object detection on a video stream using the YOLOv8 object detection model. 
Here's a breakdown of the code: 
1. The code includes necessary header files for OpenCV and file operations.
2. It defines several constants for input image size, confidence thresholds, font parameters, and colors.
3. The `draw_label` function is used to draw bounding boxes and labels on the input image.
4. The `load_class_list` function reads the class names from a file named "classes.txt" and stores them in a vector.
5. The `load_net` function loads the YOLOv8 model from the "yolov8s.onnx" file and sets the appropriate backend and target based on whether CUDA is available or not.
6. The `pre_process` function takes an input image and the loaded model, creates a blob from the image, sets it as input to the model, and performs forward propagation to get the model outputs.
7. The `post_process` function takes the input image, model outputs, and class names. It processes the outputs to extract bounding boxes, class IDs, and confidence scores. It then performs non-maximum suppression (NMS) to remove overlapping bounding boxes and draws the remaining bounding boxes and labels on the input image.
8. In the `main` function:
- The class list is loaded using `load_class_list`.
- A video capture object is created to read from the "sample.mp4" file.
- The YOLOv8 model is loaded using `load_net`. - The main loop reads frames from the video, processes them using `pre_process` and `post_process`, and displays the output with bounding boxes and labels.
- It also calculates and displays the frames per second (FPS) and inference time.
- The loop continues until the user presses the 'Esc' key. 9. Finally, the video capture object is released, and all windows are closed.
In summary, this code demonstrates how to use the OpenCV library and the YOLOv8 object detection model to perform real-time object detection on a video stream, with the ability to display bounding boxes, class labels, FPS, and inference time.

