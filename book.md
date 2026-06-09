# 📚 แหล่งข้อมูลและหนังสือประกอบการเรียน (Learning Resources)
## วิชา 31909-2007: การประมวลผลภาพดิจิทัล (Digital Image Processing)

> [!NOTE]
> เอกสารนี้รวบรวมแหล่งเรียนรู้ทั้งหนังสือ เว็บไซต์ เอกสารออนไลน์ และวิดีโอสอน  
> สำหรับทุกหัวข้อตลอด 15 สัปดาห์ อย่างน้อย 2–3 แหล่งต่อ 1 หัวข้อ  
> แหล่งข้อมูลจัดเรียงตามลำดับ **แนะนำมากที่สุด → เสริมความเข้าใจ**

---

## สารบัญ (Table of Contents)

- [แหล่งข้อมูลหลักของวิชา (Core References)](#แหล่งข้อมูลหลักของวิชา-core-references)
- [สัปดาห์ที่ 1: การติดตั้งสภาพแวดล้อมและ Image I/O](#สัปดาห์ที่-1-การติดตั้งสภาพแวดล้อมและ-image-io)
- [สัปดาห์ที่ 2: Image Manipulation & Geometric Transform](#สัปดาห์ที่-2-image-manipulation--geometric-transform)
- [สัปดาห์ที่ 3: Contrast & Filtering](#สัปดาห์ที่-3-contrast--filtering)
- [สัปดาห์ที่ 4: Edge Detection & Morphology](#สัปดาห์ที่-4-edge-detection--morphology)
- [สัปดาห์ที่ 5: Contour Detection](#สัปดาห์ที่-5-contour-detection)
- [สัปดาห์ที่ 6: Fourier Transform & Image Inpainting](#สัปดาห์ที่-6-fourier-transform--image-inpainting)
- [สัปดาห์ที่ 7: Feature Matching (SIFT, ORB, Homography)](#สัปดาห์ที่-7-feature-matching-sift-orb-homography)
- [สัปดาห์ที่ 8: สอบกลางภาค — แหล่งทบทวน](#สัปดาห์ที่-8-สอบกลางภาค--แหล่งทบทวน)
- [สัปดาห์ที่ 9: Deep Learning & CNN ด้วย PyTorch](#สัปดาห์ที่-9-deep-learning--cnn-ด้วย-pytorch)
- [สัปดาห์ที่ 10: Transfer Learning & ONNX](#สัปดาห์ที่-10-transfer-learning--onnx)
- [สัปดาห์ที่ 11: YOLO Object Detection (Inference)](#สัปดาห์ที่-11-yolo-object-detection-inference)
- [สัปดาห์ที่ 12: Custom YOLO Training & Evaluation](#สัปดาห์ที่-12-custom-yolo-training--evaluation)
- [สัปดาห์ที่ 13: MediaPipe (Pose, Hand, Face Landmark)](#สัปดาห์ที่-13-mediapipe-pose-hand-face-landmark)
- [สัปดาห์ที่ 14–15: โครงงานและสอบปลายภาค](#สัปดาห์ที่-14-15-โครงงานและสอบปลายภาค)

---

## แหล่งข้อมูลหลักของวิชา (Core References)

หนังสือและเว็บไซต์ที่ใช้อ้างอิงได้ตลอดทั้งเทอม:

| # | ประเภท | ชื่อ | ลิงก์ |
|---|--------|------|-------|
| 1 | 📖 ตำรา | **Digital Image Processing** — Rafael C. Gonzalez & Richard E. Woods (4th Edition) | [Pearson](https://www.pearson.com/en-us/subject-catalog/p/digital-image-processing/P200000003224) |
| 2 | 📖 ตำรา | **Learning OpenCV 4** — Adrian Kaehler & Gary Bradski (O'Reilly) | [O'Reilly](https://www.oreilly.com/library/view/learning-opencv-4/9781492057994/) |
| 3 | 🌐 เว็บ | **OpenCV Official Documentation** | [docs.opencv.org](https://docs.opencv.org/4.6.0/) |
| 4 | 🌐 เว็บ | **PyImageSearch** — Adrian Rosebrock (บทความ + โค้ดตัวอย่าง OpenCV & DL) | [pyimagesearch.com](https://pyimagesearch.com/) |
| 5 | 🌐 เว็บ | **LearnOpenCV** — Satya Mallick (บทความ + GitHub) | [learnopencv.com](https://learnopencv.com/) |
| 6 | 🎬 วิดีโอ | **freeCodeCamp — OpenCV Course (Full)** | [YouTube](https://www.youtube.com/watch?v=oXlwWbU8l2o) |

---

## สัปดาห์ที่ 1: การติดตั้งสภาพแวดล้อมและ Image I/O

**หัวข้อ**: VS Code, Miniconda, Virtual Environment, Pixels, Channels (BGR), `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **Miniconda Installation Guide** — วิธีติดตั้งและใช้งาน Conda | [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html) |
| 2 | 🌐 เอกสาร | **VS Code — Python Tutorial** — การตั้งค่า Python ใน VS Code | [code.visualstudio.com](https://code.visualstudio.com/docs/python/python-tutorial) |
| 3 | 🌐 บทความ | **OpenCV — Getting Started with Images** — อ่าน/เขียน/แสดงภาพ | [docs.opencv.org](https://docs.opencv.org/4.6.0/db/deb/tutorial_display_image.html) |
| 4 | 🎬 วิดีโอ | **Murtaza's Workshop — OpenCV in 3 Hours** — บทนำ OpenCV Python | [YouTube](https://www.youtube.com/watch?v=WQeoO7MI0Bs) |
| 5 | 📖 ตำรา | **Gonzalez & Woods — Ch.1: Introduction** — ทฤษฎีภาพดิจิทัลเบื้องต้น | ดูหนังสือ Core Reference #1 |

---

## สัปดาห์ที่ 2: Image Manipulation & Geometric Transform

**หัวข้อ**: Resize, Crop, Flip, Rotation, Affine Transform, NumPy Array Slicing, Data Augmentation

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — Geometric Transformations** — resize, warpAffine, flip | [docs.opencv.org](https://docs.opencv.org/4.6.0/da/d6e/tutorial_py_geometric_transformations.html) |
| 2 | 🌐 บทความ | **PyImageSearch — Image Rotation and Translation** | [pyimagesearch.com](https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/) |
| 3 | 🌐 บทความ | **LearnOpenCV — Image Resizing with OpenCV** | [learnopencv.com](https://learnopencv.com/image-resizing-with-opencv/) |
| 4 | 📖 ตำรา | **Gonzalez & Woods — Ch.2: Digital Image Fundamentals** — พิกัด, ระบบสี, การสุ่มตัวอย่าง | ดูหนังสือ Core Reference #1 |
| 5 | 🌐 เอกสาร | **NumPy — Array Indexing & Slicing** — การจัดการ pixel ด้วย NumPy | [numpy.org](https://numpy.org/doc/stable/user/basics.indexing.html) |

---

## สัปดาห์ที่ 3: Contrast & Filtering

**หัวข้อ**: Histogram, CLAHE, Brightness/Contrast, Gaussian Blur, Median Blur, Bilateral Filter, Noise Reduction

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — Histograms** — การวิเคราะห์ฮิสโตแกรมและ Equalization | [docs.opencv.org](https://docs.opencv.org/4.6.0/d1/db7/tutorial_py_histogram_begins.html) |
| 2 | 🌐 เอกสาร | **OpenCV — Smoothing Images (Blurring)** — Gaussian, Median, Bilateral | [docs.opencv.org](https://docs.opencv.org/4.6.0/d4/d13/tutorial_py_filtering.html) |
| 3 | 🌐 บทความ | **LearnOpenCV — Histogram Equalization & CLAHE** | [learnopencv.com](https://learnopencv.com/histogram-equalization-and-adaptive-histogram-equalization/) |
| 4 | 📖 ตำรา | **Gonzalez & Woods — Ch.3: Intensity Transformations and Spatial Filtering** | ดูหนังสือ Core Reference #1 |
| 5 | 🎬 วิดีโอ | **Sentdex — OpenCV Filtering Tutorial** | [YouTube](https://www.youtube.com/watch?v=ZW0XQ3lMvLI) |

---

## สัปดาห์ที่ 4: Edge Detection & Morphology

**หัวข้อ**: Sobel, Laplacian, Canny Edge Detection, Erosion, Dilation, Opening, Closing, Morphological Operations

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — Canny Edge Detection** | [docs.opencv.org](https://docs.opencv.org/4.6.0/da/d22/tutorial_py_canny.html) |
| 2 | 🌐 เอกสาร | **OpenCV — Morphological Transformations** — Erosion, Dilation, Opening, Closing | [docs.opencv.org](https://docs.opencv.org/4.6.0/d9/d61/tutorial_py_morphological_ops.html) |
| 3 | 🌐 บทความ | **PyImageSearch — Sobel and Laplacian Edge Detection** | [pyimagesearch.com](https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/) |
| 4 | 📖 ตำรา | **Gonzalez & Woods — Ch.9: Morphological Image Processing** | ดูหนังสือ Core Reference #1 |
| 5 | 🌐 บทความ | **LearnOpenCV — Edge Detection Using OpenCV** | [learnopencv.com](https://learnopencv.com/edge-detection-using-opencv/) |

---

## สัปดาห์ที่ 5: Contour Detection

**หัวข้อ**: `findContours()`, `drawContours()`, Bounding Box, Moments, Area, Perimeter, ROI Cropping

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — Contours: Getting Started** | [docs.opencv.org](https://docs.opencv.org/4.6.0/d4/d73/tutorial_py_contours_begin.html) |
| 2 | 🌐 เอกสาร | **OpenCV — Contour Features** — Area, Perimeter, Bounding Rect, Moments | [docs.opencv.org](https://docs.opencv.org/4.6.0/dd/d49/tutorial_py_contour_features.html) |
| 3 | 🌐 บทความ | **PyImageSearch — Contour Detection with OpenCV** | [pyimagesearch.com](https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/) |
| 4 | 📖 ตำรา | **Gonzalez & Woods — Ch.10: Image Segmentation** — การแบ่งส่วนภาพ | ดูหนังสือ Core Reference #1 |
| 5 | 🌐 บทความ | **LearnOpenCV — Contour Detection using OpenCV** | [learnopencv.com](https://learnopencv.com/contour-detection-using-opencv-python-c/) |

---

## สัปดาห์ที่ 6: Fourier Transform & Image Inpainting

**หัวข้อ**: 2D DFT, Frequency Domain, Low-pass/High-pass Filter, Inpainting, JPEG Compression, MSE, PSNR

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — Fourier Transform** — DFT ใน OpenCV | [docs.opencv.org](https://docs.opencv.org/4.6.0/de/dbc/tutorial_py_fourier_transform.html) |
| 2 | 🌐 เอกสาร | **OpenCV — Image Inpainting** — `cv2.inpaint()` | [docs.opencv.org](https://docs.opencv.org/4.6.0/df/d3d/tutorial_py_inpainting.html) |
| 3 | 📖 ตำรา | **Gonzalez & Woods — Ch.4: Filtering in the Frequency Domain** — ทฤษฎี Fourier | ดูหนังสือ Core Reference #1 |
| 4 | 🌐 บทความ | **LearnOpenCV — Image Inpainting with OpenCV** | [learnopencv.com](https://learnopencv.com/image-inpainting-with-opencv-c-python/) |
| 5 | 🎬 วิดีโอ | **3Blue1Brown — But what is the Fourier Transform?** — ทฤษฎี Fourier แบบเข้าใจง่าย | [YouTube](https://www.youtube.com/watch?v=spUNpyF58BY) |

---

## สัปดาห์ที่ 7: Feature Matching (SIFT, ORB, Homography)

**หัวข้อ**: Keypoints, Descriptors, SIFT, ORB, Brute-Force Matcher, Homography, Perspective Transform

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV — ORB (Oriented FAST and Rotated BRIEF)** | [docs.opencv.org](https://docs.opencv.org/4.6.0/d1/d89/tutorial_py_orb.html) |
| 2 | 🌐 เอกสาร | **OpenCV — Feature Matching** — BFMatcher, FLANN | [docs.opencv.org](https://docs.opencv.org/4.6.0/dc/dc3/tutorial_py_matcher.html) |
| 3 | 🌐 เอกสาร | **OpenCV — Feature Homography** — `findHomography()`, Perspective Transform | [docs.opencv.org](https://docs.opencv.org/4.6.0/d1/de0/tutorial_py_feature_homography.html) |
| 4 | 🌐 บทความ | **PyImageSearch — OpenCV Feature Matching with SIFT and ORB** | [pyimagesearch.com](https://pyimagesearch.com/2015/07/16/where-did-sift-and-surf-go-in-opencv-3/) |
| 5 | 📖 ตำรา | **Gonzalez & Woods — Ch.11: Feature Extraction** | ดูหนังสือ Core Reference #1 |

---

## สัปดาห์ที่ 8: สอบกลางภาค — แหล่งทบทวน

**หัวข้อ**: ทบทวนสัปดาห์ 1–7 (ภาพดิจิทัล, การกรอง, ฮิสโตแกรม, Edge, Contour, Fourier, Feature Matching)

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 บทความ | **OpenCV Python Tutorials (รวมทุกหัวข้อ)** — สรุปครบ | [docs.opencv.org](https://docs.opencv.org/4.6.0/d6/d00/tutorial_py_root.html) |
| 2 | 🎬 วิดีโอ | **freeCodeCamp — OpenCV Full Course** — ทบทวนทั้งหมดในวิดีโอเดียว | [YouTube](https://www.youtube.com/watch?v=oXlwWbU8l2o) |
| 3 | 📖 ตำรา | **Gonzalez & Woods — Ch.1–11** — ทบทวนทฤษฎีทั้งหมด | ดูหนังสือ Core Reference #1 |
| 4 | 🌐 บทความ | **GeeksforGeeks — Digital Image Processing** — สรุปสูตรและทฤษฎี | [geeksforgeeks.org](https://www.geeksforgeeks.org/digital-image-processing-basics/) |

---

## สัปดาห์ที่ 9: Deep Learning & CNN ด้วย PyTorch

**หัวข้อ**: Neural Networks, CNN Architecture, Loss Function, Optimizer, Train Loop, MNIST, Dataset & DataLoader

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **PyTorch Official Tutorials — Deep Learning with PyTorch: A 60 Minute Blitz** | [pytorch.org](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) |
| 2 | 🌐 เอกสาร | **PyTorch — Training a Classifier (CIFAR-10)** — วิธีเทรน CNN เบื้องต้น | [pytorch.org](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) |
| 3 | 📖 ตำรา | **Dive into Deep Learning** — Zhang, Lipton, Li & Smola (ฟรี) — CNN Chapters | [d2l.ai](https://d2l.ai/) |
| 4 | 🎬 วิดีโอ | **Andrej Karpathy — Neural Networks: Zero to Hero** — สร้าง NN จากศูนย์ | [YouTube](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) |
| 5 | 🎬 วิดีโอ | **3Blue1Brown — Neural Networks** — อธิบาย NN ด้วยภาพอนิเมชัน | [YouTube](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) |
| 6 | 🌐 บทความ | **PyImageSearch — Intro to CNN with PyTorch** | [pyimagesearch.com](https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/) |

---

## สัปดาห์ที่ 10: Transfer Learning & ONNX

**หัวข้อ**: Pre-trained Models (MobileNetV3, ResNet), Fine-tuning, ONNX Export, OpenCV DNN Module

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **PyTorch — Transfer Learning Tutorial** — Fine-tune MobileNet/ResNet | [pytorch.org](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) |
| 2 | 🌐 เอกสาร | **PyTorch — Export to ONNX** — วิธีส่งออกโมเดลเป็น `.onnx` | [pytorch.org](https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html) |
| 3 | 🌐 เอกสาร | **OpenCV DNN Module** — `cv2.dnn.readNetFromONNX()` | [docs.opencv.org](https://docs.opencv.org/4.6.0/d2/d58/tutorial_table_of_content_dnn.html) |
| 4 | 🌐 บทความ | **LearnOpenCV — Using Pre-trained Models with OpenCV DNN** | [learnopencv.com](https://learnopencv.com/deep-learning-with-opencvs-dnn-module-a-definitive-guide/) |
| 5 | 🌐 เอกสาร | **ONNX Official Documentation** — สเปกมาตรฐาน ONNX | [onnx.ai](https://onnx.ai/onnx/intro/) |

---

## สัปดาห์ที่ 11: YOLO Object Detection (Inference)

**หัวข้อ**: Object Detection vs Classification, YOLO Architecture, Bounding Box, Confidence, IoU, Ultralytics YOLOv8

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **Ultralytics YOLOv8 Documentation** — คู่มือหลัก | [docs.ultralytics.com](https://docs.ultralytics.com/) |
| 2 | 🌐 เอกสาร | **Ultralytics — Predict Mode** — การรัน Inference กับภาพ/วิดีโอ | [docs.ultralytics.com](https://docs.ultralytics.com/modes/predict/) |
| 3 | 🌐 บทความ | **LearnOpenCV — YOLOv8 Ultralytics: State-of-the-Art YOLO Models** | [learnopencv.com](https://learnopencv.com/ultralytics-yolov8/) |
| 4 | 📄 เปเปอร์ | **You Only Look Once (YOLO) — Original Paper** — Joseph Redmon et al. | [arXiv](https://arxiv.org/abs/1506.02640) |
| 5 | 🎬 วิดีโอ | **Nicolai Nielsen — YOLOv8 Object Detection Python** | [YouTube](https://www.youtube.com/watch?v=WgPbbWmnXJ8) |

---

## สัปดาห์ที่ 12: Custom YOLO Training & Evaluation

**หัวข้อ**: Dataset Annotation, YOLO Label Format, Train/Val/Test Split, Custom Training, mAP, Confusion Matrix

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **Ultralytics — Train Mode** — วิธีเทรน Custom YOLO | [docs.ultralytics.com](https://docs.ultralytics.com/modes/train/) |
| 2 | 🌐 เครื่องมือ | **Roboflow** — เครื่องมือ Label ภาพ + จัดการ Dataset ออนไลน์ (ฟรี) | [roboflow.com](https://roboflow.com/) |
| 3 | 🌐 เครื่องมือ | **CVAT (Computer Vision Annotation Tool)** — เครื่องมือ Label แบบ Open Source | [cvat.ai](https://www.cvat.ai/) |
| 4 | 🌐 บทความ | **Ultralytics — Tips for Best Training Results** — เทคนิคเทรนให้แม่นยำ | [docs.ultralytics.com](https://docs.ultralytics.com/guides/model-training-tips/) |
| 5 | 🎬 วิดีโอ | **Roboflow — Train YOLOv8 on Custom Dataset (Full Walkthrough)** | [YouTube](https://www.youtube.com/watch?v=wuZtUMEiKWY) |

---

## สัปดาห์ที่ 13: MediaPipe (Pose, Hand, Face Landmark)

**หัวข้อ**: Landmark Detection, Hand/Face/Pose Estimation, Angle Calculation, Gesture Recognition, Drowsiness Detection

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **MediaPipe Solutions — Official Guide** — Pose, Hand, Face Mesh | [developers.google.com](https://developers.google.com/mediapipe/solutions) |
| 2 | 🌐 เอกสาร | **MediaPipe — Pose Landmark Detection** — จุดข้อต่อร่างกาย 33 จุด | [developers.google.com](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |
| 3 | 🌐 เอกสาร | **MediaPipe — Hand Landmark Detection** — จุดพิกัดนิ้วมือ 21 จุด | [developers.google.com](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) |
| 4 | 🎬 วิดีโอ | **Murtaza's Workshop — Advanced Computer Vision with Python (MediaPipe)** | [YouTube](https://www.youtube.com/watch?v=01sAkU_NvOY) |
| 5 | 🌐 บทความ | **LearnOpenCV — MediaPipe Pose Estimation** | [learnopencv.com](https://learnopencv.com/introduction-to-mediapipe/) |
| 6 | 🌐 GitHub | **MediaPipe Python Examples** — ตัวอย่างโค้ดจาก Google | [GitHub](https://github.com/google-ai-edge/mediapipe) |

---

## สัปดาห์ที่ 14–15: โครงงานและสอบปลายภาค

**หัวข้อ**: ทบทวน CNN, YOLO, MediaPipe, OpenCV DNN, การประเมินโมเดล (IoU, Accuracy, Confusion Matrix), Project Presentation

### แหล่งทบทวนรวม

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เอกสาร | **OpenCV Python Tutorials (สารบัญรวม)** | [docs.opencv.org](https://docs.opencv.org/4.6.0/d6/d00/tutorial_py_root.html) |
| 2 | 🌐 เอกสาร | **PyTorch Tutorials (สารบัญรวม)** | [pytorch.org](https://pytorch.org/tutorials/) |
| 3 | 🌐 เอกสาร | **Ultralytics Documentation (สารบัญรวม)** | [docs.ultralytics.com](https://docs.ultralytics.com/) |
| 4 | 🌐 เอกสาร | **MediaPipe Solutions (สารบัญรวม)** | [developers.google.com](https://developers.google.com/mediapipe/solutions) |

### แหล่งเรียนรู้เพิ่มเติมสำหรับทำโครงงาน

| # | ประเภท | ชื่อ / คำอธิบาย | ลิงก์ |
|---|--------|-----------------|-------|
| 1 | 🌐 เว็บ | **Kaggle** — ชุดข้อมูล Dataset ฟรีและ Notebook ตัวอย่าง | [kaggle.com](https://www.kaggle.com/) |
| 2 | 🌐 เว็บ | **Papers with Code** — ค้นหา Paper + โค้ดของงานวิจัย CV ล่าสุด | [paperswithcode.com](https://paperswithcode.com/) |
| 3 | 🌐 เว็บ | **Roboflow Universe** — Dataset สำเร็จรูปพร้อม Label สำหรับ Object Detection | [universe.roboflow.com](https://universe.roboflow.com/) |
| 4 | 📖 ตำรา | **Deep Learning for Computer Vision with Python** — Adrian Rosebrock | [pyimagesearch.com](https://pyimagesearch.com/deep-learning-computer-vision-python-book/) |

---

## สรุปแหล่งข้อมูลแยกตามประเภท

### 📖 หนังสือ / ตำรา

| ชื่อ | ผู้แต่ง | เหมาะสำหรับ |
|------|---------|-------------|
| Digital Image Processing (4th Ed.) | Gonzalez & Woods | ทฤษฎีภาพดิจิทัลทุกบท (สัปดาห์ 1–7) |
| Learning OpenCV 4 | Kaehler & Bradski | เขียนโค้ด OpenCV เชิงปฏิบัติ |
| Dive into Deep Learning | Zhang et al. (ฟรีออนไลน์) | Deep Learning, CNN, PyTorch (สัปดาห์ 9–10) |
| Deep Learning for CV with Python | Adrian Rosebrock | โครงงาน CV ขั้นสูง |

### 🌐 เว็บไซต์หลัก

| เว็บไซต์ | URL | จุดเด่น |
|----------|-----|---------|
| OpenCV Docs | [docs.opencv.org](https://docs.opencv.org/4.6.0/) | เอกสาร + ตัวอย่างโค้ด OpenCV ครบ |
| PyTorch Tutorials | [pytorch.org/tutorials](https://pytorch.org/tutorials/) | เอกสาร + ตัวอย่าง DL/CNN |
| Ultralytics Docs | [docs.ultralytics.com](https://docs.ultralytics.com/) | คู่มือ YOLO ครบจบ |
| MediaPipe Docs | [developers.google.com/mediapipe](https://developers.google.com/mediapipe/solutions) | Pose / Hand / Face |
| PyImageSearch | [pyimagesearch.com](https://pyimagesearch.com/) | บทความ CV + DL คุณภาพสูง |
| LearnOpenCV | [learnopencv.com](https://learnopencv.com/) | บทความพร้อมโค้ด GitHub |
| d2l.ai | [d2l.ai](https://d2l.ai/) | ตำรา DL ฟรีออนไลน์ |

### 🎬 ช่อง YouTube แนะนำ

| ช่อง | เนื้อหาหลัก | ลิงก์ |
|------|-------------|-------|
| 3Blue1Brown | ทฤษฎี Neural Networks, Fourier Transform (อนิเมชัน) | [YouTube](https://www.youtube.com/@3blue1brown) |
| Andrej Karpathy | Neural Networks จากศูนย์ | [YouTube](https://www.youtube.com/@andaborean) |
| Murtaza's Workshop | OpenCV + MediaPipe (ปฏิบัติ) | [YouTube](https://www.youtube.com/@maboreanvr) |
| freeCodeCamp | คอร์ส OpenCV เต็ม (ฟรี) | [YouTube](https://www.youtube.com/@freecodecamp) |
| Nicolai Nielsen | YOLO, Object Detection | [YouTube](https://www.youtube.com/@NicolaiAI) |

---

> **จัดทำโดย**: ระบบเอกสารวิชา Digital Image Processing  
> **อัปเดตล่าสุด**: มิถุนายน 2026
