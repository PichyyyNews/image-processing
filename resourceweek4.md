# 📚 แหล่งข้อมูลและทรัพยากรประกอบการสอน — สัปดาห์ที่ 4
## หัวข้อ: การสกัดเส้นขอบและการดัดแปลงสัณฐานวิทยา (Edge Detection & Morphological Operations)

> [!NOTE]
> เอกสารฉบับนี้รวบรวมแหล่งข้อมูลอ้างอิง สื่อประกอบการสอน แหล่งภาพประกอบ และทรัพยากรต่าง ๆ สำหรับใช้ในการจัดทำสไลด์บรรยายและเตรียมการสอนสัปดาห์ที่ 4 ของวิชา 31909-2007 การประมวลผลภาพดิจิทัล

---

## สารบัญ

1. [ตำราเรียนและเอกสารวิชาการ (Textbooks & Academic References)](#1-ตำราเรียนและเอกสารวิชาการ)
2. [บทความวิจัยต้นฉบับ (Original Research Papers)](#2-บทความวิจัยต้นฉบับ)
3. [เอกสาร OpenCV อย่างเป็นทางการ (Official OpenCV Documentation)](#3-เอกสาร-opencv-อย่างเป็นทางการ)
4. [บทเรียนออนไลน์ — Edge Detection](#4-บทเรียนออนไลน์--edge-detection)
5. [บทเรียนออนไลน์ — Morphological Operations](#5-บทเรียนออนไลน์--morphological-operations)
6. [วิดีโอการสอน (YouTube & Video Resources)](#6-วิดีโอการสอน)
7. [แหล่งภาพประกอบสไลด์ (Image Resources for Slides)](#7-แหล่งภาพประกอบสไลด์)
8. [เครื่องมือ Interactive และ Demo ออนไลน์](#8-เครื่องมือ-interactive-และ-demo-ออนไลน์)
9. [แหล่งข้อมูลภาษาไทย](#9-แหล่งข้อมูลภาษาไทย)
10. [สรุปเนื้อหาหลักสำหรับทำสไลด์](#10-สรุปเนื้อหาหลักสำหรับทำสไลด์)

---

## 1. ตำราเรียนและเอกสารวิชาการ

### 📖 ตำราหลัก (Primary Textbooks)

| ตำรา | ผู้แต่ง | สำนักพิมพ์ / ปี | บทที่เกี่ยวข้อง | หมายเหตุ |
|---|---|---|---|---|
| **Digital Image Processing** (4th Edition) | Rafael C. Gonzalez & Richard E. Woods | Pearson, 2018 | **Ch.10**: Image Segmentation (Edge Detection: Sobel, Prewitt, Canny) / **Ch.9**: Morphological Image Processing | ✅ ตำราหลักที่ใช้กันทั่วโลก มีสูตรคณิตศาสตร์ครบถ้วน เหมาะอ้างอิงสำหรับทฤษฎี |
| **Computer Vision: Algorithms and Applications** (2nd Edition) | Richard Szeliski | Springer, 2022 | **Ch.3**: Image Processing (Edge Detection, Morphology) | ✅ **ดาวน์โหลด PDF ฟรี** ที่ [szeliski.org/Book](https://szeliski.org/Book/) |
| **Computer Vision: Principles, Algorithms, Applications, Learning** (5th Edition) | E.R. Davies | Academic Press, 2018 | บทว่าด้วย Edge Detection and Morphological Filtering | เน้นการประยุกต์ใช้งานจริงในอุตสาหกรรม |
| **Learning OpenCV 4** | Adrian Kaehler & Gary Bradski | O'Reilly, 2019 | บทว่าด้วย Image Gradients, Edge Detection, Morphology | เหมาะสำหรับอ้างอิงโค้ด OpenCV |

### 📖 ตำราเสริม (Supplementary References)

| ตำรา | ผู้แต่ง | หมายเหตุ |
|---|---|---|
| **Digital Image Processing Using MATLAB** | Rafael C. Gonzalez, Richard E. Woods & Steven L. Eddins | เหมาะสำหรับดูตัวอย่างภาพผลลัพธ์ประกอบ |
| **Image Processing, Analysis, and Machine Vision** (4th Edition) | Milan Sonka, Vaclav Hlavac & Roger Boyle | เนื้อหาเชิงลึกเรื่อง Edge Detection |

---

## 2. บทความวิจัยต้นฉบับ

### 📄 Edge Detection — บทความสำคัญ

| บทความ | ผู้แต่ง | ปี | วารสาร / การประชุม | ความสำคัญ |
|---|---|---|---|---|
| **A Computational Approach to Edge Detection** | John F. Canny | 1986 | IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), Vol.8, No.6, pp.679–698 | 🏆 **บทความต้นฉบับของ Canny Edge Detection** — อัลกอริทึมที่ได้รับความนิยมสูงสุดในการหาเส้นขอบ |
| **An Isotropic 3×3 Image Gradient Operator** (Sobel Operator) | Irwin Sobel & Gary Feldman | 1968 | นำเสนอที่ Stanford Artificial Intelligence Project | บทความต้นฉบับของตัวกรอง Sobel |
| **Theory of Edge Detection** | David Marr & Ellen Hildreth | 1980 | Proceedings of the Royal Society of London, Series B, Vol.207, pp.187–217 | ทฤษฎี Marr-Hildreth และ Laplacian of Gaussian (LoG) |
| **Image Analysis Using Mathematical Morphology** | Robert M. Haralick, Stanley R. Sternberg & Xinhua Zhuang | 1987 | IEEE TPAMI, Vol.9, No.4, pp.532–550 | บทความรากฐาน Morphological Operations |
| **Image Algebra and Morphological Image Processing** | Jean Serra | 1982 | หนังสือ "Image Analysis and Mathematical Morphology" (Academic Press) | 📘 ผู้คิดค้น Mathematical Morphology |

> [!TIP]
> สำหรับการทำสไลด์ แนะนำให้อ้างอิง **Canny (1986)** เป็นหลัก เพราะเป็น "Gold Standard" ของ Edge Detection และเพิ่ม **Gonzalez & Woods** สำหรับภาพรวมทฤษฎี

---

## 3. เอกสาร OpenCV อย่างเป็นทางการ

### 🔧 Tutorial Pages (Python)

| หัวข้อ | URL | เนื้อหาหลัก |
|---|---|---|
| **Image Gradients (Sobel, Laplacian)** | [docs.opencv.org — Image Gradients](https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html) | อธิบาย Sobel, Scharr, Laplacian พร้อมโค้ดตัวอย่าง |
| **Canny Edge Detection** | [docs.opencv.org — Canny Edge](https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html) | อธิบาย 5 ขั้นตอนของ Canny พร้อมตัวอย่าง |
| **Morphological Transformations** | [docs.opencv.org — Morphological Ops](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) | Erosion, Dilation, Opening, Closing, Gradient, Top Hat, Black Hat |

### 🔧 API Reference (ฟังก์ชันที่ใช้ใน Week 4)

| ฟังก์ชัน | URL | คำอธิบาย |
|---|---|---|
| `cv2.Sobel()` | [docs.opencv.org — Sobel](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gacea54f142e81b6758d467564f5c7e3e3) | ตัวกรอง Sobel สำหรับ Gradient แกน X/Y |
| `cv2.Laplacian()` | [docs.opencv.org — Laplacian](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gad78703e4c8fe703d479c1860d76f7842) | ตัวกรอง Laplacian (อนุพันธ์อันดับสอง) |
| `cv2.Canny()` | [docs.opencv.org — Canny](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de) | Canny Edge Detection |
| `cv2.erode()` | [docs.opencv.org — erode](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaeb1e0c1033e3f6b891a25d0511362aeb) | Erosion (กัดกร่อน) |
| `cv2.dilate()` | [docs.opencv.org — dilate](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga4ff0f3318642c4f469d0e11f242f3b6c) | Dilation (ขยาย) |
| `cv2.morphologyEx()` | [docs.opencv.org — morphologyEx](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#ga67493776e3ad1a3df63883829375201f) | Opening, Closing, Gradient, Top Hat, Black Hat |
| `cv2.getStructuringElement()` | [docs.opencv.org — getStructuringElement](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac342a1bb6eabf6f55c803b09268e36dc) | สร้าง Structuring Element (Kernel) |

---

## 4. บทเรียนออนไลน์ — Edge Detection

### 🌐 เว็บไซต์บทเรียนชั้นนำ

| แหล่ง | URL | เนื้อหาหลัก | ระดับ |
|---|---|---|---|
| **PyImageSearch** — Zero-parameter Auto Canny | [pyimagesearch.com — Auto Canny](https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/) | วิธีตั้ง Threshold อัตโนมัติด้วยค่า Median | ⭐⭐⭐ |
| **PyImageSearch** — Edge Detection 101 | [pyimagesearch.com](https://pyimagesearch.com) | พื้นฐาน Edge Detection ด้วย OpenCV | ⭐⭐ |
| **LearnOpenCV** — Edge Detection Guide | [learnopencv.com — Edge Detection](https://learnopencv.com/edge-detection-using-opencv/) | เปรียบเทียบ Sobel, Laplacian, Canny พร้อมโค้ด Python/C++ | ⭐⭐⭐ |
| **GeeksforGeeks** — Canny Edge Detection | [geeksforgeeks.org — Canny](https://www.geeksforgeeks.org/python-opencv-canny-function/) | อธิบาย Canny พร้อมตัวอย่างภาพ | ⭐⭐ |
| **GeeksforGeeks** — Image Gradients | [geeksforgeeks.org — Gradients](https://www.geeksforgeeks.org/python-opencv-cv2-sobel-method/) | อธิบาย Sobel Method | ⭐⭐ |
| **Real Python** — Image Processing | [realpython.com](https://realpython.com/python-opencv-color-spaces/) | บทเรียน OpenCV เชิงปฏิบัติ | ⭐⭐ |
| **scikit-image** — Edge Detection Gallery | [scikit-image.org — Edge examples](https://scikit-image.org/docs/stable/auto_examples/edges/plot_edge_filter.html) | ตัวอย่างเปรียบเทียบ Edge Detector หลายแบบ | ⭐⭐⭐ |

### 📓 Lecture Notes จากมหาวิทยาลัย

| มหาวิทยาลัย | หัวข้อ | หมายเหตุ |
|---|---|---|
| Stanford CS231A | Computer Vision Lecture Notes | เนื้อหา Edge Detection เชิงทฤษฎี |
| MIT 6.869 | Advances in Computer Vision | Lecture Notes ฟรี |
| UC Berkeley CS280 | Computer Vision | Slides เกี่ยวกับ Image Gradients |

---

## 5. บทเรียนออนไลน์ — Morphological Operations

### 🌐 เว็บไซต์บทเรียนชั้นนำ

| แหล่ง | URL | เนื้อหาหลัก | ระดับ |
|---|---|---|---|
| **LearnOpenCV** — Morphological Operations | [learnopencv.com — Morphological Ops](https://learnopencv.com/morphological-operations-using-opencv/) | Erosion, Dilation, Opening, Closing ครบ พร้อม Python code | ⭐⭐⭐ |
| **PyImageSearch** — Morphological Operations | [pyimagesearch.com](https://pyimagesearch.com) | คู่มือ Morphological Operations อย่างละเอียด | ⭐⭐⭐ |
| **GeeksforGeeks** — Morphological Transformations | [geeksforgeeks.org — Morphology](https://www.geeksforgeeks.org/python-opencv-morphological-operations/) | อธิบายทุก Operation พร้อมภาพ | ⭐⭐ |
| **OpenCV Official Tutorial** — Morphological Ops | [docs.opencv.org — Morphological Ops](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html) | Tutorial อย่างเป็นทางการ ครอบคลุมที่สุด | ⭐⭐⭐ |
| **Wikipedia** — Mathematical Morphology | [en.wikipedia.org — Mathematical morphology](https://en.wikipedia.org/wiki/Mathematical_morphology) | ทฤษฎีพื้นฐานและประวัติ | ⭐⭐ |

---

## 6. วิดีโอการสอน

### 🎥 YouTube — Edge Detection

| ช่อง / ผู้สอน | ชื่อวิดีโอ | เนื้อหา | ลิงก์ค้นหา |
|---|---|---|---|
| **Computerphile** | "Canny Edge Detector" | อธิบายหลักการ Canny อย่างเข้าใจง่าย พร้อมภาพประกอบ | ค้นหา: `Computerphile Canny Edge Detector` |
| **Computerphile** | "Finding the Edges (Sobel Operator)" | อธิบาย Sobel Operator ด้วย Visual | ค้นหา: `Computerphile Sobel Operator` |
| **3Blue1Brown** | Calculus Series (Derivatives) | ความเข้าใจพื้นฐานเรื่องอนุพันธ์ (สำคัญสำหรับ Image Gradient) | ค้นหา: `3Blue1Brown derivatives` |
| **Sentdex** | OpenCV with Python Tutorials | Playlist สอน OpenCV ตั้งแต่พื้นฐาน มี Edge Detection | ค้นหา: `Sentdex OpenCV Python edge detection` |
| **Murtaza's Workshop** | OpenCV Course | สอน OpenCV เชิงปฏิบัติ ครอบคลุม Edge + Morphology | ค้นหา: `Murtaza Workshop OpenCV` |
| **freeCodeCamp** | OpenCV Course – Full Tutorial | วิดีโอ 4+ ชั่วโมง ครอบคลุม Edge Detection | ค้นหา: `freeCodeCamp OpenCV course` |

### 🎥 YouTube — Morphological Operations

| ช่อง / ผู้สอน | ชื่อวิดีโอ | เนื้อหา | ลิงก์ค้นหา |
|---|---|---|---|
| **Sentdex** | "Morphological Transformations" | สอน Erosion, Dilation, Opening, Closing ด้วย OpenCV | ค้นหา: `Sentdex morphological transformations OpenCV` |
| **PyImageSearch (YouTube)** | Morphological Operations | อธิบาย Morphology Operations เชิงปฏิบัติ | ค้นหา: `PyImageSearch morphological operations` |

---

## 7. แหล่งภาพประกอบสไลด์

### 🖼️ ภาพทดสอบมาตรฐาน (Standard Test Images)

| แหล่ง | URL | ภาพที่แนะนำ | ใช้สำหรับ |
|---|---|---|---|
| **USC-SIPI Image Database** | [sipi.usc.edu/database](https://sipi.usc.edu/database/) | Lena, Cameraman, Peppers, Baboon, Boat | ✅ ภาพมาตรฐานที่ใช้ใน Paper วิจัย เหมาะแสดงผล Edge Detection |
| **OpenCV Sample Images** | มาพร้อมกับ OpenCV Installation | `cv2.samples.findFile()` | ✅ ภาพตัวอย่างจาก OpenCV โดยตรง |
| **scikit-image Data Module** | `from skimage import data` | `data.camera()`, `data.coins()`, `data.text()` | ✅ ภาพทดสอบพร้อมใช้ใน Python |

### 🖼️ ภาพถ่ายฟรีสำหรับ Demo (Free Stock Photos)

| แหล่ง | URL | ประเภท | ลิขสิทธิ์ |
|---|---|---|---|
| **Unsplash** | [unsplash.com](https://unsplash.com) | ภาพถ่ายคุณภาพสูง (แนะนำค้นหา: electronics, PCB, coins, fruits) | ฟรี ใช้เชิงพาณิชย์ได้ |
| **Pexels** | [pexels.com](https://pexels.com) | ภาพถ่ายและวิดีโอฟรี | ฟรี ใช้เชิงพาณิชย์ได้ |
| **Pixabay** | [pixabay.com](https://pixabay.com) | ภาพถ่าย กราฟิก Vector | ฟรี Royalty-free |
| **Wikimedia Commons** | [commons.wikimedia.org](https://commons.wikimedia.org) | ภาพทางวิทยาศาสตร์และการศึกษา | CC License (ระบุแหล่งที่มา) |

### 🖼️ แผนภาพและไดอะแกรมสำหรับสไลด์

| หัวข้อ | แหล่งที่แนะนำ | รายละเอียด |
|---|---|---|
| **Sobel Kernel Diagram** | [Wikipedia — Sobel Operator](https://en.wikipedia.org/wiki/Sobel_operator) | แผนภาพ Sobel Kernel 3×3 ทั้ง X และ Y พร้อมตัวอย่างภาพ |
| **Canny Pipeline Flowchart** | [Wikipedia — Canny Edge Detector](https://en.wikipedia.org/wiki/Canny_edge_detector) | Flowchart 5 ขั้นตอนของ Canny พร้อมภาพตัวอย่างแต่ละขั้น |
| **Laplacian Kernel** | OpenCV Docs + Wikipedia | แผนภาพ Kernel 3×3 พร้อมกราฟ Zero-Crossing |
| **Morphological Operations** | [Wikipedia — Mathematical Morphology](https://en.wikipedia.org/wiki/Mathematical_morphology) | ไดอะแกรม Erosion, Dilation, Opening, Closing |
| **Structuring Element Shapes** | OpenCV Docs | ภาพเปรียบเทียบ RECT, ELLIPSE, CROSS |

### 🖼️ คำแนะนำการค้นหาภาพสำหรับแต่ละสไลด์

| สไลด์ | คำค้นหาที่แนะนำ (Unsplash/Pexels) | ใช้เพื่อ |
|---|---|---|
| สไลด์ที่ 1: Title Slide | `electronic components conveyor belt`, `PCB circuit board inspection` | ภาพเปรียบเทียบภาพจริง vs ภาพเส้นขอบ |
| สไลด์ที่ 3: Edge คืออะไร? | `apple on white table`, `single fruit white background` | ภาพแอปเปิ้ลสำหรับแสดง Intensity Profile |
| สไลด์ที่ 4: Image Gradient | `gradient black to white`, `abstract gradient` | ภาพไล่สีจากดำไปขาว |
| สไลด์ที่ 5-8: Edge Detection | `building architecture`, `cityscape`, `coins on table` | ภาพที่มีขอบชัดเจนสำหรับ Demo |
| สไลด์ที่ 9-14: Morphology | `binary image noise`, `fingerprint scan`, `medical x-ray` | ภาพ Binary ที่มี Noise / รูโหว่ |
| สไลด์ที่ 15: Pipeline | `PCB board`, `electronic circuit` | ภาพแผงวงจรสำหรับ Pipeline Demo |

---

## 8. เครื่องมือ Interactive และ Demo ออนไลน์

### 🎮 Interactive Demos

| เครื่องมือ | URL | คำอธิบาย | ใช้สำหรับ |
|---|---|---|---|
| **Image Kernels Explained Visually** | [setosa.io/ev/image-kernels](https://setosa.io/ev/image-kernels/) | 🌟 **แนะนำอย่างยิ่ง!** แสดง Kernel Convolution แบบ Interactive ลากเลื่อนดูผลลัพธ์ได้ทันที | เหมาะใช้ Demo ในห้องเรียนเพื่ออธิบาย Kernel |
| **OpenCV.js Demos** | [docs.opencv.org — OpenCV.js](https://docs.opencv.org/4.x/d5/d10/tutorial_js_root.html) | รัน OpenCV บน Browser ได้โดยตรง | ทดสอบ Edge Detection แบบ Live |
| **Google Colab — Edge Detection** | ค้นหา: `edge detection OpenCV colab notebook` | Jupyter Notebook พร้อมรัน ไม่ต้องติดตั้งอะไร | ให้นักศึกษาลองรัน Online |

### 📓 Jupyter Notebooks บน GitHub

| Repository | คำอธิบาย | URL |
|---|---|---|
| OpenCV Python Tutorials | Notebook ครบทุกหัวข้อ OpenCV | ค้นหา: `github opencv python tutorials notebooks` |
| Digital Image Processing Notebooks | Notebook เรียน Image Processing | ค้นหา: `github digital image processing jupyter` |
| Edge Detection Comparison | เปรียบเทียบ Edge Detector หลายแบบ | ค้นหา: `github edge detection comparison opencv` |

---

## 9. แหล่งข้อมูลภาษาไทย

### 🇹🇭 บทความและบทเรียนภาษาไทย

| แหล่ง | เนื้อหา | คำค้นหา |
|---|---|---|
| **BorntoDev** | ชุมชนโปรแกรมเมอร์ไทย มีบทความ OpenCV | ค้นหา: `BorntoDev OpenCV Edge Detection` |
| **Medium (ผู้เขียนไทย)** | บทความ Canny Edge Detection ภาษาไทยหลายบทความ | ค้นหา: `Medium Canny Edge Detection ภาษาไทย` |
| **บล็อกชุมชน AI Thailand** | บทความเกี่ยวกับ Computer Vision | ค้นหา: `การตรวจจับขอบภาพ OpenCV ภาษาไทย` |

### 🇹🇭 วิดีโอภาษาไทย

| คำค้นหาบน YouTube | เนื้อหาที่คาดว่าจะพบ |
|---|---|
| `สอน OpenCV Edge Detection ภาษาไทย` | วิดีโอสอนหาเส้นขอบด้วย OpenCV |
| `Canny Edge Detection Python ไทย` | สอน Canny ภาษาไทย |
| `Morphological Operations OpenCV ภาษาไทย` | สอน Erosion, Dilation ภาษาไทย |
| `การประมวลผลภาพดิจิทัล สอนไทย` | บทเรียน Image Processing ทั่วไป |

### 🇹🇭 Open Courseware มหาวิทยาลัยไทย

| มหาวิทยาลัย | หมายเหตุ |
|---|---|
| จุฬาลงกรณ์มหาวิทยาลัย | บางรายวิชามี Open Courseware เรื่อง Image Processing |
| มหาวิทยาลัยเกษตรศาสตร์ | เอกสารประกอบการสอน Computer Vision |
| สถาบัน NECTEC / NSTDA | บทความวิจัยและเอกสารเทคนิคภาษาไทย |

---

## 10. สรุปเนื้อหาหลักสำหรับทำสไลด์

### 📋 โครงสร้างเนื้อหา Week 4 (16 สไลด์)

เนื้อหาแบ่งเป็น **4 หมวด** ดังนี้:

---

### หมวดที่ 1: แนวคิดพื้นฐานของเส้นขอบ (Edge Fundamentals)

#### สไลด์ที่ 1: Title Slide
- **เนื้อหา:** ชื่อหัวข้อ "การสกัดเส้นขอบและการดัดแปลงสัณฐานวิทยา (Edge Detection & Morphological Operations)"
- **ภาพประกอบ:** ภาพเปรียบเทียบ — ภาพถ่ายจริง (ซ้าย) vs ภาพเส้นขอบ (ขวา)
- **แหล่งภาพ:** ถ่ายภาพชิ้นส่วนอิเล็กทรอนิกส์แล้วรันผ่าน Canny / หรือใช้ภาพจาก Unsplash ค้นหา `PCB circuit board`

#### สไลด์ที่ 2: วัตถุประสงค์การเรียนรู้
- **เนื้อหา:** 5 วัตถุประสงค์:
  1. อธิบายแนวคิด Image Gradient
  2. เปรียบเทียบ Sobel, Laplacian, Canny
  3. เข้าใจ Morphological Operations
  4. เลือก Structuring Element ที่เหมาะสม
  5. ประยุกต์ Pipeline Edge + Morphology
- **ภาพประกอบ:** ไอคอน 1-5 เรียงแนวตั้ง

#### สไลด์ที่ 3: เส้นขอบ (Edge) คืออะไร?
- **เนื้อหา:** Edge = จุดเปลี่ยนแปลงความสว่างอย่างฉับพลัน
- **ภาพประกอบ:** ภาพแอปเปิ้ล + กราฟ Intensity Profile ตัดขวาง
- **แหล่งภาพ:** Unsplash ค้นหา `apple white background` / สร้างกราฟด้วย Matplotlib
- **อ้างอิง:** Gonzalez & Woods, Ch.10.1

#### สไลด์ที่ 4: Image Gradient — ความชันของความสว่าง
- **เนื้อหา:** สมการ Gradient (Gx, Gy), Magnitude, Direction
- **สูตรสำคัญ:**
  - $G_x = \frac{\partial I}{\partial x}$, $G_y = \frac{\partial I}{\partial y}$
  - $|G| = \sqrt{G_x^2 + G_y^2}$
  - $\theta = \arctan(G_y / G_x)$
- **ภาพประกอบ:** ภาพไล่สีดำ-ขาว + ภาพ Step Edge พร้อมกราฟอนุพันธ์
- **อ้างอิง:** Gonzalez & Woods, Ch.10.2

---

### หมวดที่ 2: อัลกอริทึมการตรวจจับเส้นขอบ (Edge Detection Algorithms)

#### สไลด์ที่ 5: Sobel Filter
- **เนื้อหา:** Kernel 3×3 แยก X, Y / First-order Derivative / Convolution
- **สูตรสำคัญ:** Sobel Kernel X & Y matrices
- **โค้ด OpenCV:** `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)`
- **ภาพประกอบ:** ภาพต้นฉบับ → Sobel X → Sobel Y → Magnitude
- **อ้างอิง:** Sobel & Feldman (1968), OpenCV Docs — Image Gradients

#### สไลด์ที่ 6: Laplacian Filter
- **เนื้อหา:** Second-order Derivative / ตรวจจับทุกทิศทาง / ไวต่อ Noise
- **สูตรสำคัญ:** $\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$
- **โค้ด OpenCV:** `cv2.Laplacian(blurred, cv2.CV_64F)`
- **ภาพประกอบ:** กราฟ 3 ชั้น — I(x), I'(x), I''(x)
- **อ้างอิง:** Marr & Hildreth (1980), OpenCV Docs

#### สไลด์ที่ 7: Canny Edge Detection ⭐
- **เนื้อหา:** 5 ขั้นตอน (Gaussian → Gradient → NMS → Double Threshold → Hysteresis)
- **โค้ด OpenCV:** `cv2.Canny(gray, 50, 150)`
- **ภาพประกอบ:** Flowchart 5 ขั้น + ภาพผลลัพธ์แต่ละขั้น
- **แหล่งภาพ Flowchart:** Wikipedia — Canny Edge Detector
- **อ้างอิง:** **Canny, J.F. (1986)**

#### สไลด์ที่ 8: ตารางเปรียบเทียบ Sobel vs Laplacian vs Canny
- **เนื้อหา:** ตาราง 7 เกณฑ์เปรียบเทียบ
- **ภาพประกอบ:** ภาพผลลัพธ์ 3 วิธีจากภาพเดียวกัน (สร้างด้วย Python)
- **อ้างอิง:** ตำราหลัก + OpenCV Docs

---

### หมวดที่ 3: สัณฐานวิทยาของภาพ (Morphological Operations)

#### สไลด์ที่ 9: Structuring Element
- **เนื้อหา:** MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS / การเลือกขนาด
- **โค้ด OpenCV:** `cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))`
- **ภาพประกอบ:** ตาราง Kernel 3 แบบ (0 และ 1)
- **อ้างอิง:** Gonzalez & Woods, Ch.9.1 / Serra (1982)

#### สไลด์ที่ 10: Erosion (การกัดกร่อน)
- **เนื้อหา:** กฎ: ขาวก็ต่อเมื่อทุกตัวใต้ Kernel ขาว / ขอบหดเข้า
- **โค้ด OpenCV:** `cv2.erode(binary, kernel, iterations=1)`
- **ภาพประกอบ:** Before/After Erosion + Animation ของ Kernel เลื่อน
- **แหล่งภาพ:** สร้างจาก Python หรือใช้จาก OpenCV Tutorial

#### สไลด์ที่ 11: Dilation (การขยาย)
- **เนื้อหา:** กฎ: ขาวถ้ามีอย่างน้อย 1 ตัวใต้ Kernel ขาว / ขอบขยายออก
- **โค้ด OpenCV:** `cv2.dilate(binary, kernel, iterations=1)`
- **ภาพประกอบ:** Before/After Dilation

#### สไลด์ที่ 12: Opening (Erosion → Dilation)
- **เนื้อหา:** กำจัดจุดขาวเล็ก ๆ (White Noise)
- **โค้ด OpenCV:** `cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)`
- **ภาพประกอบ:** Flowchart + ภาพ 3 ขั้น (Original → Erosion → Dilation)
- **เปรียบเทียบจำ:** "เปิดประตูปัดฝุ่นออก"

#### สไลด์ที่ 13: Closing (Dilation → Erosion)
- **เนื้อหา:** อุดรูดำเล็ก ๆ ในวัตถุขาว
- **โค้ด OpenCV:** `cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)`
- **ภาพประกอบ:** Flowchart + ภาพ 3 ขั้น (Original → Dilation → Erosion)
- **เปรียบเทียบจำ:** "ปิดรอยรั่วอุดรูให้เรียบ"

#### สไลด์ที่ 14: Morphological Gradient
- **เนื้อหา:** Dilation – Erosion = เส้นขอบ / Closed Contour
- **โค้ด OpenCV:** `cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)`
- **ภาพประกอบ:** สมการภาพ Dilated – Eroded = Gradient
- **อ้างอิง:** Haralick et al. (1987)

---

### หมวดที่ 4: Pipeline รวมและการประยุกต์ใช้งาน

#### สไลด์ที่ 15: ภาพรวม Pipeline
- **เนื้อหา:** Grayscale → Blur → Canny → Dilation → Closing → Binary Mask
- **ภาพประกอบ:** Flowchart แนวนอน พร้อมภาพตัวอย่างทุกขั้น + ชื่อฟังก์ชัน OpenCV บนลูกศร
- **แหล่งภาพ:** สร้างด้วย Python จากภาพ PCB (Unsplash ค้นหา `PCB board`)
- **อ้างอิง:** สรุปจากทุกหัวข้อ

#### สไลด์ที่ 16: สรุปและเตรียมตัว LAB 4
- **เนื้อหา:** สรุป 4 Concepts + โจทย์ LAB 4: Custom Object Edge Extraction
- **ภาพประกอบ:** กรอบสี 4 กรอบ (Edge Detection, Morphology Basics, Opening/Closing, Pipeline)

---

## 📌 รายการอ้างอิงฉบับย่อ (Quick Reference List)

### ตำรา
1. Gonzalez, R.C. & Woods, R.E. (2018). *Digital Image Processing* (4th ed.). Pearson.
2. Szeliski, R. (2022). *Computer Vision: Algorithms and Applications* (2nd ed.). Springer. [ฟรี PDF](https://szeliski.org/Book/)
3. Davies, E.R. (2018). *Computer Vision: Principles, Algorithms, Applications, Learning* (5th ed.). Academic Press.

### บทความวิจัย
4. Canny, J.F. (1986). A Computational Approach to Edge Detection. *IEEE TPAMI*, 8(6), 679–698.
5. Sobel, I. & Feldman, G. (1968). An Isotropic 3×3 Image Gradient Operator. Stanford AI Project.
6. Marr, D. & Hildreth, E. (1980). Theory of Edge Detection. *Proc. Royal Society of London*, 207, 187–217.
7. Haralick, R.M., Sternberg, S.R. & Zhuang, X. (1987). Image Analysis Using Mathematical Morphology. *IEEE TPAMI*, 9(4), 532–550.
8. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.

### เว็บไซต์
9. OpenCV Documentation — Image Gradients: https://docs.opencv.org/4.x/d5/d0f/tutorial_py_gradients.html
10. OpenCV Documentation — Canny Edge: https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
11. OpenCV Documentation — Morphological Ops: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
12. LearnOpenCV — Edge Detection: https://learnopencv.com/edge-detection-using-opencv/
13. LearnOpenCV — Morphological Operations: https://learnopencv.com/morphological-operations-using-opencv/
14. scikit-image — Edge Examples: https://scikit-image.org/docs/stable/auto_examples/
15. Image Kernels Interactive: https://setosa.io/ev/image-kernels/

### ภาพประกอบ
16. USC-SIPI Image Database: https://sipi.usc.edu/database/
17. Wikipedia — Sobel Operator: https://en.wikipedia.org/wiki/Sobel_operator
18. Wikipedia — Canny Edge Detector: https://en.wikipedia.org/wiki/Canny_edge_detector
19. Wikipedia — Mathematical Morphology: https://en.wikipedia.org/wiki/Mathematical_morphology
20. Unsplash (ภาพถ่ายฟรี): https://unsplash.com
21. Pexels (ภาพถ่ายฟรี): https://pexels.com
22. Pixabay (ภาพถ่ายฟรี): https://pixabay.com

---

> [!IMPORTANT]
> **การใช้ภาพประกอบในสไลด์:**
> - ภาพจาก **Unsplash, Pexels, Pixabay** ใช้ได้ฟรีทั้งเชิงการศึกษาและเชิงพาณิชย์
> - ภาพจาก **Wikipedia** ส่วนใหญ่เป็น Creative Commons ควรระบุแหล่งที่มา
> - ภาพจาก **ตำรา/Paper** ควรระบุ citation ในสไลด์เสมอ
> - **แนะนำ:** สร้างภาพ Before/After ด้วยตนเองโดยรัน Python code จาก tutorial — จะได้ภาพที่ตรงกับเนื้อหาที่สอนที่สุด
