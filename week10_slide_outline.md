# โครงร่างสไลด์นำเสนอ - สัปดาห์ที่ 10
## การปรับใช้โมเดลสำเร็จรูปและการสั่งทำงานโมเดลบน OpenCV (Transfer Learning & ONNX Export)

---

### Slide 1: หน้าปก (Title Slide)
* **หัวข้อ:** Transfer Learning, ONNX Model Export, and OpenCV DNN Engine
* **วิชา:** การประมวลผลภาพดิจิทัล (Digital Image Processing)
* **สัปดาห์ที่ 10:** จาก PyTorch สู่การรัน Production บน OpenCV

---

### Slide 2: ปัญหาของการสร้างโมเดลจากศูนย์ (Why Transfer Learning?)
* การสร้าง CNN จากศูนย์ต้องใช้ภาพหลักหมื่น-แสนภาพ และการเทรนหลายวัน
* **Transfer Learning Solution:**
  * นำค่าน้ำหนักที่ผ่านการฝึกฝนจาก ImageNet (1.4 ล้านภาพ, 1,000 คลาส) มาใช้ต่อ
  * แช่แข็ง (Freeze) Feature Extractor (Convolutional Layers)
  * ปรับแต่งเฉพาะ Classifier Head (Fully Connected Layer) สำหรับงานของเรา

---

### Slide 3: สถาปัตยกรรม MobileNetV3 (Lightweight Neural Network)
* ออกแบบมาสำหรับอุปกรณ์ขอบเขตทรัพยากรจำกัด (Edge Devices / CPUs)
* เทคโนโลยี **Depthwise Separable Convolution:**
  * ลดจำนวนการคำนวณ (FLOPs) และขนาดไฟล์โมเดลลง 80–90% เมื่อเทียบกับ ResNet50
  * ค่า Accuracy ใกล้เคียงกันแต่รันเรียลไทม์บน CPU ได้สบาย

---

### Slide 4: มาตรฐาน ONNX (Open Neural Network Exchange)
* **ปัญหา:** โมเดลที่เทรนด้วย PyTorch มักต้องพึ่งพา PyTorch Environment ในการทำ Inference (ขนาดใหญ่ 2-5 GB)
* **ทางออกด้วย ONNX:**
  * ONNX เป็นฟอร์แมตกลางที่แปลง Computational Graph ออกเป็นไฟล์ไบนารี `.onnx`
  * สามารถนำไปเปิดรันด้วย OpenCV, TensorRT, DirectML, หรือ ONNX Runtime บนทุกภาษา (C++, Python, C#)

---

### Slide 5: การทำงานร่วมกับ OpenCV DNN Module
* OpenCV มาพร้อมกับเอนจิน DNN ในตัว โดยไม่ต้องลง PyTorch บนเครื่องปลายทาง
* **ขั้นตอนการทำ Inference ใน OpenCV:**
  1. `net = cv2.dnn.readNetFromONNX("model.onnx")`
  2. `blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), (0,0,0), swapRB=True)`
  3. `net.setInput(blob)`
  4. `preds = net.forward()`

---

### Slide 6: สรุปปฏิบัติการ LAB 10
* เทรน MobileNetV3 คัดแยกคลาส custom ด้วย PyTorch
* ส่งออกไฟล์ `mobilenet_v3_custom.onnx`
* เขียนสคริปต์ OpenCV ดึงเฟรมจากกล้อง Webcam ประมวลผลและแสดง Class Name + Confidence Score บนวิดีโอสด
