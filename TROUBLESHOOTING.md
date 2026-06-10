# 📋 คู่มือแก้ไขปัญหา Library Error สำหรับวิชา Digital Image Processing
## Troubleshooting Guide — dip_env Environment

> [!NOTE]
> เอกสารนี้รวบรวมวิธีการแก้ไขปัญหาที่พบบ่อยจากการติดตั้งไลบรารีในรายวิชา Digital Image Processing  
> ครอบคลุมหลายแนวทาง ตั้งแต่การใช้ Miniconda, pip + venv, ไปจนถึง Docker  
> **เลือกวิธีที่เหมาะกับสถานการณ์ของนักศึกษา**

---

## สารบัญ (Table of Contents)

1. [วิธีที่ 1: แก้ไขผ่าน Miniconda (แนะนำ)](#วิธีที่-1-แก้ไขผ่าน-miniconda-แนะนำ)
2. [วิธีที่ 2: ใช้ Python + pip + requirements.txt (ไม่ใช้ Conda)](#วิธีที่-2-ใช้-python--pip--requirementstxt-ไม่ใช้-conda)
3. [วิธีที่ 3: ใช้ Python venv + pip (Virtual Environment แบบ Built-in)](#วิธีที่-3-ใช้-python-venv--pip-virtual-environment-แบบ-built-in)
4. [วิธีที่ 4: ใช้ Docker Container (สำหรับผู้ใช้ขั้นสูง)](#วิธีที่-4-ใช้-docker-container-สำหรับผู้ใช้ขั้นสูง)
5. [ตารางสรุป Error ที่พบบ่อยและวิธีแก้ไข](#ตารางสรุป-error-ที่พบบ่อยและวิธีแก้ไข)
6. [คำถามที่พบบ่อย (FAQ)](#คำถามที่พบบ่อย-faq)

---

## ข้อมูล Environment ดั้งเดิม (Original Specification)

```yaml
name: dip_env
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.10.12
  - numpy=1.24.3
  - matplotlib=3.7.1
  - opencv=4.6.0
  - pytorch=2.0.1
  - torchvision=0.15.2
  - cpuonly
  - pip
  - pip:
    - ultralytics==8.0.196
    - mediapipe==0.10.7
```

---

## วิธีที่ 1: แก้ไขผ่าน Miniconda (แนะนำ)

> [!TIP]
> วิธีนี้เป็น **วิธีหลักที่แนะนำ** เพราะ Conda จะจัดการ binary dependency ระดับ OS ให้โดยอัตโนมัติ

### 1.1 ลบ Environment เก่าและสร้างใหม่ทั้งหมด (Nuclear Reset)

หากเกิด error ที่ซับซ้อนจนแก้ไม่ได้ ให้ลบ environment เก่าทิ้งแล้วสร้างใหม่:

```bash
# ปิดการใช้งาน environment ก่อน
conda deactivate

# ลบ environment เก่าทิ้ง
conda env remove -n dip_env

# ล้าง cache ที่อาจเสียหาย
conda clean --all -y

# สร้าง environment ใหม่จากไฟล์ environment.yml
conda env create -f environment.yml

# เปิดใช้งาน
conda activate dip_env

# ตรวจสอบว่าไลบรารีทำงานได้
python -c "import cv2; import torch; import numpy; print('OpenCV:', cv2.__version__); print('PyTorch:', torch.__version__); print('NumPy:', numpy.__version__)"
```

### 1.2 แก้ปัญหา Conda Resolve ช้าหรือหา Package ไม่เจอ

หากคำสั่ง `conda env create` ค้างนานหรือหา package ไม่เจอ:

```bash
# อัปเดตตัว conda เอง
conda update -n base conda -y

# ติดตั้ง libmamba solver (เร็วกว่า default solver มาก)
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba

# ลองสร้าง environment ใหม่อีกครั้ง
conda env create -f environment.yml
```

### 1.3 แก้ปัญหาเวอร์ชันขัดแย้ง (Version Conflict)

หากได้ error เกี่ยวกับ `UnsatisfiableError` หรือ `ResolvePackageNotFound`:

```bash
# สร้าง environment แบบไม่ล็อกเวอร์ชันย่อย (ให้ conda หาเวอร์ชันที่เข้ากันเอง)
conda create -n dip_env python=3.10 numpy matplotlib opencv pytorch torchvision cpuonly -c pytorch -c conda-forge -y

# เปิดใช้งาน
conda activate dip_env

# ติดตั้ง pip packages แยก
pip install ultralytics==8.0.196 mediapipe==0.10.7
```

### 1.4 แก้ปัญหา `conda activate` ไม่ทำงานใน PowerShell

```powershell
# เปิด PowerShell แบบ Administrator แล้วรัน
conda init powershell

# ปิดแล้วเปิด Terminal ใหม่
# หากยังไม่ได้ ลองตั้งค่า Execution Policy
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# ลองอีกครั้ง
conda activate dip_env
```

### 1.5 แก้ปัญหา `conda` command not found

```bash
# ตรวจสอบว่า Miniconda อยู่ใน PATH หรือไม่
# Windows — เปิด System Environment Variables > Path
# เพิ่ม path ของ Miniconda (ตัวอย่าง):
#   C:\Users\<ชื่อผู้ใช้>\miniconda3
#   C:\Users\<ชื่อผู้ใช้>\miniconda3\Scripts
#   C:\Users\<ชื่อผู้ใช้>\miniconda3\Library\bin

# หรือใช้ Anaconda Prompt แทน PowerShell
```

---

## วิธีที่ 2: ใช้ Python + pip + requirements.txt (ไม่ใช้ Conda)

> [!IMPORTANT]
> วิธีนี้เหมาะสำหรับนักศึกษาที่ **ไม่ต้องการติดตั้ง Miniconda** หรือ Conda ใช้ไม่ได้  
> ต้องดาวน์โหลด Python 3.10 มาติดตั้งแยกต่างหาก

### 2.1 ขั้นตอนติดตั้ง Python 3.10

1. ดาวน์โหลด Python 3.10 จาก [python.org/downloads](https://www.python.org/downloads/release/python-31012/)
2. ในหน้าต่างติดตั้ง **ต้องทำเครื่องหมายถูก** ✅ ที่ช่อง **"Add Python 3.10 to PATH"**
3. เลือก **"Customize installation"** > ติ๊กถูกทุกอย่าง > กด **Install**

### 2.2 สร้างไฟล์ requirements.txt

สร้างไฟล์ `requirements.txt` ในโฟลเดอร์โปรเจกต์ โดยมีเนื้อหาดังนี้:

```txt
numpy==1.24.3
matplotlib==3.7.1
opencv-python==4.6.0.66
torch==2.0.1+cpu
torchvision==0.15.2+cpu
ultralytics==8.0.196
mediapipe==0.10.7
```

### 2.3 ติดตั้งด้วย pip

```bash
# อัปเดต pip ให้เป็นเวอร์ชันล่าสุดก่อน
python -m pip install --upgrade pip

# ติดตั้ง PyTorch จาก official index (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ติดตั้งที่เหลือจาก requirements.txt (ข้าม torch/torchvision เพราะลงไปแล้ว)
pip install numpy==1.24.3 matplotlib==3.7.1 opencv-python==4.6.0.66 ultralytics==8.0.196 mediapipe==0.10.7
```

> [!WARNING]
> สำหรับ PyTorch เวอร์ชัน CPU บน pip ต้องระบุ `--index-url` พิเศษ  
> **ห้าม** ใช้ `pip install torch` ธรรมดา เพราะจะได้เวอร์ชัน CUDA ที่มีขนาดใหญ่มาก (~2.5GB)

### 2.4 ตรวจสอบการติดตั้ง

```bash
python -c "
import cv2
import torch
import numpy as np
import matplotlib
import mediapipe
import ultralytics

print('✅ OpenCV      :', cv2.__version__)
print('✅ PyTorch     :', torch.__version__)
print('✅ NumPy       :', np.__version__)
print('✅ Matplotlib  :', matplotlib.__version__)
print('✅ MediaPipe   :', mediapipe.__version__)
print('✅ Ultralytics :', ultralytics.__version__)
print()
print('🎉 ไลบรารีทั้งหมดพร้อมใช้งาน!')
"
```

---

## วิธีที่ 3: ใช้ Python venv + pip (Virtual Environment แบบ Built-in)

> [!TIP]
> วิธีนี้ **ไม่ต้องติดตั้งอะไรเพิ่ม** นอกจาก Python เอง เพราะ `venv` มาในตัว Python อยู่แล้ว  
> เหมาะสำหรับนักศึกษาที่ต้องการแยก environment โดยไม่ใช้ Conda

### 3.1 สร้าง Virtual Environment

```bash
# สร้าง venv ในโฟลเดอร์โปรเจกต์
python -m venv dip_venv
```

### 3.2 เปิดใช้งาน (Activate)

```powershell
# Windows (PowerShell)
.\dip_venv\Scripts\Activate.ps1

# Windows (CMD)
.\dip_venv\Scripts\activate.bat

# macOS / Linux
source dip_venv/bin/activate
```

> หาก PowerShell แจ้ง error `cannot be loaded because running scripts is disabled`:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3.3 ติดตั้ง Library

```bash
# อัปเดต pip
python -m pip install --upgrade pip

# ติดตั้ง PyTorch CPU
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu

# ติดตั้งที่เหลือ
pip install numpy==1.24.3 matplotlib==3.7.1 opencv-python==4.6.0.66 ultralytics==8.0.196 mediapipe==0.10.7
```

### 3.4 ตั้งค่า VS Code ให้ใช้ venv

1. กด `Ctrl + Shift + P` → พิมพ์ `Python: Select Interpreter`
2. เลือก Python ที่อยู่ในโฟลเดอร์ `dip_venv` เช่น:
   - `.\dip_venv\Scripts\python.exe` (Windows)
   - `./dip_venv/bin/python` (macOS/Linux)

### 3.5 เพิ่ม `.gitignore` (ป้องกัน push โฟลเดอร์ venv ขึ้น Git)

```bash
# เพิ่มบรรทัดนี้ในไฟล์ .gitignore
dip_venv/
```

---

## วิธีที่ 4: ใช้ Docker Container (สำหรับผู้ใช้ขั้นสูง)

> [!IMPORTANT]
> วิธีนี้เหมาะสำหรับกรณีที่ OS ของเครื่องมีปัญหาเข้ากันไม่ได้กับ library ไม่ว่าจะลงวิธีไหนก็ error  
> Docker จะสร้าง container แยกออกจาก OS หลัก ทำให้การติดตั้งสำเร็จ 100%

### 4.1 ติดตั้ง Docker Desktop

- ดาวน์โหลดจาก [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
- ติดตั้งและเปิด Docker Desktop

### 4.2 สร้าง Dockerfile

สร้างไฟล์ `Dockerfile` ในโฟลเดอร์โปรเจกต์:

```dockerfile
FROM python:3.10.12-slim

WORKDIR /app

# ติดตั้ง system dependencies ที่ OpenCV ต้องการ
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# ติดตั้ง Python libraries
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    matplotlib==3.7.1 \
    opencv-python-headless==4.6.0.66 \
    ultralytics==8.0.196 \
    mediapipe==0.10.7

RUN pip install --no-cache-dir \
    torch==2.0.1+cpu \
    torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY . .

CMD ["python", "main.py"]
```

### 4.3 Build และ Run

```bash
# Build image
docker build -t dip-env .

# Run container (mount โฟลเดอร์ปัจจุบัน)
docker run -it -v ${PWD}:/app dip-env python main.py
```

> [!WARNING]
> Docker ใช้ `opencv-python-headless` แทน `opencv-python`  
> ฟังก์ชัน `cv2.imshow()` จะ **ไม่ทำงาน** ใน Docker  
> ให้ใช้ `cv2.imwrite()` บันทึกภาพแทน หรือใช้ `matplotlib.pyplot.savefig()` แทนการแสดงผลหน้าต่าง

---

## ตารางสรุป Error ที่พบบ่อยและวิธีแก้ไข

| # | Error Message | สาเหตุ | วิธีแก้ไข |
|---|---------------|--------|-----------|
| 1 | `ModuleNotFoundError: No module named 'cv2'` | ยังไม่ได้ติดตั้ง OpenCV หรือ activate environment ไม่ถูก | ตรวจสอบว่า activate environment แล้ว → `conda activate dip_env` หรือ `pip install opencv-python==4.6.0.66` |
| 2 | `ImportError: DLL load failed while importing cv2` | ไฟล์ DLL ของ OpenCV เสียหาย หรือ Visual C++ Redistributable ไม่ได้ติดตั้ง | ติดตั้ง [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) แล้วลง OpenCV ใหม่ |
| 3 | `ERROR: Could not find a version that satisfies the requirement opencv==4.6.0` | ชื่อ package ใน pip ไม่ตรง (pip ใช้ `opencv-python` ไม่ใช่ `opencv`) | ใช้ `pip install opencv-python==4.6.0.66` (ชื่อต่างจาก conda) |
| 4 | `ModuleNotFoundError: No module named 'torch'` | PyTorch ยังไม่ได้ติดตั้ง หรือลง channel ผิด | ใช้ `pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu` |
| 5 | `RuntimeError: Numpy is not available` | NumPy เวอร์ชันไม่เข้ากับ PyTorch | `pip install numpy==1.24.3` |
| 6 | `error: Microsoft Visual C++ 14.0 or greater is required` | ต้องการ Build Tools ในการคอมไพล์ | ติดตั้ง [Build Tools for VS](https://visualstudio.microsoft.com/visual-cpp-build-tools/) หรือ **ใช้ Conda แทน** (Conda ดาวน์โหลด binary สำเร็จรูป ไม่ต้องคอมไพล์) |
| 7 | `ERROR: pip's dependency resolver does not currently consider all packages` | Dependency ขัดแย้งกัน | `pip install --force-reinstall <package>` หรือสร้าง environment ใหม่ |
| 8 | `CondaError: Run 'conda init' before 'conda activate'` | ยังไม่ได้ init conda กับ shell | รัน `conda init powershell` แล้วเปิด Terminal ใหม่ |
| 9 | `UnsatisfiableError` / `ResolvePackageNotFound` | Conda หาเวอร์ชันที่เข้ากันไม่เจอ | ลดความเข้มงวดของเวอร์ชัน เช่น ใช้ `python=3.10` แทน `python=3.10.12` |
| 10 | `OSError: [WinError 126] The specified module could not be found (mediapipe)` | ขาด dependency ของ MediaPipe | ติดตั้ง Visual C++ Redistributable + ลองใช้ `pip install mediapipe==0.10.7` ใหม่ |
| 11 | `AttributeError: module 'numpy' has no attribute ...` | NumPy เวอร์ชันใหม่เกินไป ไม่เข้ากับ library อื่น | `pip install numpy==1.24.3` (ล็อกเวอร์ชันตามสเปก) |
| 12 | `CUDA not available` (ไม่ใช่ error แต่เป็น warning) | ใช้ CPU-only build ซึ่งเป็นปกติ | ไม่ต้องแก้ไข — เราใช้ CPU-only build โดยตั้งใจ |

---

## คำถามที่พบบ่อย (FAQ)

### ❓ Q1: ควรเลือกวิธีไหน?

| สถานการณ์ | วิธีที่แนะนำ |
|-----------|-------------|
| ใช้ในห้องเรียนตามปกติ | **วิธีที่ 1** — Miniconda (ตามที่อาจารย์กำหนด) |
| Conda ติดตั้งไม่ได้ / ไม่อยากใช้ Conda | **วิธีที่ 2** — Python + pip + requirements.txt |
| อยากแยก environment แบบเบาๆ | **วิธีที่ 3** — Python venv + pip |
| ลงยังไงก็ error ทุกทาง / เครื่อง OS แปลก | **วิธีที่ 4** — Docker |

### ❓ Q2: ติดตั้งแล้วใช้ได้แต่พอ import mediapipe แล้ว error

```bash
# ลองลง protobuf เวอร์ชันเก่ากว่า
pip install protobuf==3.20.3

# แล้วลง mediapipe ใหม่
pip install --force-reinstall mediapipe==0.10.7
```

### ❓ Q3: `cv2.imshow()` ค้างหรือหน้าต่างไม่แสดง

- **ตรวจสอบ**: ต้องมี `cv2.waitKey(0)` หลัง `imshow()` เสมอ
- **ห้าม**: กดปุ่ม ❌ ปิดหน้าต่างภาพโดยตรง ให้กดแป้นพิมพ์แทน
- หากใช้ Jupyter Notebook: ใช้ `matplotlib` แสดงภาพแทน `cv2.imshow()`

### ❓ Q4: ต้อง push ไฟล์อะไรขึ้น Git บ้าง?

**ควร push:**
- `environment.yml` — ไฟล์กำหนด environment ของ Conda
- `requirements.txt` — ไฟล์กำหนด library ของ pip
- `Dockerfile` — (ถ้ามี) ไฟล์กำหนด Docker image
- ไฟล์โค้ด `.py`, `.ipynb`
- `TROUBLESHOOTING.md` — เอกสารนี้

**ห้าม push:**
- โฟลเดอร์ `dip_venv/` หรือ `dip_env/`
- โฟลเดอร์ `__pycache__/`
- ไฟล์ `.pyc`

---

## การ Push ขึ้น Git

### ขั้นตอนเตรียมไฟล์ก่อน Push

1. **สร้าง / อัปเดตไฟล์ `.gitignore`** ให้มีเนื้อหาอย่างน้อย:

```gitignore
# Python
__pycache__/
*.pyc
*.pyo

# Virtual Environments
dip_venv/
dip_env/
.conda/
*.egg-info/

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db

# Output files
*.onnx
*.pt
runs/
```

2. **ตรวจสอบว่ามีไฟล์ setup ครบ:**

```
📁 โฟลเดอร์โปรเจกต์/
├── environment.yml        ← สำหรับ Conda
├── requirements.txt       ← สำหรับ pip
├── Dockerfile             ← (Optional) สำหรับ Docker
├── .gitignore             ← ป้องกันไฟล์ไม่จำเป็นขึ้น Git
├── TROUBLESHOOTING.md     ← เอกสารนี้
├── main.py                ← โค้ดหลัก
└── ...
```

### คำสั่ง Git Push

```bash
# เพิ่มไฟล์ทั้งหมด
git add .

# ตรวจสอบว่าจะ push อะไรบ้าง
git status

# Commit
git commit -m "docs: เพิ่มเอกสารแก้ไขปัญหาและไฟล์ setup (environment.yml, requirements.txt)"

# Push ขึ้น remote
git push origin main
```

> [!CAUTION]
> ตรวจสอบ `git status` ก่อน push ทุกครั้ง เพื่อให้แน่ใจว่า **ไม่มีโฟลเดอร์ environment หรือไฟล์โมเดลขนาดใหญ่** ติดไปด้วย

---

## สคริปต์ทดสอบการตรวจสอบสุขภาพ Environment (Health Check)

บันทึกไฟล์นี้เป็น `check_env.py` แล้วรัน `python check_env.py` เพื่อตรวจสอบว่าทุกอย่างพร้อมใช้งาน:

```python
# check_env.py — สคริปต์ตรวจสอบ Environment
import sys

def check_library(name, import_name=None, version_attr="__version__"):
    """ตรวจสอบว่า library ติดตั้งแล้วหรือยัง"""
    if import_name is None:
        import_name = name
    try:
        mod = __import__(import_name)
        ver = getattr(mod, version_attr, "N/A")
        print(f"  ✅ {name:20s} → เวอร์ชัน {ver}")
        return True
    except ImportError as e:
        print(f"  ❌ {name:20s} → ไม่พบ! ({e})")
        return False

if __name__ == "__main__":
    print("=" * 55)
    print("  🔍 ตรวจสอบสุขภาพ Environment (Health Check)")
    print("=" * 55)
    print(f"\n  Python: {sys.version}")
    print(f"  Path:   {sys.executable}\n")
    print("-" * 55)

    results = []
    results.append(check_library("numpy",        "numpy"))
    results.append(check_library("matplotlib",   "matplotlib"))
    results.append(check_library("opencv",       "cv2",        "__version__"))
    results.append(check_library("pytorch",      "torch",      "__version__"))
    results.append(check_library("torchvision",  "torchvision"))
    results.append(check_library("ultralytics",  "ultralytics"))
    results.append(check_library("mediapipe",    "mediapipe"))

    print("-" * 55)
    passed = sum(results)
    total = len(results)
    if passed == total:
        print(f"\n  🎉 ผ่านทั้งหมด {passed}/{total} รายการ — พร้อมใช้งาน!")
    else:
        print(f"\n  ⚠️  ผ่าน {passed}/{total} รายการ — กรุณาติดตั้ง library ที่ขาดหายไป")
    print("=" * 55)
```

---

> **จัดทำโดย**: ระบบเอกสารวิชา Digital Image Processing  
> **อัปเดตล่าสุด**: มิถุนายน 2026
