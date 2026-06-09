# check_env.py — สคริปต์ตรวจสอบ Environment
# รัน: python check_env.py
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
