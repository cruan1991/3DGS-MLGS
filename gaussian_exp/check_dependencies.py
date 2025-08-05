import shutil
import importlib

required_modules = [
    'torch',
    'scipy',
    'sklearn',
    'cv2',            # opencv-python
    'plyfile',
    'joblib',
    'diff_gaussian_rasterization',
    'simple_knn',
    'fused_sse'  # 可能会失败，部分项目不导入此模块
]

print("🔍 Checking Python packages:\n")
for mod in required_modules:
    try:
        importlib.import_module(mod)
        print(f"✅ {mod}")
    except ImportError:
        print(f"❌ {mod} not found")

print("\n🔍 Checking system tools:\n")

if shutil.which("colmap"):
    print("✅ colmap is in PATH")
else:
    print("❌ colmap not found in PATH")

print("\n✅ Done.")
