# 3D Gaussian Splatting 项目记忆档案
## 📅 Date: 2025-01-28 | Session: Evaluation Debug & Student Network Design

---

## 🎯 **今日主要任务**
1. **修复eval_like_train.py的ZeroDivisionError**
2. **Debug低PSNR和视觉扭曲问题** 
3. **设计Student Network架构**
4. **发现数据集无train/test划分，重新设计策略**

---

## 🔧 **核心问题与解决方案**

### **Problem 1: ZeroDivisionError**
```python
# 原因：测试相机列表为空
test_cameras = scene.getTestCameras()  # 返回[]
# 解决：添加fallback机制
if len(test_cameras) == 0:
    test_subset = [train_cameras[idx] for idx in range(0, min(len(train_cameras), 10), 2)]
```

### **Problem 2: 超低PSNR (10-12 dB) + 视觉扭曲**
**症状**：
- 上下颠倒 (上下颠倒)
- 杂乱线条 (杂乱的线条)
- 卡车缺失 (卡车没有生成)
- 背景拉宽 (背景的树还拉宽了)
- 车牌错位 (车牌区域完全错位)

**根本原因分析**：
1. **相机参数错误** - `cameras.json`中的FoV计算有误
2. **Exposure参数丢失** - `gaussian_model.py`的bug
3. **渲染参数不匹配** - 与train.py的设置不一致

**解决路径**：
```python
# Fix 1: 直接从COLMAP读取准确相机参数
from scene.colmap_loader import read_intrinsics_binary, read_extrinsics_binary

# Fix 2: 修复exposure bug
# 在gaussian_model.py line 340:
if not use_train_test_exp:
    self.pretrained_exposures = None  # 只在不需要时才清空

# Fix 3: 确保所有渲染参数一致
SPARSE_ADAM_AVAILABLE = False  # 与train.py匹配
train_test_exp = False
resolution_scale = 2.0
```

### **Problem 3: PSNR提升历程**
```
初始: 10-12 dB (完全错误)
  ↓ 修复PSNR计算函数
中期: 12-15 dB (仍然很低)
  ↓ 修复相机参数和exposure
最终: 27.62-28.07 dB (接近合理水平)
```

---

## 📋 **关键代码修复记录**

### **文件1: eval_with_correct_cameras.py (最终工作版本)**
```python
def load_cameras_from_colmap(sparse_dir, images_dir, resolution_scale=1.0):
    """从COLMAP binary文件直接加载准确的相机参数"""
    # 关键：绕过cameras.json，直接读取二进制文件
    cameras = read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    images = read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
    
    # 关键：正确计算FoV
    fx, fy = intrinsics.params[0], intrinsics.params[1]
    FoVx = focal2fov(fx, width)
    FoVy = focal2fov(fy, height)  # 不能简单相等！
```

### **文件2: gaussian_model.py (Bug修复)**
```python
# Line 340 原代码：
self.pretrained_exposures = None  # 🔥 无条件清空，导致exposure丢失

# 修复后：
if not use_train_test_exp:
    self.pretrained_exposures = None  # 只在不需要时才清空
```

### **文件3: train.py参考设置**
```python
# 关键配置参数：
SPARSE_ADAM_AVAILABLE = False
train_test_exp = False  # cfg_args确认
resolution = -1  # 表示自动缩放
eval = False  # 所以没有test set
```

---

## 🎓 **经验教训总结**

### **🔥 关键教训**
1. **永远不要相信中间保存的参数文件** - `cameras.json`有计算错误
2. **直接从原始数据读取** - COLMAP binary files才是ground truth
3. **小心无条件的重置操作** - `self.pretrained_exposures = None`这种代码很危险
4. **渲染参数必须完全一致** - train.py是唯一标准
5. **视觉问题往往指向几何错误** - "上下颠倒"通常是相机参数问题

### **🔍 Debug策略**
1. **先修复简单错误**（ZeroDivisionError）
2. **然后追查根本原因**（相机参数）
3. **对比工作基准**（train.py）
4. **逐项验证参数**（FoV, exposure, resolution等）
5. **最后优化细节**（filtering等）

### **🛠️ 有效工具**
- `codebase_search`: 快速找到相关代码
- `grep_search`: 精确定位符号使用
- 小规模debug脚本验证假设
- 对比train.py作为golden reference

---

## 🧠 **技术洞察**

### **3DGS评估的核心要素**
1. **Camera Parameters**: FoVx, FoVy, R, T (最关键)
2. **Exposure Parameters**: 影响颜色准确性
3. **Resolution Scaling**: 影响图像尺寸和内参
4. **Rendering Settings**: SPARSE_ADAM_AVAILABLE, train_test_exp
5. **Background Color**: 影响透明度处理

### **PSNR计算标准**
```python
def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
# 注意：使用这个版本，不是cal_psnr
```

### **文件路径关系**
```
model_path/
├── gaussian_ball/iteration_XXX/gaussian_ball.ply  # 高斯参数
├── cameras.json                                   # ❌ 有错误
├── sparse/0/cameras.bin                          # ✅ 准确的内参
├── sparse/0/images.bin                           # ✅ 准确的外参
└── images/ -> 原始数据集/images/                   # 需要symlink
```

---

## 🚀 **Student Network设计思路**

### **问题发现**: 数据集无train/test划分
```python
eval=False  # cfg_args中确认
test_cam_names_list = []  # 导致没有test cameras
```

### **解决方案**: 自监督学习策略
```python
class SelfSupervisedGaussianStudent:
    """
    核心思路：
    1. Teacher: 已训练的3DGS (gaussian_ball.ply)
    2. Student: 神经网络 (稀疏点云 → 密集高斯)
    3. 监督信号: 渲染consistency + 几何consistency
    """
```

### **训练流程**
```
稀疏点云(10% Teacher) → Student Network → 完整高斯参数
                                            ↓
                                    与Teacher对比(Loss)
```

### **架构推荐**: PointNet++ + Cross-Attention
- **编码器**: PointNet++处理点云几何
- **融合**: Cross-attention结合图像特征
- **输出头**: 多任务预测各种高斯参数
- **激活**: 任务特定(Softplus for scale, Sigmoid for opacity)

---

## 📊 **性能基准**

### **当前PSNR水平**
- **eval_with_correct_cameras.py**: 27.62-28.07 dB
- **历史最佳(train.py)**: 33.83 dB
- **差距**: ~5.8 dB (可能原因：评估相机选择不同)

### **PLY文件对比**
```
gaussian_ball.ply (original):        27.62 dB  ✅ 最佳
filtered_alpha001.ply:               19.47 dB
filtered_alpha003.ply:               19.64 dB  
filtered_alpha005.ply:               19.56 dB
结论：过滤反而降低性能，原始模型最好
```

---

## 🔄 **下次对话重点**

### **立即可做的任务**
1. **实现Student Network的基础框架**
2. **加载Teacher 3DGS参数的工具函数**
3. **自监督数据生成器**
4. **渐进式训练策略**

### **需要解决的问题**
1. **5.8 dB PSNR差距的原因** - 可能是相机选择策略不同
2. **Student网络的具体架构细节** - PointNet++的完整实现
3. **训练数据的规模** - 从Teacher生成多少样本合适
4. **评估指标** - 除了PSNR还需要什么指标

### **技术方向**
1. **知识蒸馏**: Teacher → Student的参数传递
2. **渐进式学习**: 模仿3DGS的densification过程  
3. **多视角一致性**: 利用现有cameras做consistency check
4. **实时推理**: Student网络的推理速度优化

---

## 💾 **重要文件清单**

### **Working Scripts** ✅
- `gaussian_exp/eval_with_correct_cameras.py` - 最终工作的评估脚本
- `gaussian_exp/student_network_design.py` - Student网络架构设计
- `gaussian_exp/student_self_supervised.py` - 自监督训练策略

### **Reference Files** 📚  
- `train.py` - 训练基准，所有参数的标准
- `scene/gaussian_model.py` - 高斯模型实现(已修复exposure bug)
- `scene/colmap_loader.py` - COLMAP数据读取工具
- `utils/graphics_utils.py` - focal2fov等工具函数

### **Data Paths** 📁
- **Model**: `/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/output/truck-150w/`
- **PLY**: `gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply`
- **Cameras**: `sparse/0/{cameras.bin, images.bin}`
- **Images**: `images/` (symlinked from original dataset)

---

## 🎯 **Action Items for Next Session**

### **High Priority** 🔥
1. **实现完整的PLY加载器** - 从gaussian_ball.ply提取Teacher参数
2. **设计数据采样策略** - 稀疏输入的最佳采样方法
3. **验证渲染器集成** - 确保Student输出可以正确渲染

### **Medium Priority** ⚡
1. **实现PointNet++的完整版本** - 目前只有占位符
2. **设计训练循环** - batch processing, logging等
3. **添加更多评估指标** - SSIM, LPIPS等

### **Future Work** 🌟
1. **多场景泛化** - 在其他数据集上测试
2. **实时推理优化** - 模型压缩和加速
3. **交互式编辑** - 基于Student网络的实时编辑工具

---

## 🧩 **提示词模板 (For Future Sessions)**

```
我是在做3D Gaussian Splatting的项目。之前我们已经：

1. 修复了eval_like_train.py的ZeroDivisionError和低PSNR问题(从10dB提升到28dB)
2. 发现数据集没有train/test划分(eval=False)
3. 设计了自监督Student网络策略，用Teacher 3DGS做知识蒸馏
4. 关键教训：永远从COLMAP原始数据读取相机参数，不要信任中间文件

当前状态：
- eval_with_correct_cameras.py可以正确评估(28.07 dB PSNR)
- 设计了Student网络框架(PointNet++ + Cross-Attention)
- 需要实现自监督训练的具体代码

项目路径：/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs/
核心文件：output/truck-150w/gaussian_ball/iteration_994230_best_psnr/gaussian_ball.ply

请继续帮我实现Student网络的训练代码。
```

---

**📝 记忆档案创建完成！下次对话时请提供此文件，我将完全记住今天的所有工作内容。** 