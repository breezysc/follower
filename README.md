# POE2 Auto Follower + OCR System

《流放之路2》自动跟随 + OCR 触发系统。基于计算机视觉实现自动跟随队长、OCR 文字识别触发传送、模板匹配点击目标等功能。

## 功能特性

- **自动跟随 (WASD)** — 通过 HSV 颜色识别队长位置，自动控制键盘跟随
- **OCR 触发系统** — 识别聊天区域文字，匹配条件后自动执行传送操作
- **模板匹配** — 在指定 ROI 区域搜索目标图像并自动点击
- **可视化调试面板** — 实时显示检测状态、HSV 参数、候选目标等信息
- **ROI 区域管理** — 支持自定义多个 ROI 区域，灵活配置识别范围

## 项目结构

```
.
├── main.py                 # 主程序（跟随 + OCR 完整功能）
├── ocr_main.py             # OCR 独立调试程序
├── config.py               # 全局配置参数
│
├── capture/                # 屏幕捕获模块
│   ├── screen.py           # 全屏截图
│   └── window_capture.py   # 指定窗口截图
│
├── vision/                 # 计算机视觉模块
│   ├── leader_detector.py  # 队长检测器（HSV + 模板匹配）
│   ├── ocr_engine.py       # OCR 文字识别引擎
│   ├── roi_manager.py      # ROI 区域管理器
│   └── debugger.py         # 视觉调试工具
│
├── logic/                  # 逻辑控制模块
│   ├── tracker.py          # 目标追踪器
│   ├── rule_engine.py      # OCR 规则引擎
│   └── event_manager.py    # 事件管理器
│
├── control/                # 输入控制模块
│   ├── keyboard.py         # 键盘控制器
│   └── mouse_controller.py # 鼠标控制器
│
├── utils/                  # 工具函数
│   └── math_utils.py       # 数学计算工具
│
└── templates/              # 模板图片目录
    ├── leader_template.png     # 队长标记模板
    └── target_c_template.png   # 传送目标模板
```

## 环境依赖

- Python 3.10+
- OpenCV (`cv2`)
- NumPy
- pynput
- pytesseract / easyocr

## 快速开始

### 1. 安装依赖

```bash
pip install opencv-python numpy pynput
```

OCR 引擎请选择安装其一：

```bash
# 方案 A: Tesseract (需额外安装系统级 Tesseract)
pip install pytesseract

# 方案 B: EasyOCR (纯 Python，无需额外安装)
pip install easyocr
```

### 2. 运行主程序

```bash
python main.py
```

程序会自动查找标题为 **"Path of Exile 2"** 或 **"Path of Exile"** 的游戏窗口进行捕获。

### 3. 首次使用配置

| 步骤 | 操作 | 按键 |
|------|------|------|
| 1 | 框选队长标记区域保存模板 | `T` |
| 2 | 绘制 ROI_A（传送目标搜索区域） | `2` 拖拽 |
| 3 | 绘制 ROI_B（OCR 文字识别区域） | `2` 拖拽 |
| 4 | 绘制 ROI_C（二次点击区域） | `2` 拖拽 |
| 5 | 保存 ROI 配置 | `4` |
| 6 | 保存 HSV 配置 | `S` |

## 操作说明

### 主程序 (`main.py`)

| 按键 | 功能 |
|------|------|
| `ESC` | 退出程序 |
| `M` | 切换调试模式 / 实时模式 |
| `S` | 保存 HSV 配置 |
| `T` | 捕获队长模板 |
| `G` | 捕获目标 C 模板（ROI_A 内） |
| `H` | 捕获 ROI_C 模板（二次点击目标） |
| `1` | ROI 选择模式 |
| `2` | ROI 绘制模式 |
| `3` | 删除选中 ROI |
| `4` | 保存 ROI 配置 |
| `O` | 测试 ROI_B 的 OCR 识别 |
| `SPACE` | 手动点击目标 C |

### OCR 调试程序 (`ocr_main.py`)

| 按键 | 功能 |
|------|------|
| `1` | 选择 ROI 模式 |
| `2` | 绘制 ROI 模式 |
| `3` | 删除选中 ROI |
| `4` | 保存 ROI 配置 |
| `5` | 清空规则 |
| `6` | 添加自定义规则 |
| `G` / `T` | 手动截图保存模板 |
| `SPACE` | 点击检测到的目标 C |
| `ESC` | 退出 |

## 工作流程

### 自动跟随流程

```
捕获游戏画面
    ↓
HSV 颜色过滤（蓝色队长标记）
    ↓
轮廓检测 → 候选目标筛选
    ↓
模板匹配确认队长身份
    ↓
计算与屏幕中心的偏移量
    ↓
模拟 WASD 键盘输入跟随
```

### OCR 触发传送流程

```
持续监控 ROI_B 区域
    ↓
OCR 识别文字（如 "111"）
    ↓
规则引擎匹配触发条件
    ↓
在 ROI_A 区域模板匹配目标 C
    ↓
点击目标 C（传送门/传送点）
    ↓
等待 0.8 秒后点击 ROI_C（确认/二次目标）
    ↓
等待 3 秒后持续检查 5 秒目标是否仍存在
    ↓
若目标消失 → 发送 /clear 清理聊天
```

## 配置说明

`config.py` 关键参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `BLUE_LOWER` / `BLUE_UPPER` | HSV 蓝色范围 | `(90,100,100)` ~ `(130,255,255)` |
| `DEAD_ZONE` | 死区像素（不移动阈值） | 5 |
| `FOLLOW_DISTANCE_MIN` | 最小跟随距离 | 25 |
| `FOLLOW_DISTANCE_MAX` | 最大跟随距离 | 200 |
| `TICK_INTERVAL` | 帧间隔 (ms) | 120 |
| `DEBUG_MODE` | 0=调试模式, 1=实时模式 | 0 |
| `USE_ROI` | 是否限制检测区域 | False |
| `ROI_SIZE` | ROI 区域大小 | 800 |

## ROI 区域定义

| ROI | 用途 | 说明 |
|-----|------|------|
| `ROI_A` | 传送目标搜索区 | 模板匹配查找传送门/传送点 |
| `ROI_B` | OCR 文字识别区 | 通常为聊天框区域，识别触发关键词 |
| `ROI_C` | 二次点击区域 | 传送后的确认按钮或二次目标 |

## 注意事项

1. **游戏窗口** — 程序优先捕获标题为 "Path of Exile 2" 的窗口，找不到则回退到 "Path of Exile" 或全屏捕获
2. **管理员权限** — 键盘鼠标模拟可能需要以管理员身份运行
3. **Tesseract 路径** — 如使用 pytesseract，需在 `ocr_engine.py` 中配置正确的 `tesseract_cmd` 路径
4. **模板质量** — 队长模板和目标模板建议选择特征明显、背景干净的区域
5. **ROI 比例坐标** — ROI 配置使用比例坐标，适配不同分辨率

## 许可证

MIT License
