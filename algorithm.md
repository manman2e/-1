# 古长城实景模型深度图生成算法

> ✅ **直接运行**：`python generate_depth_maps.py <输入模型> <输出目录>`

仓库中提供了完整的 Python 管道（参见 `great_wall_depth/` 包以及 `generate_depth_maps.py` 脚本），可在 PyCharm 中直接运行。以下文档保留了详细的数学背景与实现思路，用于帮助理解与二次开发。

## 代码结构速览

| 模块/脚本 | 作用 | 核心接口 |
| --- | --- | --- |
| `generate_depth_maps.py` | 命令行入口脚本，负责解析参数、构建 `PipelineConfig` 并调用整条流水线。 | `main()` |
| `great_wall_depth/__init__.py` | 暴露对外使用的公共 API，便于从包中导入配置和管道函数。 | `PipelineConfig`、`CameraConfig`、`run_pipeline` |
| `great_wall_depth/config.py` | 定义相机与流水线的所有可调参数，并提供校验、序列化等工具方法。 | `CameraConfig`、`PipelineConfig.validate()`、`PipelineConfig.from_paths()` |
| `great_wall_depth/io.py` | 处理输入/输出：读取点云或网格、保存深度图（PNG/NumPy）和 JSON 元数据。 | `load_geometry()`、`write_depth_png()` |
| `great_wall_depth/geometry.py` | 集中实现几何基础函数，例如长轴估计、坐标系转换、折线重采样和平滑。 | `estimate_long_axis()`、`resample_polyline()` |
| `great_wall_depth/centerline.py` | 在长轴坐标系下统计点云，生成平滑的中心线并返回本地/世界坐标。 | `compute_centerline()` |
| `great_wall_depth/segments.py` | 将全局点云按中心线切分成带重叠的段，便于分块渲染和并行处理。 | `split_into_segments()` |
| `great_wall_depth/cameras.py` | 基于中心线采样点和切向量生成 A 面、B 面、顶视的相机姿态。 | `generate_poses()` |
| `great_wall_depth/depth_renderer.py` | 提供轻量级的点云投影器，实现深度图的逐像素可见性计算。 | `render_depth_map()` |
| `great_wall_depth/pipeline.py` | 串联整套流程：加载数据、提取中心线、分段、渲染并写出结果。 | `run_pipeline()` |

这些模块均为普通 Python 代码，可在 PyCharm 中逐个调试。若需要在别的项目中复用，只需安装依赖并从 `great_wall_depth` 包导入对应函数。

## 快速使用指引

1. **准备环境**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows 请使用 .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   依赖中最关键的是 `open3d`（负责载入点云/网格）和 `imageio`（输出 16-bit 深度 PNG）。

2. **准备输入数据**
   将无人机三维实景模型存为 `PLY/PCD/OBJ/XYZ/LAS/...` 等支持的格式。例如 `great_wall_segment.ply`。

3. **运行命令行脚本**
   ```bash
   python generate_depth_maps.py great_wall_segment.ply outputs/
   ```
   常用可选参数：
   - `--voxel 0.15`：体素下采样尺寸（米）。
   - `--step 1.5`：沿中心线的采样间距（米）。
   - `--chunk-length 200` / `--chunk-overlap 20`：分段长度及重叠量。
   - `--width 1280 --height 720 --fov 75`：深度图分辨率与视场角。

   程序会在输出目录下生成：
   - `centerline_world.npy` / `centerline_local.npy`（可选，保存中间结果）；
   - `segment_xxx/A|B|Top/sample_yyyy_depth.npy/.png`（深度图和 NumPy 原始数组）；
   - `segment_xxx/..._meta.json`（每张深度图的相机姿态与内参）；
   - `summary.json`（总览，记录调用配置与各文件路径）。

4. **在 PyCharm 中运行**
   - 打开仓库后，配置好 Python 解释器（推荐使用上面创建的虚拟环境）。
   - 右键 `generate_depth_maps.py` → “Run 'generate_depth_maps'...”，在“Parameters”栏填入 `<输入路径> <输出目录> [可选参数]`。
   - 在“Run”窗口可实时查看日志，若需调试，可在包内任意模块上打断点。

5. **复用或扩展**
   - 从别的脚本导入 `from great_wall_depth import PipelineConfig, run_pipeline`，构造配置后直接调用。
   - 若需要替换某一步（例如中心线算法），可在保留同名函数接口的前提下自行实现，然后在 `PipelineConfig` 或 `pipeline.py` 中切换。

6. **合并输出为完整 A/B/Top 深度图**
   - 管道会在 `segment_xxx/A|B|Top/` 中保存沿中心线每个采样点的深度图切片。若希望得到“整段长城”的三张长条深度图，可使用仓库新增的 `combine_depth_maps.py` 脚本：
     ```bash
     python combine_depth_maps.py outputs/ merged_depths/ \
         --poses A B Top --axis horizontal --save-npy
     ```
   - 其中 `outputs/` 为原始流水线的输出目录，`merged_depths/` 是希望存放合成结果的新目录。
   - `--axis` 控制拼接方向（`horizontal` 表示把所有切片在水平方向排成一张很宽的图，`vertical` 则按行堆叠），`--poses` 可选择只合并某几个视角，`--save-npy` 会额外输出 NumPy 格式以便后续数值处理。
   - 脚本会读取 `summary.json`，按 `segment_index` 与 `sample_index` 自动排序所有切片，并生成：
     1. `combined_<pose>.png`：与单帧相同的 16-bit 深度图，只是尺寸更大；
     2. （可选）`combined_<pose>.npy`：合成后的浮点深度矩阵；
     3. `combined_<pose>_sources.json`：记录每个像素块对应的原始切片，方便溯源。

以上流程完成后，可继续阅读下文的算法细节章节，了解每个阶段的数学与实现原理。

## 1. 输入数据与全局参数
- `M`: 全局三维模型，采用稠密点云或三角网格表示，包含坐标 `(x, y, z)`，单位为米。
- `G`: 已知的重力方向向量，默认取世界坐标轴 `+Z` 方向。
- `L_chunk`: 切块长度上限（例如 300 m），用于控制每次处理的空间范围。
- `S_step`: 沿中心线的采样间隔（例如 2 m）。
- `H_cam`: 相机距地高度（例如 5 m），决定深度图视角。
- `FOV_h`, `FOV_v`: 水平/垂直视场角，用于计算相机内参。
- `R_out`: 输出深度图分辨率（宽、高）。

## 2. 预处理与切块
```
2.1 估计场景边界框 B_all = bounding_box(M)。
2.2 按长城走向的主轴（初始可使用 PCA 第一个主成分）将 B_all 沿 X 方向切分为若干区段 {B_k}，满足每个区段长度 ≤ L_chunk。
2.3 对每个区段提取局部子集 M_k = M ∩ B_k。
2.4 可选：对每个 M_k 执行噪声滤波、体素降采样 (voxel_size ≈ 0.05 m)。
```

## 3. 古长城主体分离
```
3.1 计算地面标高：对 M_k 做高度直方图或 RANSAC 平面拟合，得到地面高度 z_ground。
3.2 仅保留 z > z_ground + Δz_min 的点，得到 M_wall_k。
3.3 将 M_wall_k 投影到水平面得到二维栅格图 H_k(x, y) = max_z(M_wall_k)。
3.4 对 H_k 进行形态学闭运算，去除洞穴；再进行阈值化获得二值掩膜 mask_k。
```

## 4. 中心线提取与平滑
```
4.1 对 mask_k 运行骨架提取 (medial axis/skeletonization)，得到初始骨架 S_k。
4.2 将骨架点提升回三维：对 S_k 中每个 (x, y) 点，在 M_wall_k 中找到最高点的 z 值，得到 3D 点集 C_k。
4.3 沿骨架排序：利用最小生成树 + 主路径追踪，使点序列满足曲线连续。
4.4 使用三次 B 样条或低通滤波器对 C_k 拟合，得到平滑中心线曲线 γ_k(s)，其中 s ∈ [0, L_k]。
4.5 计算弧长参数化，记录曲线上均匀采样点 {γ_k(s_i)}，步长为 S_step。
```

## 5. 局部坐标系构建
对于中心线上的每个采样点 i：
```
5.1 计算切向向量 t_i = normalize(γ_k(s_{i+1}) - γ_k(s_{i-1}))。
5.2 定义上向量 u_i = normalize(G)。
5.3 计算侧向法向量 n_i = normalize(u_i × t_i)。
5.4 构建相机朝向：
     - A 面方向 d_A = n_i
     - B 面方向 d_B = -n_i
     - 顶视方向 d_T = -u_i
5.5 相机右手坐标：
     - r_A = normalize(t_i × d_A)
     - r_B = normalize(t_i × d_B)
     - r_T = normalize(d_T × t_i)
5.6 构建旋转矩阵 R_{view} = [r, up, -d]^T，其中 up = t_i 对于侧视，相机上方向与长城轴向一致。
```

## 6. 相机位置与投影矩阵
```
6.1 计算相机基点 p_i = γ_k(s_i) + h_offset * u_i，其中 h_offset = H_cam。
6.2 A/B 面相机可再向侧面偏移 δ = w_wall/2（长城半宽）以保证可见。
6.3 计算内参矩阵 K，使用焦距 f = 0.5 * R_out_width / tan(FOV_h/2)。
6.4 构建透视投影矩阵 P = K · [R | -R p_i]。
```

## 7. 深度渲染
```
7.1 对于每个视角 (A, B, 顶视)，执行以下步骤：
     a. 将局部点云/网格转换到相机坐标：X_cam = R (X - p_i)。
     b. 过滤 Z_cam ≤ 0 的点（在相机后方的点）。
     c. 计算像素坐标 (u, v) = project(K, X_cam)。
     d. 使用 Z 缓冲策略：对每个像素保留最小正 Z_cam 作为深度值。
     e. 输出深度图 D_i^{view}，未命中的像素填充为 0 或最大深度。
7.2 采用图像格式：16-bit PNG 或浮点 EXR。
7.3 保存伴随元数据：中心线参数 s_i、相机位置 p_i、朝向 d、内参 K、切块索引 k。
```

## 8. 后处理与质量控制
```
8.1 深度图平滑：使用双边滤波减小噪声但保留边缘。
8.2 孔洞填充：对深度图中空值执行邻域插值或多视角融合。
8.3 相邻帧一致性：比较相邻 s_i 的深度图差分，检测异常突变。
8.4 如果检测到遮挡严重区域，缩短 S_step 或增加倾斜视角（调整 d）。
```

## 9. 批处理框架伪代码
```python
for B_k in split_blocks(M, L_chunk):
    M_k = preprocess_block(M, B_k)
    M_wall_k = extract_wall_body(M_k)
    centerline = smooth_centerline(M_wall_k)
    for s in sample_along(centerline, step=S_step):
        pose_set = build_poses(centerline, s, G, H_cam)
        for pose in pose_set:  # A 面, B 面, 顶视
            depth = render_depth(M_k, pose, FOV_h, FOV_v, R_out)
            save_depth(depth, metadata)
```

## 10. 时间与资源评估
- 若每个切块约 300 m，10 km 数据量约为 34 个切块；按每块 5000 万点估计，需使用点云加速结构（KD-tree/Octree）优化投影。
- 深度图渲染可多线程或 GPU 并行，每块约数十秒至数分钟，视硬件与分辨率而定。
- 元数据体积较小，应集中管理（如 JSON/CSV）。

## 11. 验证与交付
- 随机抽取若干位置，使用真实高度测量或其他数据源验证深度图的准确性。
- 输出最终深度图集、相机姿态文件和中心线数据（如 GeoJSON/PLY）。
```
