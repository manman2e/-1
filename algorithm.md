# 古长城实景模型深度图生成算法

> ✅ **直接运行**：`python generate_depth_maps.py <输入模型> <输出目录>`

仓库中提供了完整的 Python 管道（参见 `great_wall_depth/` 包以及 `generate_depth_maps.py` 脚本），可在 PyCharm 中直接运行。以下文档保留了详细的数学背景与实现思路，用于帮助理解与二次开发。

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
