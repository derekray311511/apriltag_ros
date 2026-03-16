# AprilTag CUDA 加速計畫

## 專案背景

- **應用場景**：自駕車多相機 AprilTag 偵測
- **Tag Family**：16h5（30 個 tag codes，4×4 bit payload）
- **硬體**：NVIDIA RTX 3090 / RTX 4090 / Pro 5000（桌上型 dGPU）
- **現狀瓶頸**：多顆相機同時跑時，threads=1 就已吃滿 CPU 所有核心
- **已排除方案**：
  - 多執行緒（CPU 已飽和）
  - 降解析度（需遠距離偵測）
  - Isaac ROS AprilTag（cuAprilTags 閉源，不支援 16h5）
  - wykvictor/AprilTag-GPU（無實際 CUDA 程式碼，名不副實）

## AprilTag 偵測 Pipeline 分析

```
[Image Input]
     │
     ▼
┌─────────────────────────────────────┐
│ Step 1: 預處理                       │  O(W×H)    GPU潛力: ★★★
│ grayscale 轉換 + Gaussian blur       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 2: 梯度計算                     │  O(W×H)    GPU潛力: ★★★★★
│ Sobel gradient + atan2               │  已有 OMP
│ 輸出: fimTheta, fimMag               │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 3: 邊緣聚類 (Union-Find)        │  O(N logN) GPU潛力: ★★★
│ 生成 edges → sort → merge clusters   │  sort 佔大頭
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 4: 聚類統計                     │  O(W×H)    GPU潛力: ★★
│ 收集每個 cluster 的像素座標與權重     │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 5: 線段擬合                     │  O(clusters) GPU潛力: ★★★
│ least-squares fit → Segment          │  各 cluster 獨立
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 6: 線段鏈接                     │  O(segments²) GPU潛力: ★
│ Gridder 空間雜湊 → 建立連接圖        │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 7: Quad 搜索                    │  DFS        GPU潛力: ★
│ 遞迴搜索 4-segment 封閉迴路          │  高度序列化
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 8: Quad 解碼                    │  O(quads)   GPU潛力: ★★★★
│ GrayModel → bit sampling → 查表      │  各 quad 獨立
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ Step 9: 去重                         │  O(det²)    GPU潛力: ★
│ 重複偵測消除                         │
└─────────────────────────────────────┘
```

## 實施策略：分三階段

---

### Phase 1：Profiling + OpenCV CUDA 預處理（1-2 週）

**目標**：量化各步驟耗時，用現成 API 加速前處理

#### 1.1 Detailed Profiling

AprilTag C library 內建 `timeprofile`，在 apriltag_ros 中設定 `profile: true` 即可啟用。

```yaml
# 在你的 tag config yaml 加入
profile: true
```

這會印出類似：
```
 0.000000 init
 0.004532 blur/decimate
 0.012847 gradient
 0.028134 clusters
 0.031245 fit lines
 0.033421 quads
 0.034102 decode
 0.034567 reconcile
```

**要收集的數據**：
- 各步驟在不同解析度下的耗時（1080p, 720p, 4K）
- 多相機同時跑時的 CPU utilization breakdown
- 記憶體頻寬使用情況（`nvidia-smi`, `perf stat`）

#### 1.2 OpenCV CUDA 預處理

替換 CPU 版影像前處理為 GPU 版：

```cpp
// 現有 CPU 路徑
cv::Mat gray;
cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);

// 改為 GPU 路徑
cv::cuda::GpuMat d_input, d_gray;
d_input.upload(input);
cv::cuda::cvtColor(d_input, d_gray, cv::COLOR_BGR2GRAY);
// 如果需要 blur
cv::Ptr<cv::cuda::Filter> gauss = cv::cuda::createGaussianFilter(
    CV_8UC1, CV_8UC1, cv::Size(3,3), 0.8);
gauss->apply(d_gray, d_gray);
d_gray.download(gray);
```

**預估加速**：10-20%（前處理佔比不大，但減少 CPU 負擔有意義）

#### 1.3 驗證指標
- [ ] 各步驟耗時基線數據（至少 3 種解析度）
- [ ] GPU 前處理 vs CPU 前處理的時間對比
- [ ] 偵測準確度不變（regression test）

---

### Phase 2：核心 CUDA Kernel（2-4 週）

**目標**：將最耗時的 Step 2-3 移到 GPU

#### 2.1 梯度計算 CUDA Kernel

這是最適合 GPU 加速的步驟 — 每個 pixel 完全獨立。

```cuda
__global__ void compute_gradient_kernel(
    const float* __restrict__ image,    // 輸入影像
    float* __restrict__ theta,          // 輸出梯度方向
    float* __restrict__ magnitude,      // 輸出梯度強度
    int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x + 1;  // skip border
    int y = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (x >= width - 1 || y >= height - 1) return;

    float Ix = image[y * stride + (x+1)] - image[y * stride + (x-1)];
    float Iy = image[(y+1) * stride + x] - image[(y-1) * stride + x];

    int idx = y * stride + x;
    theta[idx] = atan2f(Iy, Ix);
    magnitude[idx] = Ix * Ix + Iy * Iy;
}
```

**Launch config 建議**：
- Block size: 16×16 或 32×32
- Grid size: ceil(width/blockDim.x) × ceil(height/blockDim.y)
- RTX 4090 有 128 SM，1080p 影像可完全飽和

#### 2.2 邊緣生成 CUDA Kernel

每個 pixel 生成最多 4 條 edge，也是 pixel-level 平行：

```cuda
__global__ void generate_edges_kernel(
    const float* __restrict__ theta,
    const float* __restrict__ mag,
    Edge* __restrict__ edges,
    int* __restrict__ edge_count,       // atomic counter
    int width, int height,
    float min_mag, float max_edge_cost)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width - 1 || y >= height - 1) return;
    if (mag[y * width + x] < min_mag) return;

    // 4 個鄰居方向: right, down, right-down, left-down
    const int dx[] = {1, 0, 1, -1};
    const int dy[] = {0, 1, 1, 1};

    for (int i = 0; i < 4; i++) {
        int nx = x + dx[i], ny = y + dy[i];
        if (nx < 0 || nx >= width || ny >= height) continue;
        if (mag[ny * width + nx] < min_mag) continue;

        float theta_err = fabsf(theta[y*width+x] - theta[ny*width+nx]);
        // normalize to [0, pi]
        if (theta_err > M_PI) theta_err = 2*M_PI - theta_err;

        if (theta_err <= max_edge_cost) {
            int idx = atomicAdd(edge_count, 1);
            edges[idx] = {y*width+x, ny*width+nx, theta_err};
        }
    }
}
```

#### 2.3 GPU Radix Sort

邊緣排序用 **CUB library** 的 `cub::DeviceRadixSort`（比 thrust 更快）：

```cpp
#include <cub/device/device_radix_sort.cuh>

// 按 edge cost 排序
cub::DeviceRadixSort::SortPairs(
    d_temp, temp_bytes,
    d_edge_costs, d_sorted_costs,
    d_edge_indices, d_sorted_indices,
    num_edges);
```

**預估加速**：Step 2-3 合計 **5-10x**，整體 pipeline **2-3x**

#### 2.4 整合方式

修改 AprilTag C library 的 `apriltag_detector_detect()` 函式，在 Step 2-3 插入 CUDA 路徑：

```
apriltag_detector_detect()
  ├── [GPU] Step 1: image upload to device
  ├── [GPU] Step 2: gradient kernel
  ├── [GPU] Step 3a: edge generation kernel
  ├── [GPU] Step 3b: edge sort (CUB radix sort)
  ├── [GPU→CPU] download sorted edges + gradient data
  ├── [CPU] Step 3c: union-find merge (sequential)
  ├── [CPU] Step 4-9: 原有邏輯
  └── return detections
```

#### 2.5 驗證指標
- [ ] 梯度計算結果與 CPU 版本 bit-exact（或 < 1e-6 誤差）
- [ ] 邊緣排序結果與 CPU 版本一致
- [ ] 整體偵測結果不變
- [ ] profiling 數據確認加速幅度

---

### Phase 3：深度優化 + Pipeline 重構（4-6 週，可選）

**目標**：最大化 GPU 利用率，減少 CPU-GPU 資料傳輸

#### 3.1 Union-Find on GPU

使用 GPU-accelerated connected components（參考文獻）：
- Jaiganesh & Burtscher, "A High-Performance Connected Components Implementation for GPUs" (2018)
- 用 label equivalence 方法取代傳統 union-find
- 適合影像級連通域分析

#### 3.2 多影像 Batch Processing

多顆相機的影像可以 batch 進同一個 GPU：

```
Camera 0 ─┐
Camera 1 ─┤
Camera 2 ─┼──→ [GPU Batch Kernel] ──→ [CPU Decode per camera]
Camera 3 ─┤
Camera 4 ─┘
```

- 合併多張影像到一個大 buffer
- 用 batch offset 區分不同相機
- 單次 kernel launch 處理所有相機的梯度計算
- **這對多相機場景效益最大**

#### 3.3 CUDA Stream Pipeline

用 CUDA streams 實現 CPU/GPU 重疊執行：

```
Stream 0: [Upload img0] [Gradient img0] [Edge img0] [Download img0]
Stream 1:                [Upload img1]   [Gradient img1] [Edge img1] [Download img1]
CPU:      [Decode prev]  [Decode prev]   [Decode img0]   [Decode img1]
```

#### 3.4 Zero-Copy Memory（如果 PCIe 頻寬是瓶頸）

```cpp
// Pinned memory for faster transfers
cudaMallocHost(&h_image, image_size);
// 或 mapped memory 避免顯式傳輸
cudaHostAlloc(&h_image, image_size, cudaHostAllocMapped);
```

#### 3.5 預估效能

| 解析度 | CPU Only | Phase 1 | Phase 2 | Phase 3 |
|--------|----------|---------|---------|---------|
| 720p   | ~30ms    | ~25ms   | ~12ms   | ~6ms    |
| 1080p  | ~60ms    | ~50ms   | ~22ms   | ~10ms   |
| 4K     | ~200ms   | ~170ms  | ~70ms   | ~30ms   |

*以上為單相機估計值，實際需 profiling 驗證*

---

## 技術依賴

### 必要
- CUDA Toolkit 11.8+ (建議 12.x)
- CUB library（CUDA 11+ 已內建）
- CMake 3.18+（支援 CUDA as first-class language）
- OpenCV 4.x with CUDA support（`-DWITH_CUDA=ON` 編譯）

### 建議
- Nsight Systems / Nsight Compute（GPU profiling）
- nvtop（即時 GPU 監控）

### Build 設定

```cmake
# CMakeLists.txt 新增
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86;89")  # 86=RTX3090, 89=RTX4090

find_package(CUDAToolkit REQUIRED)

add_library(apriltag_cuda
    src/cuda/gradient_kernel.cu
    src/cuda/edge_kernel.cu
    src/cuda/sort_wrapper.cu
)
target_link_libraries(apriltag_cuda CUDA::cudart)
```

---

## 檔案結構（建議）

```
apriltag_ros/
├── src/
│   ├── AprilTagNode.cpp          # 修改：加入 GPU 路徑選擇
│   ├── cuda/
│   │   ├── apriltag_cuda.h       # CUDA 加速 API
│   │   ├── gradient_kernel.cu    # Step 2 GPU kernel
│   │   ├── edge_kernel.cu        # Step 3a GPU kernel
│   │   ├── sort_wrapper.cu       # Step 3b CUB sort wrapper
│   │   └── cuda_utils.h          # GPU memory helpers
│   ├── pose_estimation.cpp
│   └── ...
├── CMakeLists.txt                # 修改：加入 CUDA 支援
└── docs/
    └── cuda_acceleration_plan.md  # 本文件
```

---

## 風險與緩解

| 風險 | 影響 | 緩解措施 |
|------|------|----------|
| GPU-CPU 傳輸延遲抵消計算加速 | Phase 2 效益低於預期 | 用 pinned memory + async transfer；Phase 3 batch 處理 |
| Union-Find GPU 實作複雜度高 | Phase 3 延期 | 保持 union-find 在 CPU，只加速前面步驟 |
| OpenCV CUDA build 問題 | Phase 1 阻塞 | 可先跳過，直接用 CUDA kernel |
| 數值精度差異導致偵測結果不同 | 誤偵測 | 嚴格 regression test，比對每步中間結果 |

---

## 參考資料

1. AprilTag 3 Paper: Krogius et al., "Flexible Layouts for Fiducial Tags" (2019)
2. GPU Connected Components: Jaiganesh & Burtscher (2018)
3. CUB Library: https://nvlabs.github.io/cub/
4. NVIDIA cuAprilTags: 閉源，僅供參考架構設計
5. AprilTag C Library: https://github.com/AprilRobotics/apriltag
