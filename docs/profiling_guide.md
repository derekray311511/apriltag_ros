# AprilTag Profiling 指南（到公司後執行）

## 前置準備

### 1. 確認 CUDA 環境
```bash
nvidia-smi                    # 確認 GPU 可用
nvcc --version                # 確認 CUDA toolkit
```

### 2. 編譯帶 CUDA 的 OpenCV（如果還沒有）
```bash
# 檢查現有 OpenCV 是否有 CUDA
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep CUDA

# 如果沒有，需要重新編譯 OpenCV
# 參考: https://docs.opencv.org/4.x/d6/d15/tutorial_building_tegra_cuda.html
```

### 3. 安裝 profiling 工具
```bash
# Nsight Systems（通常隨 CUDA toolkit 安裝）
which nsys

# nvtop（即時 GPU 監控）
sudo apt install nvtop   # Ubuntu
```

---

## Step 1：AprilTag 內建 Profiling

### 啟用方式

修改你的 tag config YAML：
```yaml
# e.g. cfg/tags_16h5.yaml
family: "16h5"
size: 0.162
profile: true           # ← 加這行
detector:
  threads: 1
  decimate: 1.0         # 不降解析度
  blur: 0.0
  refine: true
  sharpening: 0.25
```

### 執行並收集數據

```bash
# 啟動節點
ros2 launch apriltag_ros camera_16h5.launch.yml

# 觀察 stdout 輸出的 timeprofile
# 會看到類似：
#  0.000000 init
#  0.004532 blur/decimate
#  0.012847 gradient
#  0.028134 clusters
#  0.031245 fit lines
#  0.033421 quads
#  0.034102 decode+verify
#  0.034567 reconcile
```

### 要記錄的數據

請在以下表格中填入實測數值：

#### 單相機 profiling（解析度 × 步驟耗時 ms）

| 步驟 | 720p (1280×720) | 1080p (1920×1080) | 4K (3840×2160) |
|------|-----------------|-------------------|----------------|
| blur/decimate | | | |
| gradient | | | |
| clusters (edge+sort+UF) | | | |
| fit lines | | | |
| quads | | | |
| decode+verify | | | |
| reconcile | | | |
| **Total** | | | |

#### 多相機 system-level profiling

| 指標 | 2 相機 | 4 相機 | 6 相機 |
|------|--------|--------|--------|
| CPU utilization (%) | | | |
| 平均每幀延遲 (ms) | | | |
| 最大每幀延遲 (ms) | | | |
| FPS per camera | | | |
| 記憶體用量 (MB) | | | |

---

## Step 2：CPU 熱點分析（可選但有用）

```bash
# 用 perf 分析 CPU 熱點
sudo perf record -g ros2 run apriltag_ros apriltag_node \
    --ros-args --params-file cfg/tags_16h5.yaml

sudo perf report
# 看哪些函式佔最多 CPU 時間
```

```bash
# 或用 valgrind/callgrind（更詳細但很慢）
valgrind --tool=callgrind ros2 run apriltag_ros apriltag_node \
    --ros-args --params-file cfg/tags_16h5.yaml
# 跑幾秒後 Ctrl+C
kcachegrind callgrind.out.*
```

---

## Step 3：GPU 傳輸 Baseline

在開始寫 CUDA kernel 前，先測一下 GPU 資料傳輸成本：

```cpp
// test_transfer.cu - 簡單的傳輸測試
#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

int main() {
    // 模擬不同解析度
    int sizes[][2] = {{1280,720}, {1920,1080}, {3840,2160}};
    const char* names[] = {"720p", "1080p", "4K"};

    for (int s = 0; s < 3; s++) {
        int w = sizes[s][0], h = sizes[s][1];
        size_t bytes = w * h * sizeof(float);

        float *h_data, *d_data;
        cudaMallocHost(&h_data, bytes);  // pinned memory
        cudaMalloc(&d_data, bytes);

        // Warmup
        cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        // Benchmark upload
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double upload_ms = std::chrono::duration<double,std::milli>(t1-t0).count() / 100;

        // Benchmark download
        t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; i++) {
            cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
        }
        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        double download_ms = std::chrono::duration<double,std::milli>(t1-t0).count() / 100;

        printf("%s (%dx%d, %.1f MB): upload=%.3f ms, download=%.3f ms\n",
               names[s], w, h, bytes/1e6, upload_ms, download_ms);

        cudaFree(d_data);
        cudaFreeHost(h_data);
    }
    return 0;
}
```

編譯並執行：
```bash
nvcc -O2 test_transfer.cu -o test_transfer && ./test_transfer
```

### 要記錄的結果

| 解析度 | Upload (ms) | Download (ms) | 備註 |
|--------|-------------|---------------|------|
| 720p | | | |
| 1080p | | | |
| 4K | | | |

**判斷標準**：如果 upload+download > gradient 步驟耗時，Phase 2 需要更多步驟搬到 GPU 才有意義。

---

## Step 4：驗證測試方案

準備一組固定的測試影像，確保 CUDA 版本結果與 CPU 版本一致：

```bash
# 錄一段包含 AprilTag 的 rosbag
ros2 bag record /camera/image_rect /camera/camera_info -o apriltag_test_bag

# 之後可以反覆 replay 測試
ros2 bag play apriltag_test_bag --loop
```

### Regression Test Checklist
- [ ] 偵測到的 tag ID 完全一致
- [ ] tag 角點座標誤差 < 0.5 pixel
- [ ] pose 估計誤差 < 1mm translation, < 0.1° rotation
- [ ] 無漏偵測（false negative）
- [ ] 無誤偵測（false positive）

---

## 預期成果

完成以上 profiling 後，我們會有：
1. **精確的各步驟耗時數據** → 決定 Phase 2 要加速哪些步驟
2. **GPU 傳輸成本** → 決定要不要用 pinned memory / batch processing
3. **Regression test 基線** → 確保 CUDA 版本正確性
4. **系統級瓶頸分析** → 確認 GPU 加速能解決多相機場景的問題
