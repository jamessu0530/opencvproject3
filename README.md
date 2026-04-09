# ChArUco Calibration and ArUco AR Overlay

這個專案完成兩個任務：

1. 使用 `ChArUco_board.mp4` 做相機校正，輸出 `camera matrix` 與 `distortion coefficients`  
2. 在 `ArUco_marker.mp4` 偵測 6 個 ArUco marker，將 6 支影片做 AR 貼附

## 檔案說明

- `charuco_aruco_ar.py`：主程式
- `ChArUco_board.mp4`：校正用影片
- `ArUco_marker.mp4`：ArUco marker 偵測與 AR 疊合用影片
- `overlays/*.mp4`：6 支要貼到 marker 的影片素材

## 環境需求

- Python 3.10+
- OpenCV contrib（含 aruco/charuco）

建議使用虛擬環境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install opencv-contrib-python numpy
```

## 執行方式

直接執行（已內建 6 支 overlay 預設路徑）：

```bash
.venv/bin/python charuco_aruco_ar.py
```

可自訂輸出檔名：

```bash
.venv/bin/python charuco_aruco_ar.py --output-video aruco_ar_output_custom.mp4
```

## 程式輸出

終端機會印出：

- `camera matrix`
- `distortion coefficients`
- 偵測到的 marker 數量與綁定結果

輸出影片預設為：

- `aruco_ar_output_6videos.mp4`

## 注意事項

- 程式目前設計為必須有 6 支 overlay 影片，缺一支會報錯
- 座標軸繪製已移除，輸出僅保留 AR 貼圖結果
