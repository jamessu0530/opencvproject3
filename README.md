# ChArUco 相機校正與 ArUco AR 疊影

專案完成兩件事：

1. 使用 `ChArUco_board.mp4` 做相機校正，在終端機印出 **camera matrix** 與 **distortion coefficients**  
2. 在 `ArUco_marker.mp4` 偵測 6 個 ArUco marker，將 **6 支不同影片** 貼附在 marker 上（隨透視變形）

## 主程式（兩種貼圖流程）

| 檔案 | 說明 | 預設輸出影片 |
|------|------|----------------|
| `charuco_aruco_ar.py` | 直接用 `detectMarkers` 的四角做 `findHomography` + `warpPerspective` | `aruco_ar_output_6videos.mp4` |
| `charuco_aruco_ar_pose.py` | 講義路線：內參／畸變 + `solvePnP` + `projectPoints` 得到四角，再 homography 貼圖（檔名含 **pose** / **pnp** 即此意） | `pose_pnp_ar_output.mp4` |

兩者在角點準、`marker_length` 正確時，**畫面常非常接近**；差異在於後者顯式使用校正後的相機模型做 PnP 與投影。

`charuco_aruco_ar_pose.py` 會 `import charuco_aruco_ar` 以共用校正與偵測輔助函式，請與前者放在同一資料夾執行。

## 本機需自備的檔案（未納入 Git）

以下檔案體積大，`.gitignore` 已排除，clone 後請自行放到專案根目錄：

- `ChArUco_board.mp4`
- `ArUco_marker.mp4`
- `overlays/` 內 6 支 `.mp4`（或自行修改 `--overlay-videos`）

預設 overlay 路徑見兩支程式內的 `parse_args`。

## 環境

- Python 3.10+
- `opencv-contrib-python`（含 `cv2.aruco` / ChArUco）

```bash
python3 -m venv .venv
.venv/bin/python -m pip install opencv-contrib-python numpy
```

## 執行

```bash
cd /path/to/opencvproject3

# 直接貼圖版
.venv/bin/python charuco_aruco_ar.py

# PnP + projectPoints 版（輸出檔名不同，不會覆蓋上一支）
.venv/bin/python charuco_aruco_ar_pose.py
```

自訂輸出或素材：

```bash
.venv/bin/python charuco_aruco_ar.py --output-video my_out.mp4
.venv/bin/python charuco_aruco_ar_pose.py --marker-length 6.0 --no-rotate-texture-180
```

## 程式輸出

- 終端機：`camera matrix`、`distortion coefficients`、偵測到的 marker 數量與 id 綁定對照  
- 影片：僅 **影像**，OpenCV `VideoWriter` **不含音軌**（overlay 與底片聲音都不會寫入）

## 注意

- 必須能開啟 **6 支** overlay 影片，缺檔會直接報錯  
- ArUco 字典為 `DICT_7X7_50`；ChArUco 校正為 `DICT_6X6_250`、棋盤規格與講義一致（5×7、方格 4 cm 等），見 `calibrate_from_charuco_video`

## Repository

<https://github.com/jamessu0530/opencvproject3>
