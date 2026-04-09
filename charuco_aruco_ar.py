import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def make_aruco_detector(dictionary_id: int, detector_params: "cv2.aruco.DetectorParameters"):
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(aruco_dict, detector_params), aruco_dict
    return None, aruco_dict


def detect_markers(frame, detector, aruco_dict, detector_params):
    if detector is not None:
        corners, ids, rejected = detector.detectMarkers(frame)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=detector_params)
    return corners, ids, rejected


def calibrate_from_charuco_video(video_path: str) -> Tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector, aruco_dict = make_aruco_detector(cv2.aruco.DICT_6X6_250, detector_params)

    grid_x, grid_y = 5, 7
    square_size = 4.0
    board = cv2.aruco.CharucoBoard((grid_x, grid_y), square_size, square_size / 2.0, aruco_dict)

    charuco_params = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    obj_points_list: List[np.ndarray] = []
    img_points_list: List[np.ndarray] = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))

        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)

        if charuco_corners is not None and len(charuco_corners) >= 4:
            obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
            # 每隔固定幀取一次，避免重複視角影響校正穩定性。
            if obj_points is not None and img_points is not None and obj_points.shape[0] >= 4 and frame_id % 25 == 0:
                obj_points_list.append(obj_points.astype(np.float32))
                img_points_list.append(img_points.astype(np.float32))
        elif marker_ids is not None and len(marker_ids) >= 4 and marker_corners is not None:
            # 後備方案：若 charuco 角點不足，嘗試由 marker 插值得到 charuco 角點。
            retval, interp_corners, interp_ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, frame, board
            )
            if retval is not None and retval >= 4:
                obj_points, img_points = board.matchImagePoints(interp_corners, interp_ids)
                if obj_points is not None and img_points is not None and obj_points.shape[0] >= 4 and frame_id % 25 == 0:
                    obj_points_list.append(obj_points.astype(np.float32))
                    img_points_list.append(img_points.astype(np.float32))

        frame_id += 1

    cap.release()

    if len(obj_points_list) < 8:
        raise RuntimeError(f"可用的校正樣本太少: {len(obj_points_list)}，請確認 ChArUco board 是否清楚可見。")

    camera_matrix_init = np.array(
        [[1000.0, 0.0, frame_width / 2.0], [0.0, 1000.0, frame_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs_init = np.zeros((5, 1), dtype=np.float64)

    _, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        obj_points_list,
        img_points_list,
        (frame_width, frame_height),
        camera_matrix_init,
        dist_coeffs_init,
    )
    return camera_matrix, dist_coeffs


def build_overlay_sources(overlay_paths: List[str], num_sources: int = 6):
    if len(overlay_paths) != num_sources:
        raise RuntimeError(f"必須提供 {num_sources} 支影片，目前收到 {len(overlay_paths)} 支。")

    sources = []
    for i in range(num_sources):
        if not os.path.exists(overlay_paths[i]):
            raise RuntimeError(f"找不到第 {i + 1} 支影片: {overlay_paths[i]}")
        cap = cv2.VideoCapture(overlay_paths[i])
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟第 {i + 1} 支影片: {overlay_paths[i]}")
        sources.append(cap)
    return sources


def read_looping_frame(src) -> np.ndarray:
    ok, frame = src.read()
    if ok and frame is not None:
        return frame
    src.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame = src.read()
    if ok and frame is not None:
        return frame
    raise RuntimeError("影片讀取失敗，且無法循環重播。")


def overlay_video_on_marker(
    frame: np.ndarray, source_frame: np.ndarray, marker_corners: np.ndarray, rotate_180: bool = True
) -> np.ndarray:
    h, w = source_frame.shape[:2]
    src_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    dst_pts = marker_corners.astype(np.float32)
    if rotate_180:
        # OpenCV 偵測到的 marker 角點方向和要貼上的畫面方向可能相反，預設旋轉 180 度。
        dst_pts = dst_pts[[2, 3, 0, 1]]

    homography, _ = cv2.findHomography(src_pts, dst_pts)
    if homography is None:
        return frame

    warped = cv2.warpPerspective(source_frame, homography, (frame.shape[1], frame.shape[0]))
    mask_src = np.full((h, w), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, homography, (frame.shape[1], frame.shape[0]))
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(warped, warped, mask=mask)
    return cv2.add(bg, fg)


def run_aruco_ar(
    input_video: str,
    output_video: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    overlay_paths: List[str],
) -> None:
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"無法開啟影片: {input_video}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    frame_width, frame_height = src_w // 2, src_h // 2

    writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"無法建立輸出影片: {output_video}")

    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector, aruco_dict = make_aruco_detector(cv2.aruco.DICT_7X7_50, detector_params)
    sources = build_overlay_sources(overlay_paths, num_sources=6)

    marker_to_source: Dict[int, int] = {}
    seen_ids = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (frame_width, frame_height))

        corners, ids, _ = detect_markers(frame, detector, aruco_dict, detector_params)
        if ids is not None and len(ids) > 0:
            ids_flat = ids.flatten().tolist()
            for marker_id in ids_flat:
                seen_ids.add(int(marker_id))
                if marker_id not in marker_to_source and len(marker_to_source) < 6:
                    marker_to_source[int(marker_id)] = len(marker_to_source)

            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            for i, marker_id in enumerate(ids_flat):
                marker_id = int(marker_id)
                if marker_id not in marker_to_source:
                    continue
                source_idx = marker_to_source[marker_id]
                source_frame = read_looping_frame(sources[source_idx])
                frame = overlay_video_on_marker(frame, source_frame, corners[i][0], rotate_180=True)

        writer.write(frame)

    cap.release()
    writer.release()
    for src in sources:
        src.release()

    print(f"在 arUco_marker.mp4 中偵測到的唯一 marker 數量: {len(seen_ids)}")
    print(f"已綁定到 AR 視訊貼圖的 marker id -> source index: {marker_to_source}")
    if len(marker_to_source) < 6:
        print("警告: 綁定到的 marker 少於 6 個，請確認影片中 6 個 marker 都有清楚出現。")
    print(f"AR 輸出影片: {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco calibration + ArUco AR overlay")
    parser.add_argument("--charuco-video", default="ChArUco_board.mp4")
    parser.add_argument("--aruco-video", default="ArUco_marker.mp4")
    parser.add_argument("--output-video", default="aruco_ar_output_6videos.mp4")
    parser.add_argument(
        "--overlay-videos",
        nargs="*",
        default=[
            "overlays/e_04ZrNroTo.mp4",
            "overlays/020g-0hhCAU.mp4",
            "overlays/WRVsOCh907o.mp4",
            "overlays/fdPu-wvl3KE.mp4",
            "overlays/MR5XSOdjKMA.mp4",
            "overlays/HHvM8DmgjAA.mp4",
        ],
        help="預設使用 overlays/ 下固定 6 支影片；可自行覆寫。",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    camera_matrix, dist_coeffs = calibrate_from_charuco_video(args.charuco_video)
    print("camera matrix:")
    print(camera_matrix)
    print("distortion coefficients:")
    print(dist_coeffs)

    run_aruco_ar(
        input_video=args.aruco_video,
        output_video=args.output_video,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        overlay_paths=args.overlay_videos,
    )


if __name__ == "__main__":
    main()
