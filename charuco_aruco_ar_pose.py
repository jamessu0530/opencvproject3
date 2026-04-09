#!/usr/bin/env python3
"""
講義版 AR：與 charuco_aruco_ar.py 分開，不覆蓋原檔。

差異（ArUco 段落）：
- 用 camera_matrix、dist_coeffs 做 solvePnP（與講義一致）
- 用 projectPoints 把 marker 平面上的四個 3D 角點投影回影像，得到貼圖四邊形
- 再用 findHomography + warpPerspective 把影片貼上（投影結果決定四邊形位置）
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import cv2
import numpy as np

from charuco_aruco_ar import (
    build_overlay_sources,
    calibrate_from_charuco_video,
    detect_markers,
    make_aruco_detector,
    read_looping_frame,
)


def marker_object_points_3d(marker_length: float) -> np.ndarray:
    """OpenCV ArUco 慣用之 marker 四角（單位須與校正時一致，此處與講義同用 6.0）。"""
    h = marker_length / 2.0
    return np.array(
        [[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]],
        dtype=np.float32,
    )


def overlay_video_via_pose(
    frame: np.ndarray,
    source_frame: np.ndarray,
    marker_corners_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_length: float,
    rotate_texture_180: bool,
) -> np.ndarray:
    obj_pts = marker_object_points_3d(marker_length)
    img_pts_in = marker_corners_2d.astype(np.float32).reshape(4, 1, 2)

    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts_in, camera_matrix, dist_coeffs)
    if not ok:
        return frame

    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, camera_matrix, dist_coeffs)
    dst_pts = proj.reshape(4, 2).astype(np.float32)
    if rotate_texture_180:
        dst_pts = dst_pts[[2, 3, 0, 1]]

    sh, sw = source_frame.shape[:2]
    src_pts = np.array([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]], dtype=np.float32)

    homography, _ = cv2.findHomography(src_pts, dst_pts)
    if homography is None:
        return frame

    warped = cv2.warpPerspective(source_frame, homography, (frame.shape[1], frame.shape[0]))
    mask_src = np.full((sh, sw), 255, dtype=np.uint8)
    mask = cv2.warpPerspective(mask_src, homography, (frame.shape[1], frame.shape[0]))
    mask_inv = cv2.bitwise_not(mask)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(warped, warped, mask=mask)
    return cv2.add(bg, fg)


def run_aruco_ar_pose(
    input_video: str,
    output_video: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    overlay_paths: List[str],
    marker_length: float,
    rotate_texture_180: bool,
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
                c2d = corners[i][0]
                frame = overlay_video_via_pose(
                    frame,
                    source_frame,
                    c2d,
                    camera_matrix,
                    dist_coeffs,
                    marker_length,
                    rotate_texture_180,
                )

        writer.write(frame)

    cap.release()
    writer.release()
    for src in sources:
        src.release()

    print(f"在 arUco_marker.mp4 中偵測到的唯一 marker 數量: {len(seen_ids)}")
    print(f"已綁定到 AR 視訊貼圖的 marker id -> source index: {marker_to_source}")
    if len(marker_to_source) < 6:
        print("警告: 綁定到的 marker 少於 6 個，請確認影片中 6 個 marker 都有清楚出現。")
    print(f"AR 輸出影片（講義版 pose+project）: {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="ChArUco 校正 + ArUco AR（講義：solvePnP + projectPoints）")
    parser.add_argument("--charuco-video", default="ChArUco_board.mp4")
    parser.add_argument("--aruco-video", default="ArUco_marker.mp4")
    parser.add_argument(
        "--output-video",
        default="pose_pnp_ar_output.mp4",
        help="預設與 charuco_aruco_ar.py 的 aruco_ar_output_6videos.mp4 不同檔名，避免覆蓋",
    )
    parser.add_argument(
        "--marker-length",
        type=float,
        default=6.0,
        help="與講義一致之 marker 邊長（單位與 ChArUco square 一致，預設 6）",
    )
    parser.add_argument(
        "--no-rotate-texture-180",
        action="store_true",
        help="關閉貼圖四角順序旋轉 180°（若畫面上下顛倒可拿掉此選項再試）",
    )
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
        help="6 支 overlay 影片路徑",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    camera_matrix, dist_coeffs = calibrate_from_charuco_video(args.charuco_video)
    print("camera matrix:")
    print(camera_matrix)
    print("distortion coefficients:")
    print(dist_coeffs)

    run_aruco_ar_pose(
        input_video=args.aruco_video,
        output_video=args.output_video,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        overlay_paths=args.overlay_videos,
        marker_length=args.marker_length,
        rotate_texture_180=not args.no_rotate_texture_180,
    )


if __name__ == "__main__":
    main()
