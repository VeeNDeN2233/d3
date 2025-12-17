
from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


MINI_RGBD_JOINT_NAMES = [
    "global",
    "leftThigh",
    "rightThigh",
    "spine",
    "leftCalf",
    "rightCalf",
    "spine1",
    "leftFoot",
    "rightFoot",
    "spine2",
    "leftToes",
    "rightToes",
    "neck",
    "leftShoulder",
    "rightShoulder",
    "head",
    "leftUpperArm",
    "rightUpperArm",
    "leftForeArm",
    "rightForeArm",
    "leftHand",
    "rightHand",
    "leftFingers",
    "rightFingers",
    "noseVertex",
]


MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_INDEX = 19
MP_RIGHT_INDEX = 20
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32


class VideoProcessor:

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    async def process_video(self, input_path: str, output_path: str, save_keypoints: bool = True) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._process_video_sync,
            input_path,
            output_path,
            save_keypoints,
        )

    def _process_video_sync(self, input_path: str, output_path: str, save_keypoints: bool) -> Dict[str, Any]:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return {"success": False, "error": "Не удалось открыть видео файл"}

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps) if fps and fps > 0 else 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0
        all_keypoints: List[Dict[str, Any]] = []

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                results = self._pose.process(frame_rgb)


                if results.pose_landmarks:
                    self._mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        self._mp_pose.POSE_CONNECTIONS,
                    )


                if save_keypoints:
                    landmarks_33 = self._landmarks_to_list(results.pose_landmarks)

                    mini_rgbd_joints = self._convert_to_mini_rgbd(results.pose_landmarks, width, height)
                    all_keypoints.append(
                        {
                            "frame": frames_processed,
                            "landmarks": landmarks_33,
                            "mini_rgbd": mini_rgbd_joints,
                            "has_person": results.pose_landmarks is not None,
                            "confidence": np.mean([lm["visibility"] for lm in landmarks_33]) if landmarks_33 else 0.0,
                        }
                    )

                out.write(frame_bgr)
                frames_processed += 1

        finally:
            cap.release()
            out.release()

        keypoints_path: Optional[str] = None
        if save_keypoints:
            keypoints_path = self._save_keypoints(output_path, all_keypoints, width, height, fps)

        return {
            "success": True,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "keypoints_path": keypoints_path,
            "keypoints_count": len(all_keypoints),
        }

    def _landmarks_to_list(self, pose_landmarks) -> List[Dict[str, float]]:
        if pose_landmarks is None:

            return [
                {"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.0}
                for _ in range(33)
            ]
        
        out: List[Dict[str, float]] = []
        for lm in pose_landmarks.landmark:
            out.append(
                {
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 0.0)),
                }
            )
        return out

    def _get_landmark_point(self, landmarks, idx: int) -> Optional[Tuple[float, float, float]]:
        if landmarks is None or idx >= len(landmarks.landmark):
            return None
        lm = landmarks.landmark[idx]
        return (lm.x, lm.y, lm.z)

    def _average_points(self, p1: Optional[Tuple[float, float, float]], p2: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
        if p1 is None and p2 is None:
            return None
        if p1 is None:
            return p2
        if p2 is None:
            return p1
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2)

    def _convert_to_mini_rgbd_from_list(
        self, landmarks_33: List[Dict[str, float]], width: int, height: int
    ) -> List[Tuple[float, float, float, float]]:
        if not landmarks_33 or len(landmarks_33) != 33:

            return [(0.0, 0.0, 0.0, i) for i in range(25)]
        


        return self._convert_landmarks_list_to_mini_rgbd(landmarks_33, width, height)
    
    def _convert_landmarks_list_to_mini_rgbd(
        self, landmarks_33: List[Dict[str, float]], width: int, height: int
    ) -> List[Tuple[float, float, float, float]]:


        class TempLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        
        class TempLandmarks:
            def __init__(self, landmarks_list):
                self.landmark = [TempLandmark(lm["x"], lm["y"], lm["z"]) for lm in landmarks_list]
        
        temp_landmarks = TempLandmarks(landmarks_33)
        return self._convert_to_mini_rgbd(temp_landmarks, width, height)
    
    def _convert_to_mini_rgbd(self, pose_landmarks, width: int, height: int) -> List[Tuple[float, float, float, float]]:
        if pose_landmarks is None:

            return [(0.0, 0.0, 0.0, i) for i in range(25)]


        nose = self._get_landmark_point(pose_landmarks, MP_NOSE)
        left_shoulder = self._get_landmark_point(pose_landmarks, MP_LEFT_SHOULDER)
        right_shoulder = self._get_landmark_point(pose_landmarks, MP_RIGHT_SHOULDER)
        left_elbow = self._get_landmark_point(pose_landmarks, MP_LEFT_ELBOW)
        right_elbow = self._get_landmark_point(pose_landmarks, MP_RIGHT_ELBOW)
        left_wrist = self._get_landmark_point(pose_landmarks, MP_LEFT_WRIST)
        right_wrist = self._get_landmark_point(pose_landmarks, MP_RIGHT_WRIST)
        left_index = self._get_landmark_point(pose_landmarks, MP_LEFT_INDEX)
        right_index = self._get_landmark_point(pose_landmarks, MP_RIGHT_INDEX)
        left_hip = self._get_landmark_point(pose_landmarks, MP_LEFT_HIP)
        right_hip = self._get_landmark_point(pose_landmarks, MP_RIGHT_HIP)
        left_knee = self._get_landmark_point(pose_landmarks, MP_LEFT_KNEE)
        right_knee = self._get_landmark_point(pose_landmarks, MP_RIGHT_KNEE)
        left_ankle = self._get_landmark_point(pose_landmarks, MP_LEFT_ANKLE)
        right_ankle = self._get_landmark_point(pose_landmarks, MP_RIGHT_ANKLE)
        left_foot_index = self._get_landmark_point(pose_landmarks, MP_LEFT_FOOT_INDEX)
        right_foot_index = self._get_landmark_point(pose_landmarks, MP_RIGHT_FOOT_INDEX)


        hip_center = self._average_points(left_hip, right_hip)
        shoulder_center = self._average_points(left_shoulder, right_shoulder)
        left_upper_arm_mid = self._average_points(left_shoulder, left_elbow)
        right_upper_arm_mid = self._average_points(right_shoulder, right_elbow)


        def to_pixel_and_depth(point: Optional[Tuple[float, float, float]]) -> Tuple[float, float, float]:
            if point is None:
                return (0.0, 0.0, 0.0)



            x_pixel = point[0] * width
            y_pixel = point[1] * height


            depth_mm = 1000.0 - (point[2] * 500.0)
            return (x_pixel, y_pixel, depth_mm)


        joints = [
            to_pixel_and_depth(hip_center),
            to_pixel_and_depth(left_hip),
            to_pixel_and_depth(right_hip),
            to_pixel_and_depth(hip_center),
            to_pixel_and_depth(left_knee),
            to_pixel_and_depth(right_knee),
            to_pixel_and_depth(shoulder_center),
            to_pixel_and_depth(left_ankle),
            to_pixel_and_depth(right_ankle),
            to_pixel_and_depth(shoulder_center),
            to_pixel_and_depth(left_foot_index),
            to_pixel_and_depth(right_foot_index),
            to_pixel_and_depth(shoulder_center),
            to_pixel_and_depth(left_shoulder),
            to_pixel_and_depth(right_shoulder),
            to_pixel_and_depth(nose),
            to_pixel_and_depth(left_upper_arm_mid),
            to_pixel_and_depth(right_upper_arm_mid),
            to_pixel_and_depth(left_elbow),
            to_pixel_and_depth(right_elbow),
            to_pixel_and_depth(left_wrist),
            to_pixel_and_depth(right_wrist),
            to_pixel_and_depth(left_index),
            to_pixel_and_depth(right_index),
            to_pixel_and_depth(nose),
        ]


        return [(x, y, depth, i) for i, (x, y, depth) in enumerate(joints)]

    def _save_keypoints(
        self,
        output_video_path: str,
        all_keypoints: List[Dict[str, Any]],
        width: int,
        height: int,
        fps: float,
    ) -> str:
        output_dir = Path(output_video_path).parent
        keypoints_dir = output_dir / "keypoints"
        keypoints_dir.mkdir(parents=True, exist_ok=True)


        jointlist_path = keypoints_dir / "jointlist.txt"
        with open(jointlist_path, "w", encoding="utf-8") as f:
            for joint_name in MINI_RGBD_JOINT_NAMES:
                f.write(f"{joint_name}\n")


        joints_2ddep_dir = keypoints_dir / "joints_2Ddep"
        joints_3d_dir = keypoints_dir / "joints_3D"
        joints_2ddep_dir.mkdir(exist_ok=True)
        joints_3d_dir.mkdir(exist_ok=True)


        for frame_data in all_keypoints:
            frame_num = frame_data["frame"]
            mini_rgbd = frame_data.get("mini_rgbd", [])


            frame_str = f"{frame_num:05d}"


            file_2ddep = joints_2ddep_dir / f"syn_joints_2Ddep_{frame_str}.txt"
            with open(file_2ddep, "w", encoding="utf-8") as f:
                for x, y, depth, joint_id in mini_rgbd:
                    f.write(f"{x:.2f} {y:.2f} {depth:.2f} {int(joint_id)}\n")




            file_3d = joints_3d_dir / f"syn_joints_3D_{frame_str}.txt"
            with open(file_3d, "w", encoding="utf-8") as f:
                for x, y, depth, joint_id in mini_rgbd:


                    x_m = (x - width / 2) * 0.001
                    y_m = (y - height / 2) * 0.001

                    z_m = depth / 1000.0
                    f.write(f"{x_m:.4f} {y_m:.4f} {z_m:.4f} {int(joint_id)}\n")


        payload = {
            "format": "mini_rgbd",
            "source": "mediapipe_pose",
            "joints": 25,
            "video": {"width": width, "height": height, "fps": fps, "frames": len(all_keypoints)},
            "frames": all_keypoints,
        }

        json_path = keypoints_dir / "keypoints.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        return str(keypoints_dir)

    def __del__(self) -> None:
        try:
            self._pose.close()
        except Exception:
            pass
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass


