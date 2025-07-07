# import numpy as np
# from angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d
# from constants import JOINT_NAMES, ANGLE_JOINTS

# # Korean labels corresponding to ANGLE_JOINTS order
# JOINT_LABELS = [
#     "왼쪽 팔꿈치 각도",
#     "오른쪽 팔꿈치 각도",
#     "왼쪽 손목 각도",
#     "오른쪽 손목 각도",
#     "왼쪽 무릎 각도",
#     "오른쪽 무릎 각도",
#     "왼쪽 발목 각도",
#     "오른쪽 발목 각도",
#     "오른쪽 어깨 각도",
#     "왼쪽 어깨 각도",
#     "왼쪽 골반 각도",
#     "오른쪽 골반 각도",
# ]

# # def generate_frame_feedback(
# #     kp_ref: np.ndarray,
# #     kp_user: np.ndarray,
# #     angle_thresh: float = np.deg2rad(5)
# # ) -> list[str]:
# #     """
# #     Generate human-readable feedback for a single frame.
# #     """
# #     ref_ang   = calc_interior_angles_2d(kp_ref[None])[0]
# #     user_ang  = calc_interior_angles_2d(kp_user[None])[0]
# #     ref_bend  = calc_signed_bend_angles_2d(kp_ref[None])[0]
# #     user_bend = calc_signed_bend_angles_2d(kp_user[None])[0]

# #     msgs = []
# #     for i, label in enumerate(JOINT_LABELS):
# #         # use bend for elbows/knees; interior otherwise
# #         # elbows and knees use bend, others use interior-angle
# #         # if i in [0, 1, 4, 5]:  # indices for bend joints
# #         #     delta = user_bend[i] - ref_bend[i]
# #         # else:
# #         #     delta = user_ang[i] - ref_ang[i]

# #         # 모든 관절을 interior‐angle 차이로만 계산
# #         delta = user_ang[i] - ref_ang[i]
        
# #         # wrap to [-π,+π]
# #         delta = (delta + np.pi) % (2*np.pi) - np.pi
# #         if abs(delta) < angle_thresh:
# #             continue
# #         deg = np.degrees(abs(delta))
# #         msgs.append(f"{label}가 {deg:.1f}° 차이납니다")
# #     return msgs


# def get_directional_angle_indices():
#     """방향 피드백이 필요한 관절 조합만 걸러서 인덱스로 반환"""
#     target_keywords = {"elbow", "knee", "ankle"}  # 굽힘/펴기 의미가 있는 관절
#     directional_joint_triplets = []

#     for a, b, c in ANGLE_JOINTS:
#         if any(k in b for k in target_keywords):  # 중심 관절 기준
#             directional_joint_triplets.append((a, b, c))

#     return [(JOINT_NAMES.index(a), JOINT_NAMES.index(b), JOINT_NAMES.index(c))
#             for (a, b, c) in directional_joint_triplets]

# def generate_frame_feedback(
#     kp_ref: np.ndarray,
#     kp_user: np.ndarray,
#     angle_thresh: float = np.deg2rad(5),
#     direction_thresh: float = 0.2
# ) -> list[str]:
#     from angle_utils import ANGLE_IDX

#     directional_idx_set = set(get_directional_angle_indices())

#     ref_ang   = calc_interior_angles_2d(kp_ref[None])[0]  # (M,)
#     user_ang  = calc_interior_angles_2d(kp_user[None])[0]
#     ref_bend  = calc_signed_bend_angles_2d(kp_ref[None])[0]
#     user_bend = calc_signed_bend_angles_2d(kp_user[None])[0]

#     feedback = []

#     for i, (a, b, c) in enumerate(ANGLE_IDX):
#         angle_diff = abs(ref_ang[i] - user_ang[i])
#         signed_diff = user_bend[i] - ref_bend[i]

#         if angle_diff <= angle_thresh:
#             continue  # 충분히 유사하면 패스

#         joint_name = JOINT_LABELS[i]

#         # 방향 피드백 적용 여부
#         if (a, b, c) not in directional_idx_set:
#             continue            
#         vec_ref = kp_ref[c, :2] - kp_ref[b, :2]
#         vec_user = kp_user[c, :2] - kp_user[b, :2]

#         norm_ref = np.linalg.norm(vec_ref)
#         norm_user = np.linalg.norm(vec_user)

#         # 선생님과 학생의 각도 게산 변수
#         teacher_deg = np.degrees(ref_ang[i])
#         student_deg = np.degrees(user_ang[i])

#         if norm_ref < 1e-6 or norm_user < 1e-6:
#             continue

#         cos_sim = np.dot(vec_ref, vec_user) / (norm_ref * norm_user + 1e-6)
#         vector_diff = 1 - cos_sim

#         direction = "펴세요" if signed_diff < 0 else "구부리세요"
#         if vector_diff > direction_thresh:
#             feedback.append(f"선생님 : {teacher_deg}도 학생 : {student_deg}도 {joint_name}를 더 {direction} (방향도 조정 필요)")
#         else:
#             feedback.append(f"선생님 : {teacher_deg}도 학생 : {student_deg}도 {joint_name}를 더 {direction}")
#         # else:
#         #     # angle_diff만 사용하는 일반 피드백
#         #     direction = "펴세요" if signed_diff < 0 else "구부리세요"
#         #     feedback.append(f"{joint_name}를 더 {direction}")

#     return feedback


import numpy as np
from angle_utils import calc_interior_angles_2d, calc_signed_bend_angles_2d
from angle_utils import ANGLE_IDX

# Korean labels corresponding to ANGLE_JOINTS order
JOINT_LABELS = [
    "왼쪽 팔꿈치 각도",    # 0
    "오른쪽 팔꿈치 각도",  # 1
    "왼쪽 손목 각도",      # 2 (skip)
    "오른쪽 손목 각도",    # 3 (skip)
    "왼쪽 무릎 각도",      # 4
    "오른쪽 무릎 각도",    # 5
    "왼쪽 발목 각도",      # 6 (skip)
    "오른쪽 발목 각도",    # 7 (skip)
    "오른쪽 팔",    # 8
    "왼쪽 팔",      # 9
    "왼쪽 골반 각도",      # 10
    "오른쪽 골반 각도",    # 11
]

def generate_frame_feedback(
    kp_ref: np.ndarray,
    kp_user: np.ndarray,
    angle_thresh: float = np.deg2rad(5),
    direction_thresh: float = 0.2,
    reverse_thresh_deg: float = 20.0

) -> list[str]:
    """
    1) 손목(2,3), 발목(6,7) 제외
    2) 팔꿈치(0,1), 무릎(4,5), 어깨(8,9), 골반(10,11) 각각 매핑:
       - 팔꿈치/무릎: user_ang<ref_ang→펴세요, >→굽히세요, 벡터차 크면 '(방향 조절이 필요합니다.)'
       - 어깨:     user_ang<ref_ang→팔을 올리세요, >→팔을 내리세요
       - 골반:     user_ang<ref_ang→골반 각도가 넓습니다, >→골반 각도가 좁습니다
    3) 메시지 형식: "{label}를 {action} – 선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}°"
    """

    # interior angles & signed bend (rad)
    ref_ang   = calc_interior_angles_2d(kp_ref[None])[0]
    user_ang  = calc_interior_angles_2d(kp_user[None])[0]
    # (signed bend는 벡터 방향 차이 확인용, interior만 매핑하므로 사실 미사용)
    ref_bend  = calc_signed_bend_angles_2d(kp_ref[None])[0]
    user_bend = calc_signed_bend_angles_2d(kp_user[None])[0]

    feedback = []

    for i, (a, b, c) in enumerate(ANGLE_IDX):
        # 1) skip 손목/발목
        if i in {2, 3, 6, 7}:
            continue

        # 2) angle diff threshold
        angle_diff = abs(user_ang[i] - ref_ang[i])
        if angle_diff <= angle_thresh:
                continue

        # common vars
        teacher_deg = np.degrees(ref_ang[i])
        student_deg = np.degrees(user_ang[i])
        label       = JOINT_LABELS[i]

        stu_1dp = round(student_deg, 1)
        tea_1dp = round(teacher_deg, 1)

        if student_deg<teacher_deg: 
                diff_angle= tea_1dp - stu_1dp
        else:
                diff_angle= stu_1dp - tea_1dp

        # 팔꿈치 / 무릎
        if i in {0, 1, 4, 5}:
            # 벡터 방향 차이
            # vec_ref  = kp_ref[c, :2]  - kp_ref[b, :2]
            # vec_user = kp_user[c, :2] - kp_user[b, :2]
            # nr, nu   = np.linalg.norm(vec_ref), np.linalg.norm(vec_user)
            # suffix = ""

            # # signed bend raw (rad)
            # ref_b = ref_bend[i]
            # usr_b = user_bend[i]
            # # 안쪽(+) vs 바깥쪽(-) 완전 반전 감지
            # if ref_b * usr_b < 0:
            #     # 부호가 다르면 무조건 반전으로 간주
            #     suffix = " (방향 조절이 필요합니다.)"
            # else:
            suffix = ""

            # action 결정: user_ang<ref_ang→펴세요, else→굽히세요

            action = "펴세요" if user_ang[i] < ref_ang[i] else "굽히세요"
            feedback.append(
                f"{label}를 {action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

        # 어깨
        elif i in {8, 9}:
            # user_ang<ref_ang→올리세요, else→내리세요
            action = f"{label}을 올리세요" if user_ang[i] < ref_ang[i] else f"{label}을 내리세요"
            suffix = ""
            feedback.append(
                f"{action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

        # 골반
        elif i in {10, 11}:
            # user_ang<ref_ang→넓습니다, else→좁습니다
            action = f"{label} 각도가 넓습니다" if user_ang[i] < ref_ang[i] else f"{label} 각도가 좁습니다"
            suffix = ""
            feedback.append(
                f"{action}{suffix} – "
                f"선생님 {teacher_deg:.1f}° / 학생 {student_deg:.1f}° / 각도 차이 {diff_angle:.1f}°"
            )

    return feedback