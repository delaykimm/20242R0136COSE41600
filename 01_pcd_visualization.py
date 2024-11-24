# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

# pcd 파일 불러오기, 필요에 맞게 경로 수정
#file_path = "test_data/1727320101-665925967.pcd"
file_path = "data/05_straight_duck_walk/pcd/pcd_000370.pcd"
# pcd 파일 불러오고 시각화하는 함수
def load_and_visualize_pcd(file_path, point_size=1.0):
    # pcd 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"Point cloud has {len(pcd.points)} points.")
    
    # 시각화 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


# PCD 파일 불러오기 및 데이터 확인 함수
def load_and_inspect_pcd(file_path):
    # PCD 파일 로드
    pcd = o3d.io.read_point_cloud(file_path)
    
    # 점 구름 데이터를 numpy 배열로 변환
    points = np.asarray(pcd.points)
    
    # 점 데이터 개수 및 일부 점 확인
    print(f"Number of points: {len(points)}")
    print("First 5 points:")
    print(points[:5])  # 처음 5개의 점 출력
    
    # 점의 x, y, z 좌표의 범위 확인
    print("X coordinate range:", np.min(points[:, 0]), "to", np.max(points[:, 0]))
    print("Y coordinate range:", np.min(points[:, 1]), "to", np.max(points[:, 1]))
    print("Z coordinate range:", np.min(points[:, 2]), "to", np.max(points[:, 2]))

def visualize_sequence(directory_path, point_size=1.0):
    # PCD 파일들을 정렬된 순서로 가져오기
    pcd_files = sorted(glob.glob(os.path.join(directory_path, "*.pcd")))
    
    if not pcd_files:
        print(f"'{directory_path}'에서 PCD 파일을 찾을 수 없습니다.")
        return
    
# 상위 70개 파일만 선택
    pcd_files = pcd_files[170:220]
        
    print(f"총 {len(pcd_files)}개의 PCD 파일을 처리합니다.")
    
    # 시각화 윈도우 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    for pcd_file in pcd_files:
        print(f"Visualizing: {pcd_file}")
        
        # PCD 파일 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        print(f"Point cloud has {len(pcd.points)} points.")
        
        # 시각화 업데이트
        vis.clear_geometries()
        vis.add_geometry(pcd)
        
        # 렌더링 옵션 설정
        render_option = vis.get_render_option()
        render_option.point_size = point_size
        
        # # 카메라 뷰포인트 설정
        # center = pcd.get_center()
        # max_bound = pcd.get_max_bound()
        # min_bound = pcd.get_min_bound()
        # size = max_bound - min_bound
        # # # 중심점을 기준으로 이동
        # # pcd.translate(-center)
        
        # # view_control = vis.get_view_control()
        # # view_control.set_front([0, 0, -1])
        # # view_control.set_lookat(center)
        # # view_control.set_up([0, -1, 0])
        
        # # 줌 레벨 설정 (값이 작을수록 더 확대됨)
        # view_control.set_zoom(0.5)  # 기존 0.5에서 0.3으로 변경하여 더 확대
        
        # # 카메라 거리 조정 (선택사항)
        # distance = np.linalg.norm(size) * 0.5  # 기존보다 더 가깝게 설정
        # view_control.set_front([-distance, 0, -distance])
        
        # 렌더링
        vis.poll_events()
        vis.update_renderer()
        
        # 프레임 간 대기
        time.sleep(0.3)
    
    # 시각화 종료
    vis.destroy_window()

if __name__ == "__main__":
    # 단일 파일 시각화
    single_file_path = "data/04_zigzag_walk/pcd/pcd_000070.pcd"
    print("단일 파일 시각화 및 분석:")
    load_and_visualize_pcd(single_file_path, 0.5)
    load_and_inspect_pcd(single_file_path)
    
    # # 시퀀스 시각화
    # print("\n연속 프레임 시각화 시작...") 
    # sequence_directory = "data/04_zigzag_walk/pcd"  # PCD 파일들이 있는 디렉토리 경로
    # visualize_sequence(sequence_directory, point_size=1.0)

# pcd 시각화 테스트
# load_and_visualize_pcd(file_path, 0.5)
# load_and_inspect_pcd(file_path)
