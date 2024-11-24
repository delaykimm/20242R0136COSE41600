import open3d as o3d
import numpy as np
import time
import glob
import os
from sklearn.cluster import DBSCAN

class PedestrianDetector:
    def __init__(self):
        # 탐지 파라미터
        self.voxel_size = 0.05
        self.cluster_eps = 0.5
        self.min_points = 30
        self.human_height_range = (1.4, 2.0)
        self.human_width_range = (0.3, 1.0)
        
    def preprocess_pointcloud(self, pcd):
        print("[INFO] Preprocessing point cloud...")
        # 다운샘플링
        downsampled = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        if len(downsampled.points) == 0:
            return downsampled

        points = np.asarray(downsampled.points)
        
        # 로컬 영역별 높이 분석
        window_size = 1.0  # 로컬 윈도우 크기 (미터)
        min_points_in_window = 10  # 최소 포인트 수
        height_threshold = 0.15  # 높이 임계값
        
        # KD-tree 구성
        pcd_tree = o3d.geometry.KDTreeFlann(downsampled)
        non_floor_indices = []
        
        # 각 포인트에 대해 로컬 높이 분석
        for i, point in enumerate(points):
            # 로컬 영역 내의 포인트 검색
            [k, idx, _] = pcd_tree.search_radius_vector_3d(point, window_size)
            
            if k > min_points_in_window:
                local_points = points[idx]
                # 로컬 영역의 최저점으로부터의 높이 차이 계산
                local_min_height = np.min(local_points[:, 2])
                height_diff = point[2] - local_min_height
                
                if height_diff > height_threshold:
                    non_floor_indices.append(i)
        
        # 비바닥 점 선택
        non_floor_pcd = downsampled.select_by_index(non_floor_indices)
        return non_floor_pcd

    def detect_pedestrians(self, pcd):
        points = np.asarray(pcd.points)
        clusters = DBSCAN(eps=self.cluster_eps, min_samples=self.min_points).fit(points)
        labels = clusters.labels_
        
        pedestrian_clusters = []
        for label in set(labels):
            if label == -1:
                continue
                
            cluster_points = points[labels == label]
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            size = max_bound - min_bound
            
            height = size[2]
            width = max(size[0], size[1])
            
            if (self.human_height_range[0] <= height <= self.human_height_range[1] and
                self.human_width_range[0] <= width <= self.human_width_range[1]):
                pedestrian_clusters.append(cluster_points)
        
        return pedestrian_clusters

def set_camera_view(vis, center, size):
    """카메라 뷰포인트와 줌 레벨을 설정하는 함수"""
    view_control = vis.get_view_control()
    
    # # 포인트 클라우드의 중심과 크기 계산
    # # center = pcd.get_center()
    # max_bound = pcd.get_max_bound()
    # min_bound = pcd.get_min_bound()
    
    # # 중심점 계산 (최대값과 최소값의 평균)
    # center = (max_bound + min_bound) / 2.0  # [x_avg, y_avg, z_avg]
    # size = max_bound - min_bound
    
    # 기본 뷰 설정
    view_control.set_front([0, 0, 1])  # 거의 정면에서 보기
    view_control.set_lookat(center)     # 중심점 바라보기
    #print(center+[0,100,0])
    view_control.set_up([0, -1, 0])     # 상단 방향 설정(멀어지는각도/가까워지는뷰)
    
    # 줌 레벨 설정 (값이 작을수록 더 확대됨)
    view_control.set_zoom(0.2)  # 기존보다 더 가깝게
    
    # 카메라 거리 및 각도 조정
    distance = np.linalg.norm(size) * 0.3
    # view_control.set_front([0.2, 1, 0.5]) # for scenario 2
    # view_control.set_front([0.5, 1, 1]) # for scenario 4 - straight walk :1, zigzag walk :1
    view_control.set_front([0, 1, 1]) # for scenario 4 - straight walk :0.5, zigzag walk :1

def visualize_sequence(directory_path, point_size=1.0):
    # PCD 파일들을 정렬된 순서로 가져오기
    pcd_files = sorted(glob.glob(os.path.join(directory_path, "*.pcd")))
    
    if not pcd_files:
        print(f"'{directory_path}'에서 PCD 파일을 찾을 수 없습니다.")
        return
    
    # 상위 50개 파일만 선택 (170:220)
    pcd_files = pcd_files[300:390]
    print(f"총 {len(pcd_files)}개의 PCD 파일을 처리합니다.")
    
    # 시각화 윈도우 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1080)
    
    # 보행자 탐지기 초기화
    detector = PedestrianDetector()
    
    # 초기 중심점 및 크기 저장 변수
    initial_center = None
    initial_size = None
    # is_view_initialized = False
    saved_camera_params = None
    
    for i, pcd_file in enumerate(pcd_files):
        print(f"Visualizing: {pcd_file} ({i+1}/{len(pcd_files)})")
        
        try:
            # PCD 파일 로드
            pcd = o3d.io.read_point_cloud(pcd_file)
            if len(pcd.points) == 0:
                print(f"경고: {pcd_file}에 포인트가 없습니다.")
                continue
            
            # 첫 번째 프레임에서 중심점 계산
            if initial_center is None:
                max_bound = pcd.get_max_bound()
                min_bound = pcd.get_min_bound()
                initial_center = (max_bound + min_bound) / 2.0  # 첫 프레임에서 중심점 계산
                initial_size = max_bound - min_bound
                print(f"초기 중심점: {initial_center}")
            
            # 전처리 및 보행자 탐지
            processed_pcd = detector.preprocess_pointcloud(pcd)
            pedestrians = detector.detect_pedestrians(processed_pcd)
            print(f"탐지된 보행자 수: {len(pedestrians)}")
            
            # 시각화를 위한 포인트 클라우드 생성
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = pcd.points
            #vis_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 배경 포인트는 회색
            
            # 보행자 클러스터 추가 (빨간색)
            for cluster in pedestrians:
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
                cluster_pcd.paint_uniform_color([1, 0, 0])  # 보행자는 빨간색
                vis_pcd.points.extend(cluster_pcd.points)
                vis_pcd.colors.extend(cluster_pcd.colors)
            
            # 시각화 업데이트
            vis.clear_geometries()
            vis.add_geometry(vis_pcd)
            
            # 렌더링 옵션 설정
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            #render_option.background_color = np.array([0, 0, 0])  # 검은 배경
            
            # 첫 프레임에서만 카메라 뷰 초기화
            view_control = vis.get_view_control()
            if saved_camera_params is None:
                saved_camera_params = set_camera_view(vis, initial_center, initial_size)
            else:
                # 이후 프레임에서는 저장된 카메라 매개변수 복원
                view_control.convert_from_pinhole_camera_parameters(saved_camera_params)
            
            # 렌더링
            vis.poll_events()
            vis.update_renderer()
            
            # 프레임 간 대기
            time.sleep(0.2)
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            continue
    
    # 시각화 종료
    vis.destroy_window()

def main():
    try:
        print("\n연속 프레임 시각화 시작...")
        sequence_directory = "data/05_straight_duck_walk/pcd"
        visualize_sequence(sequence_directory, point_size=1.0)
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("프로그램 종료")

if __name__ == "__main__":
    main()