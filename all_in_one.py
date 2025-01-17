import os
import numpy as np
import open3d as o3d
import time
from typing import Dict, List, Tuple
from read_pcd_file import collect_folder_pcd_data, process_pcd_database

def load_non_floor_points(data_dir: str, target_folder: str) -> Dict[str, np.ndarray]:
    """
    처리된 비바닥 포인트 클라우드 데이터를 로드합니다.
    
    Parameters:
        data_dir: 데이터 기본 경로
        target_folder: 처리할 폴더명 (예: "04_zigzag_walk")
    
    Returns:
        Dict[str, np.ndarray]: 파일명을 키로 하고 포인트 클라우드 데이터를 값으로 하는 딕셔너리
    """
    non_floor_points_db = {}
    processed_dir = os.path.join(data_dir, target_folder, "processed")
    
    # 폴더 존재 확인
    if not os.path.exists(processed_dir):
        print(f"오류: 처리된 데이터 폴더가 존재하지 않습니다: {processed_dir}")
        return non_floor_points_db
    
    print(f"처리된 데이터 폴더 로드 중: {processed_dir}")
    
    # .npy 파일 목록 가져오기
    npy_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('_non_floor.npy')])
    if not npy_files:
        print(f"오류: 처리된 데이터 파일을 찾을 수 없습니다.")
        return non_floor_points_db
    
    print(f"발견된 처리된 파일 수: {len(npy_files)}")
    
    # 각 파일 로드
    for file_name in npy_files:
        try:
            # 원본 PCD 파일명 복원
            original_pcd_name = file_name.replace('_non_floor.npy', '.pcd')
            
            # 데이터 로드
            file_path = os.path.join(processed_dir, file_name)
            points = np.load(file_path)
            
            # 딕셔너리에 저장
            non_floor_points_db[original_pcd_name] = points
            
            print(f"로드 완료: {original_pcd_name}")
            print(f"- 포인트 수: {len(points):,}")
            
        except Exception as e:
            print(f"파일 로드 중 오류 발생 ({file_name}): {str(e)}")
            continue
    
    return non_floor_points_db

def load_or_process_data(data_dir: str, target_folder: str, voxel_size: float = 0.2) -> Dict[str, np.ndarray]:
    """
    데이터를 로드하거나 처리합니다. 처리된 데이터가 없으면 새로 처리합니다.
    """
    processed_dir = os.path.join(data_dir, target_folder, "processed")
    
    # 1. 처리된 데이터가 있는지 확인
    if os.path.exists(processed_dir):
        print("처리된 데이터 폴더 발견. 데이터 로드 시도...")
        non_floor_points_db = load_non_floor_points(data_dir, target_folder)
        if non_floor_points_db:
            print("기존 처리 데이터 로드 완료!")
            return non_floor_points_db
    
    # 2. 처리된 데이터가 없으면 새로 처리
    print("처리된 데이터가 없습니다. 새로 처리를 시작합니다...")
    
    # PCD 파일 수집
    print(f"\n1. PCD 파일 수집 중: {target_folder}")
    pcd_database = collect_folder_pcd_data(data_dir, target_folder, voxel_size)
    
    if not pcd_database:
        raise ValueError("PCD 데이터를 찾을 수 없습니다!")
    
    # 전처리 작업 수행
    print("\n2. 전처리 작업 시작...")
    _, non_floor_points_db = process_pcd_database(pcd_database, voxel_size)
    
    # 결과 저장
    print("\n3. 처리 결과 저장 중...")
    os.makedirs(processed_dir, exist_ok=True)
    for file_name, points in non_floor_points_db.items():
        save_path = os.path.join(processed_dir, f"{os.path.splitext(file_name)[0]}_non_floor.npy")
        np.save(save_path, points)
        print(f"장 완료: {save_path}")
    
    return non_floor_points_db

def detect_moving_points(non_floor_points_db: Dict[str, np.ndarray], threshold: float = 0.1) -> Dict[str, np.ndarray]:
    """
    연속된 프레임에서 이동하는 점들을 검출합니다.
    """
    moving_points_db = {}
    sorted_files = sorted(non_floor_points_db.keys())
    
    print("\n이동 점 검출 시작...")
    
    # 첫 번째 프레임은 이전 프레임이 없으므로 빈 배열로 설정
    moving_points_db[sorted_files[0]] = np.zeros((0, 3))
    prev_points = non_floor_points_db[sorted_files[0]]
    
    for idx in range(1, len(sorted_files)):
        current_file = sorted_files[idx]
        current_points = non_floor_points_db[current_file]
        
        # 이동 점 검출
        pcd_prev = o3d.geometry.PointCloud()
        pcd_prev.points = o3d.utility.Vector3dVector(prev_points)
        tree = o3d.geometry.KDTreeFlann(pcd_prev)
        
        moving_points = []
        for point in current_points:
            _, idx_arr, dist = tree.search_knn_vector_3d(point, 1)
            nearest_point = prev_points[idx_arr[0]]
            if np.linalg.norm(point - nearest_point) > threshold:
                moving_points.append(point)
        
        moving_points = np.array(moving_points) if moving_points else np.zeros((0, 3))
        moving_points_db[current_file] = moving_points
        
        print(f"\r프레임 {idx}/{len(sorted_files)-1}: "
              f"이동 점 {len(moving_points):,}개 검출", end="")
        
        prev_points = current_points
    
    print("\n이동 점 검출 완료!")
    return moving_points_db

def detect_moving_clusters(non_floor_points_db: Dict[str, np.ndarray], 
                         threshold: float = 0.3) -> Dict[str, np.ndarray]:
    """
    클러스터의 중심점 이동을 기반으로 움직이는 클러스터를 검출합니다.
    """
    moving_clusters_db = {}
    sorted_files = sorted(non_floor_points_db.keys())
    prev_centers = None
    
    print("\n클러스터 기반 이동 객체 검출 시작...")
    
    # 첫 번째 프레임은 이전 프레임이 없으므로 빈 배열로 설정
    moving_clusters_db[sorted_files[0]] = np.zeros((0, 3))
    
    for idx in range(len(sorted_files)):
        current_file = sorted_files[idx]
        current_points = non_floor_points_db[current_file]
        
        # 현재 프레임의 포인트 클라우드 생성
        current_pcd = o3d.geometry.PointCloud()
        current_pcd.points = o3d.utility.Vector3dVector(current_points)
        
        # DBSCAN 클러스터링 수행
        labels = np.array(current_pcd.cluster_dbscan(eps=0.4, min_points=20))
        unique_labels = np.unique(labels[labels != -1])
        
        # 현재 프레임의 클러스터 중심점 계산
        current_centers = []
        cluster_points_list = []
        
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points = current_points[cluster_mask]
            cluster_center = np.mean(cluster_points, axis=0)
            current_centers.append(cluster_center)
            cluster_points_list.append(cluster_points)
        
        # 이동 클러스터 검출
        moving_points = []
        if prev_centers is not None and current_centers:  # 리스트가 비어있지 않은지 확인
            current_centers_array = np.array(current_centers)
            prev_centers_array = np.array(prev_centers)
            
            if prev_centers_array.size > 0:  # NumPy 배열이 비어있지 않은지 확인
                for i, cluster_points in enumerate(cluster_points_list):
                    # 현재 클러스터 중심점과 이전 프레임의 모든 클러스터 중심점들 간의 최소 거리 계산
                    distances = np.linalg.norm(current_centers_array[i] - prev_centers_array, axis=1)
                    min_dist = np.min(distances)
                    if min_dist > threshold:
                        moving_points.extend(cluster_points)
        
        moving_clusters_db[current_file] = np.array(moving_points) if moving_points else np.zeros((0, 3))
        prev_centers = current_centers  # 리스트 형태로 저장
        
        print(f"\r프레임 {idx+1}/{len(sorted_files)}: "
              f"클러스터 수 {len(current_centers)}, "
              f"이동 점 수 {len(moving_points)}", end="")
    
    print("\n클러스터 기반 이동 객체 검출 완료!")
    return moving_clusters_db

def check_cluster_conditions(points: np.ndarray, 
                           min_width: float = 0.2,
                           max_width: float = 1.6,
                           min_height: float = 0.3,
                           max_height: float = 1.9,
                           max_ground_height: float = 0.1,
                           min_points: int = 10,
                           max_points: int = 60,
                           min_center_height: float = 0.2,
                           max_center_height: float = 1.0) -> bool:
    """
    클러스터가 모든 조건을 만족하는지 확인합니다.
    1. 너비(x, y축)가 min_width 이상 max_width 미만
    2. 클러스터의 최소 z값이 max_ground_height보다 작음
    3. 클러스터의 높이가 min_height 이상 max_height 이하
    4. 점의 개수가 min_points 이상 max_points 미만
    5. 클러스터 중심의 높이가 min_center_height 이상 max_center_height 이하
    """
    if len(points) == 0 or len(points) >= max_points or len(points) <= min_points:
        return False
    
    # 최소/최대 좌표 계산
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    sizes = max_coords - min_coords
    
    # 1. 너비 조건 체크 (x, y축)
    width_x = sizes[0]  # x축 너비
    width_y = sizes[1]  # y축 너비
    width_check = (min_width <= width_x < max_width) and (min_width <= width_y < max_width)
    
    # 2. 최소 z값 조건 체크 (지면과의 연결 확인)
    min_z = min_coords[2]
    ground_check = min_z < max_ground_height
    
    # 3. 높이 크기 조건 체크
    height = sizes[2]  # z축 크기
    height_check = min_height <= height <= max_height
    
    # 4. 클러스터 중심 높이 체크
    center = np.mean(points, axis=0)
    center_height = center[2]
    center_height_check = min_center_height <= center_height <= max_center_height
    
    return width_check and ground_check and height_check and center_height_check

def visualize_sequence(non_floor_points_db: Dict[str, np.ndarray],
                      moving_points_db: Dict[str, np.ndarray],
                      moving_clusters_db: Dict[str, np.ndarray],
                      delay: float = 0.1):
    """
    최적화된 포인트 클라우드 시각화 함수
    """
    # 1. 초기 설정
    vis = o3d.visualization.Visualizer()
    vis.create_window("Moving Points Detection", width=1024, height=768)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.0
    
    # 포인트 클라우드 객체 생성
    static_pcd = o3d.geometry.PointCloud()
    point_moving_pcd = o3d.geometry.PointCloud()
    cluster_moving_pcd = o3d.geometry.PointCloud()
    
    # 색상 설정
    static_color = np.array([0.7, 0.7, 0.7])      # 회색
    point_moving_color = np.array([1, 0, 0])      # 빨간색
    cluster_moving_color = np.array([0, 1, 0])    # 초록색
    
    vis.add_geometry(static_pcd)
    vis.add_geometry(point_moving_pcd)
    vis.add_geometry(cluster_moving_pcd)
    
    # 바운딩 박스를 저장할 리스트
    bounding_boxes = []
    
    # 클러스터 조건 체크 함수 정의
    def check_cluster_size(points: np.ndarray) -> bool:
        if len(points) == 0:
            return False
        
        # 최소/최대 좌표 계산
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        sizes = max_coords - min_coords
        
        # 너비 조건 체크 (x, y축)
        width_x = sizes[0]
        width_y = sizes[1]
        width_check = width_x <= 2.0 and width_y <= 2.0
        
        # 높이 조건 체크 (z축)
        height = sizes[2]
        height_check = height <= 2.0
        
        # 최저 z값 조건 체크
        min_z = min_coords[2]
        ground_check = min_z <= 0.2
        
        return width_check and height_check and ground_check
    
    try:
        sorted_files = sorted(non_floor_points_db.keys())
        total_frames = len(sorted_files)
        
        for idx, file_name in enumerate(sorted_files):
            current_points = non_floor_points_db[file_name]
            point_moving = moving_points_db[file_name]
            cluster_moving = moving_clusters_db[file_name]
            
            # 이전 프레임의 바운딩 박스 제거
            # for box in bounding_boxes:
            #     vis.remove_geometry(box, False)
            # bounding_boxes.clear()
            
            # 클러스터 조건 체크 및 바운딩 박스 생성
            if len(cluster_moving) > 0:
                cluster_pcd = o3d.geometry.PointCloud()
                cluster_pcd.points = o3d.utility.Vector3dVector(cluster_moving)
                labels = np.array(cluster_pcd.cluster_dbscan(eps=0.4, min_points=20))
                
                valid_cluster_points = []
                unique_labels = np.unique(labels[labels != -1])
                
                for label in unique_labels:
                    cluster_mask = labels == label
                    cluster_points = cluster_moving[cluster_mask]
                    
                    # 수정된 클러스터 조건 체크
                    if check_cluster_size(cluster_points):
                        # 클러스터의 바운딩 박스 생성
                        min_bound = np.min(cluster_points, axis=0)
                        max_bound = np.max(cluster_points, axis=0)
                        center = (min_bound + max_bound) / 2
                        extent = max_bound - min_bound
                        
                        # 바운딩 박스 생성
                        box = o3d.geometry.OrientedBoundingBox()
                        box.center = center
                        box.extent = extent
                        box.R = np.eye(3)  # 회전 없음
                        box.color = np.array([0, 1, 0])  # 초록색
                        
                        # 바운딩 박스 추가
                        vis.add_geometry(box, False)
                        bounding_boxes.append(box)
                        
                        # 디버깅을 위한 클러스터 정보 출력
                        sizes = max_bound - min_bound
                        print(f"\n클러스터 정보:"
                              f"\n - 너비(x)={sizes[0]:.2f}m"
                              f"\n - 너비(y)={sizes[1]:.2f}m"
                              f"\n - 높이={sizes[2]:.2f}m"
                              f"\n - 최저 z={min_bound[2]:.2f}m")
                        
                        valid_cluster_points.extend(cluster_points)
                
                cluster_moving = np.array(valid_cluster_points) if valid_cluster_points else np.zeros((0, 3))
            
            # 정적 점 분리
            if len(point_moving) > 0 or len(cluster_moving) > 0:
                moving_points = np.vstack([point_moving, cluster_moving]) if len(point_moving) > 0 and len(cluster_moving) > 0 else \
                              point_moving if len(point_moving) > 0 else cluster_moving
                
                moving_pcd = o3d.geometry.PointCloud()
                moving_pcd.points = o3d.utility.Vector3dVector(moving_points)
                tree = o3d.geometry.KDTreeFlann(moving_pcd)
                
                static_mask = np.ones(len(current_points), dtype=bool)
                
                for i in range(len(current_points)):
                    _, _, dist = tree.search_knn_vector_3d(current_points[i], 1)
                    if np.asarray(dist)[0] < 0.09:
                        static_mask[i] = False
                
                static_points = current_points[static_mask]
            else:
                static_points = current_points
            
            # 포인트 클라우드 업데이트
            static_pcd.points = o3d.utility.Vector3dVector(static_points)
            static_pcd.colors = o3d.utility.Vector3dVector(np.tile(static_color, (len(static_points), 1)))
            
            point_moving_pcd.points = o3d.utility.Vector3dVector(point_moving)
            point_moving_pcd.colors = o3d.utility.Vector3dVector(np.tile(point_moving_color, (len(point_moving), 1)))
            
            cluster_moving_pcd.points = o3d.utility.Vector3dVector(cluster_moving)
            cluster_moving_pcd.colors = o3d.utility.Vector3dVector(np.tile(cluster_moving_color, (len(cluster_moving), 1)))
            
            # 상태 출력
            print(f"\r프레임 {idx+1}/{total_frames}: "
                  f"정적: {len(static_points):,}, "
                  f"점 이동: {len(point_moving):,}, "
                  f"유효 클러스터 이동: {len(cluster_moving):,}", end="", flush=True)
            
            if idx == 0:
                vis.reset_view_point(True)
            
            # 지오메트리 업데이트
            vis.update_geometry(static_pcd)
            vis.update_geometry(point_moving_pcd)
            vis.update_geometry(cluster_moving_pcd)
            
            vis.poll_events()
            vis.update_renderer()
            
            if delay > 0:
                time.sleep(delay)
            
    except Exception as e:
        print(f"\n시각화 중 오류 발생: {str(e)}")
        raise
    finally:
        vis.destroy_window()
        print("\n시각화 종료")

def main():
    """
    메인 실행 함수
    """
    # 설정
    data_directory = "data"
    target_folder = "07_straight_walk"
    voxel_size = 0.2
    
    try:
        # 1. 데이터 로드 또는 처리
        print("=== 데이터 준비 중 ===")
        non_floor_points_db = load_or_process_data(data_directory, target_folder, voxel_size)
        
        # 2. 이동 점 검출 (점 단위)
        print("\n=== 점 단위 이동 검출 중 ===")
        moving_points_db = detect_moving_points(non_floor_points_db, threshold=0.15)
        
        # 3. 이동 클러스터 검출
        print("\n=== 클러스터 단위 이동 검출 중 ===")
        moving_clusters_db = detect_moving_clusters(non_floor_points_db, threshold=0.03)
        
        # 4. 데이터 통계 출력
        print("\n=== 데이터 통계 ===")
        print(f"총 프레임 수: {len(non_floor_points_db)}")
        total_points = sum(len(points) for points in non_floor_points_db.values())
        total_moving_points = sum(len(points) for points in moving_points_db.values())
        total_moving_clusters = sum(len(points) for points in moving_clusters_db.values())
        print(f"총 포인트 수: {total_points:,}")
        print(f"점 단위 이동 수: {total_moving_points:,}")
        print(f"클러스터 단위 이동 수: {total_moving_clusters:,}")
        
        # 5. 시각화
        print("\n=== 시각화 시작 ===")
        visualize_sequence(non_floor_points_db, moving_points_db, moving_clusters_db, delay=0.1)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()