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
        print(f"저장 완료: {save_path}")
    
    return non_floor_points_db

def detect_moving_points(non_floor_points_db: Dict[str, np.ndarray], threshold: float = 0.2) -> Dict[str, np.ndarray]:
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

def visualize_sequence(non_floor_points_db: Dict[str, np.ndarray],
                      moving_points_db: Dict[str, np.ndarray],
                      delay: float = 0.1,
                      window_name: str = "Moving Points Detection"):
    """
    비바닥 포인트 클라우드 시퀀스를 연속적으로 시각화하며 이동하는 점들을 강조합니다.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 1.0
    
    # 정적/이동 포인트 클라우드 객체 생성
    static_pcd = o3d.geometry.PointCloud()
    moving_pcd = o3d.geometry.PointCloud()
    
    vis.add_geometry(static_pcd)
    vis.add_geometry(moving_pcd)
    
    try:
        sorted_files = sorted(non_floor_points_db.keys())
        total_frames = len(sorted_files)
        
        print("\n=== 시각화 컨트롤 ===")
        print("- ESC: 종료")
        print(f"- 총 프레임 수: {total_frames}")
        
        for idx, file_name in enumerate(sorted_files):
            current_points = non_floor_points_db[file_name]
            moving_points = moving_points_db[file_name]
            
            # 정적인 점들 분리
            if len(moving_points) > 0:
                static_mask = np.ones(len(current_points), dtype=bool)
                for moving_point in moving_points:
                    distances = np.linalg.norm(current_points - moving_point, axis=1)
                    static_mask &= (distances > 0.3)  # 이동 점 주변 점들도 제외
                static_points = current_points[static_mask]
            else:
                static_points = current_points
            
            # 포인트 클라우드 업데이트
            static_pcd.points = o3d.utility.Vector3dVector(static_points)
            static_pcd.paint_uniform_color([0.7, 0.7, 0.7])  # 회색
            
            moving_pcd.points = o3d.utility.Vector3dVector(moving_points)
            moving_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
            
            print(f"\r프레임 {idx+1}/{total_frames}: {file_name} "
                  f"(이동: {len(moving_points):,}, 정적: {len(static_points):,})", end="")
            
            if idx == 0:
                vis.reset_view_point(True)
            
            vis.update_geometry(static_pcd)
            vis.update_geometry(moving_pcd)
            vis.poll_events()
            vis.update_renderer()
            
            time.sleep(delay)
            
    except Exception as e:
        print(f"\n시각화 중 오류 발생: {str(e)}")
    finally:
        vis.destroy_window()
        print("\n시각화 종료")

def main():
    """
    메인 실행 함수
    """
    # 설정
    data_directory = "data"
    target_folder = "04_zigzag_walk"
    voxel_size = 0.2
    
    try:
        # 1. 데이터 로드 또는 처리
        print("=== 데이터 준비 중 ===")
        non_floor_points_db = load_or_process_data(data_directory, target_folder, voxel_size)
        
        # 2. 이동 점 검출
        print("\n=== 이동 점 검출 중 ===")
        moving_points_db = detect_moving_points(non_floor_points_db, threshold=0.15)
        
        # 3. 데이터 통계 출력
        print("\n=== 데이터 통계 ===")
        print(f"총 프레임 수: {len(non_floor_points_db)}")
        total_points = sum(len(points) for points in non_floor_points_db.values())
        total_moving = sum(len(points) for points in moving_points_db.values())
        print(f"총 포인트 수: {total_points:,}")
        print(f"총 이동 점 수: {total_moving:,}")
        
        # 4. 시각화
        print("\n=== 시각화 시작 ===")
        visualize_sequence(non_floor_points_db, moving_points_db, delay=0.1)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()