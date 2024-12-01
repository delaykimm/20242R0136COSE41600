import os
import numpy as np
import open3d as o3d
import time
from typing import Dict, List, Tuple
from read_pcd_file import collect_folder_pcd_data, process_pcd_database

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

def visualize_sequence(non_floor_points_db: Dict[str, np.ndarray], 
                      delay: float = 0.1,
                      window_name: str = "Pedestrian Detection Sequence"):
    """
    비바닥 포인트 클라우드 시퀀스를 연속적으로 시각화합니다.
    """
    # Visualizer 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1024, height=768)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
    opt.point_size = 2.0  # 포인트 크기
    
    # 초기 포인트 클라우드 객체 생성
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    
    try:
        # 정렬된 파일 목록
        sorted_files = sorted(non_floor_points_db.keys())
        total_frames = len(sorted_files)
        
        print("\n=== 시각화 컨트롤 ===")
        print("- ESC: 종료")
        print(f"- 총 프레임 수: {total_frames}")
        
        for idx, file_name in enumerate(sorted_files):
            # 현재 프레임 정보 출력
            print(f"\r프레임 {idx+1}/{total_frames}: {file_name}", end="")
            
            # 포인트 클라우드 업데이트
            points = non_floor_points_db[file_name]
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.paint_uniform_color([1, 0, 0])  # 빨간색
            
            # 첫 프레임에서만 카메라 위치 최적화
            if idx == 0:
                vis.reset_view_point(True)
            
            # 뷰 업데이트
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            
            # 지연 시간
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
    target_folder = "06_straight_crawl"
    voxel_size = 0.2
    
    try:
        # 1. 데이터 로드 또는 처리
        print("=== 데이터 준비 중 ===")
        non_floor_points_db = load_or_process_data(data_directory, target_folder, voxel_size)
        
        # 2. 데이터 통계 출력
        print("\n=== 데이터 통계 ===")
        print(f"총 프레임 수: {len(non_floor_points_db)}")
        total_points = sum(len(points) for points in non_floor_points_db.values())
        print(f"총 포인트 수: {total_points:,}")
        
        # 3. 시각화
        print("\n=== 시각화 시작 ===")
        visualize_sequence(non_floor_points_db, delay=0.1)
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return

if __name__ == "__main__":
    main()