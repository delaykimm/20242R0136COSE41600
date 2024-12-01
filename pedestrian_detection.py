import os
import numpy as np
import open3d as o3d
import time
from typing import Dict, List, Tuple

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
            print(f"- 데이터 형태: {points.shape}")
            
        except Exception as e:
            print(f"파일 로드 중 오류 발생 {file_name}: {str(e)}")
            continue
    
    return non_floor_points_db

def visualize_sequence(non_floor_points_db: Dict[str, np.ndarray], 
                      delay: float = 0.5,
                      window_name: str = "Non-floor Points Sequence"):
    """
    비바닥 포인트 클라우드 시퀀스를 연속적으로 시각화합니다.
    """
    # Visualizer 초기화
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=800, height=600)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # 검은색 배경
    opt.point_size = 1.0  # 포인트 크기
    
    # 초기 포인트 클라우드 객체 생성
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)  # 처음 한 번만 추가
    
    try:
        # 정렬된 파일 목록
        sorted_files = sorted(non_floor_points_db.keys())
        
        for idx, file_name in enumerate(sorted_files):
            # 현재 프레임 정보 출력
            print(f"\r프레임 {idx+1}/{len(sorted_files)}: {file_name}", end="")
            
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

if __name__ == "__main__":
    # 설정
    data_directory = "data"
    target_folder = "04_zigzag_walk"
    
    # 비바닥 포인트 클라우드 데이터 로드
    print(f"데이터 로드 시작: {target_folder}")
    non_floor_points_db = load_non_floor_points(data_directory, target_folder)
    
    if not non_floor_points_db:
        print("\n경고: 처리된 데이터를 찾을 수 없습니다!")
        exit()
    
    # 데이터 통계 출력
    print("\n=== 로드된 데이터 통계 ===")
    print(f"총 파일 수: {len(non_floor_points_db)}")
    
    total_points = 0
    for file_name, points in non_floor_points_db.items():
        total_points += len(points)
        print(f"\n{file_name}:")
        print(f"- 포인트 수: {len(points):,}")
        if len(points) > 0:
            print(f"- 좌표 범위:")
            print(f"  X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}]")
            print(f"  Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}]")
            print(f"  Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")
    
    print(f"\n총 포인트 수: {total_points:,}")
    
    # 연속 시각화 시작
    print("\n시퀀스 시각화 시작...")
    visualize_sequence(non_floor_points_db, delay=0.1)  # delay를 0.1초로 설정
