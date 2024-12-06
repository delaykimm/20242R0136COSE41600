import os
import open3d as o3d
import numpy as np
from preprocessing import process_point_cloud  # preprocessing.py의 함수를 임포트

def collect_folder_pcd_data(data_dir, target_folder, voxel_size=0.3):
    """
    특정 폴더의 모든 PCD 파일을 다운샘플링하여 딕셔너리에 저장
    
    Parameters:
        data_dir: 데이터 기본 경로
        target_folder: 처리할 폴더명 (예: "04_zigzag_walk")
        voxel_size: 다운샘플링 크기
    """
    pcd_data = {}
    
    # 대상 폴더 경로 생성
    folder_path = os.path.join(data_dir, target_folder, "pcd")
    
    # 폴더 존재 확인
    if not os.path.exists(folder_path):
        print(f"오류: 폴더가 존재하지 않습니다: {folder_path}")
        return pcd_data
    
    print(f"처리 중인 폴더: {folder_path}")
    
    # PCD 파일 목록 가져오기
    pcd_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pcd')])
    if not pcd_files:
        print(f"오류: PCD 파일을 찾을 수 없습니다.")
        return pcd_data
    
    print(f"발견된 PCD 파일 수: {len(pcd_files)}")
    
    # 각 PCD 파일 처리
    for file_name in pcd_files:
        file_path = os.path.join(folder_path, file_name)
        
        try:
            # PCD 파일 읽기
            pcd = o3d.io.read_point_cloud(file_path)
            
            if len(pcd.points) == 0:
                print(f"경고: {file_name}에 포인트가 없습니다.")
                continue
            
            # 원본 포인트 수 저장
            original_points = len(pcd.points)
            
            # 다운샘플링 적용
            downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
            
            # Radius Outlier Removal 적용
            cl, ind = downsampled_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
            cleaned_pcd = downsampled_pcd.select_by_index(ind)
            points = np.asarray(cleaned_pcd.points)
            
            # 딕셔너리에 저장
            key = file_name
            pcd_data[key] = points
            
            print(f"처리 완료: {key}")
            print(f"- 원본 포인트 수: {original_points:,}")
            print(f"- 다운샘플링 후 포인트 수: {len(downsampled_pcd.points):,}")
            print(f"- ROR 후 포인트 수: {len(points):,}")
            print(f"- 총 감소율: {((original_points - len(points)) / original_points) * 100:.2f}%")
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생 {file_name}: {str(e)}")
    
    return pcd_data

def process_pcd_database(pcd_database, voxel_size=0.3):
    """
    저장된 PCD 데이터베이스의 각 파일에 대해 전처리 작업 수행
    """
    processed_results = {}
    non_floor_points_db = {}  # 바닥면이 제거된 포인트 클라우드를 저장할 딕셔너리
    
    for file_name, points in pcd_database.items():
        print(f"\n처리 중인 파일: {file_name}")
        
        # numpy 배열을 포인트 클라우드로 변환
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # 전처리 작업 수행
        try:
            clusters, non_floor_pcd = process_point_cloud(pcd, voxel_size)
            
            # 클러스터 정보 저장
            processed_results[file_name] = clusters
            print(f"- 클러스터 수: {len(clusters)}")
            
            # 바닥면이 제거된 포인트 클라우드 저장
            non_floor_points = np.asarray(non_floor_pcd.points)
            non_floor_points_db[file_name] = non_floor_points
            print(f"- 바닥면 제거 후 포인트 수: {len(non_floor_points):,}")
            
        except Exception as e:
            print(f"처리 중 오류 발생 ({file_name}): {str(e)}")
            continue
    
    return processed_results, non_floor_points_db

if __name__ == "__main__":
    # 설정
    data_directory = "data"
    target_folder = "04_zigzag_walk"
    voxel_size = 0.2
    
    # 1. PCD 파일 데이터 수집
    print(f"데이터 수집 시작: {target_folder}")
    pcd_database = collect_folder_pcd_data(data_directory, target_folder, voxel_size)
    
    if not pcd_database:
        print("\n경고: 데이터를 찾을 수 없습니다!")
        exit()
    
    # 2. 전처리 작업 수행
    print("\n전처리 작업 시작...")
    processed_results, non_floor_points_db = process_pcd_database(pcd_database, voxel_size)
    
    # 3. 결과 출력
    print("\n=== 처리 결과 ===")
    print(f"처리된 파일 수: {len(processed_results)}")
    for file_name in processed_results.keys():
        print(f"\n{file_name}:")
        print(f"- 클러스터 수: {len(processed_results[file_name])}")
        print(f"- 바닥면 제거 후 포인트 수: {len(non_floor_points_db[file_name]):,}")
    
    # 4. 결과 저장 (선택사항)
    # numpy 파일로 저장하려면 다음과 같이 할 수 있습니다:
    save_dir = os.path.join(data_directory, target_folder, "processed")
    os.makedirs(save_dir, exist_ok=True)
    
    for file_name, points in non_floor_points_db.items():
        save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_non_floor.npy")
        np.save(save_path, points)
        print(f"저장 완료: {save_path}")
