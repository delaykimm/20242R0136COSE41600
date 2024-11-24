# 시각화에 필요한 라이브러리 불러오기
import os
import glob
import time

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def process_single_frame(pcd_path, eps=0.2, min_points=10):
    # pcd 파일 불러오기, 필요에 맞게 경로 수정
    file_path = "test_data/1727320101-665925967.pcd"
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)

    # Voxel Downsampling 수행
    voxel_size = 0.1  # 필요에 따라 voxel 크기를 조정
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Radius Outlier Removal (ROR) 적용
    cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    # RANSAC을 사용하여 평면 추정
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.15,
                                                ransac_n=3,
                                                num_iterations=2000)

    # 도로에 속하지 않는 포인트 (outliers) 추출
    final_point = ror_pcd.select_by_index(inliers, invert=True)

    # DBSCAN 클러스터링 적용 (개선 여지 다분함!!!!!!!)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(final_point.cluster_dbscan(eps=0.6, min_points=11, print_progress=True))

    # 각 클러스터를 색으로 표시
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    # 노이즈를 제거하고 각 클러스터에 색상 지정
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0  # 노이즈는 검정색으로 표시
    final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # 포인트 클라우드 시각화 함수
    def visualize_point_cloud_with_point_size(pcd, window_name="Point Cloud Visualization", point_size=1.0):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name)
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = point_size
        vis.run()
        vis.destroy_window()

    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)
    visualize_point_cloud_with_point_size(final_point, 
                                        window_name="DBSCAN Clustered Points", point_size=2.0)
def process_and_visualize_sequence(directory_path, eps=0.2, min_points=10):
    # PCD 파일들을 정렬된 순서로 가져오기
    pcd_files = sorted(glob.glob(os.path.join(directory_path, "*.pcd")))
    
    if not pcd_files:
        print(f"'{directory_path}'에서 PCD 파일을 찾을 수 없습니다.")
        return
    
    # 시각화 윈도우 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for pcd_file in pcd_files:
        print(f"Processing: {pcd_file}")
        
        # 포인트 클라우드 로드
        pcd = o3d.io.read_point_cloud(pcd_file)
        
        # 복셀 다운샘플링
        voxel_size = 0.02
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        
        # 통계적 이상치 제거
        cleaned_pcd, _ = downsampled_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0)
        
        # DBSCAN 클러스터링
        points = np.asarray(cleaned_pcd.points)
        db = DBSCAN(eps=eps, min_samples=min_points).fit(points)
        labels = db.labels_
        
        # 클러스터 수 계산 (-1은 노이즈)
        max_label = labels.max()
        n_clusters = max_label + 1
        print(f"클러스터 수: {n_clusters}")
        
        # 클러스터별 색상 지정
        colors = np.zeros((len(points), 3))
        cluster_colors = np.random.rand(n_clusters + 1, 3)  # 노이즈용 색상 포함
        
        # 각 포인트에 클러스터 색상 할당
        for i in range(len(points)):
            if labels[i] == -1:  # 노이즈
                colors[i] = [0.5, 0.5, 0.5]  # 회색
            else:
                colors[i] = cluster_colors[labels[i]]
        
        # 시각화를 위한 포인트 클라우드 생성
        colored_pcd = o3d.geometry.PointCloud()
        colored_pcd.points = o3d.utility.Vector3dVector(points)
        colored_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 시각화 업데이트
        vis.clear_geometries()
        vis.add_geometry(colored_pcd)
        
        # 카메라 뷰포인트 설정
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        
        # 렌더링
        vis.poll_events()
        vis.update_renderer()
        
        # 프레임 간 대기
        time.sleep(0.1)
    
    # 시각화 종료
    vis.destroy_window()

if __name__ == "__main__":
    # 단일 프레임 처리
    #pcd_path = "data/01_straight_walk/000000.pcd"
    #process_single_frame(pcd_path)
    
    # 시퀀스 처리
    print("\n연속 프레임 처리 시작...")
    directory_path = "data/01_straight_walk/pcd"
    process_and_visualize_sequence(directory_path)
