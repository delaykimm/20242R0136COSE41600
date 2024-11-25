import open3d as o3d
import numpy as np
import time
import glob
import os
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

class PedestrianDetector:
    def __init__(self):
        # 탐지 파라미터
        self.voxel_size = 0.01
        self.cluster_eps = 0.5
        self.min_points = 30
        self.human_height_range = (1.4, 2.0)
        self.human_width_range = (0.3, 1.0)

    def preprocess_pointcloud(self, pcd):
        # 1. 평면으로 투영하는 함수
        def transform_to_plane_based_coordinates(points, a, b, c, d):
            """
            평면을 새로운 XY 평면으로 간주하고,
            모든 포인트를 새로운 좌표계로 변환합니다.
            새로운 X, Y는 평면 위에서의 투영된 좌표이며,
            Z는 평면까지의 수직 거리입니다.

            Args:
                points (numpy.ndarray): 입력 포인트 클라우드, 크기 (N, 3).
                a, b, c, d (float): 평면 방정식의 계수 (ax + by + cz + d = 0).

            Returns:
                numpy.ndarray: 새로운 좌표계의 포인트 클라우드, 크기 (N, 3).
            """
            print("[DEBUG] Transforming points to plane-based coordinate system...")
            print(f"[DEBUG] Input points shape: {points.shape}")

            # 평면의 법선 벡터 및 크기 계산
            normal = np.array([a, b, c])
            normal_norm = np.linalg.norm(normal)

            # 점들을 평면에 투영
            distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm**2
            projected_points = points - distances[:, np.newaxis] * normal  # 투영된 좌표 계산

            # 새로운 Z 값: 평면까지의 수직 거리 (부호 유지)
            plane_equation_values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
            new_z = plane_equation_values / normal_norm

            # 새로운 좌표계
            new_x = projected_points[:, 0]
            new_y = projected_points[:, 1]
            transformed_points = np.stack((new_x, new_y, new_z), axis=-1)

            print(f"[DEBUG] Transformed points shape: {transformed_points.shape}")
            return transformed_points

        # 3. 높이 맵 계산 함수
        def calculate_height_map(points, x_bins, y_bins, threshold=0.2):
            print("[DEBUG] Calculating height map...")

            # Step 1: Digitize indices for x and y
            x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
            y_indices = np.digitize(points[:, 1], bins=y_bins) - 1


            # Step 2: Create a mask for valid indices
            valid_mask = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                        (y_indices >= 0) & (y_indices < len(y_bins) - 1)
            valid_points = points[valid_mask]
            x_indices, y_indices = x_indices[valid_mask], y_indices[valid_mask]

            print(f"[DEBUG] Valid points count: {len(valid_points)}")

            # Step 3: Calculate initial height map
            height_map = {}
            for (z, i, j) in zip(valid_points[:, 2], x_indices, y_indices):
                key = (i, j)
                if key in height_map:
                    height_map[key].append(z)
                else:
                    height_map[key] = [z]

            for key, z_values in height_map.items():
                if z_values:
                    sorted_z_values = np.sort(z_values)
                    index_5th_percentile = max(0, int(len(sorted_z_values) * 0.05))
                    height_map[key] = sorted_z_values[index_5th_percentile]
                else:
                    height_map[key] = None

            print(f"[DEBUG] Initial height map size: {len(height_map)}")

            # Step 4: Post-processing based on neighbor differences
            # Step 4.1: Build KDTree only for non-empty grid points
            # Build grid keys and values
            grid_keys = [key for key in height_map.keys() if height_map[key] is not None]  # 리스트로 유지
            grid_values = np.array([height_map[key] for key in grid_keys])  # 값은 numpy 배열로 생성

            # Precompute grid centers
            grid_centers = np.array([((x_bins[i] + x_bins[i + 1]) / 2, (y_bins[j] + y_bins[j + 1]) / 2) for i, j in grid_keys])
            kdtree = cKDTree(grid_centers)

            processed_height_map = {}
            search_radius = 1  # Define search radius for neighbor checks

            for (i, j), height in height_map.items():
                if height is None:
                    # Skip empty grids
                    continue

                # Get the center of the current grid
                center_x = (x_bins[i] + x_bins[i + 1]) / 2
                center_y = (y_bins[j] + y_bins[j + 1]) / 2

                # Find neighbors within the search radius
                indices = kdtree.query_ball_point([center_x, center_y], r=search_radius)
                neighbor_heights = [grid_values[idx] for idx in indices if grid_values[idx] is not None]

                # Process the height difference
                if neighbor_heights:
                    smallest_neighbor = min(neighbor_heights)
                    if abs(height - smallest_neighbor) > threshold:
                        processed_height_map[(i, j)] = smallest_neighbor
                    else:
                        processed_height_map[(i, j)] = height
                else:
                    processed_height_map[(i, j)] = height  # Leave unchanged if no neighbors found

            print(f"[DEBUG] Processed height map size: {len(processed_height_map)}")
            return processed_height_map

        def cluster_non_floor_points(non_floor_pcd, eps=0.5, min_points=10):
            """
            DBSCAN 클러스터링을 사용하여 비바닥(non-floor) 점을 군집화합니다.
            
            Args:
                non_floor_pcd (open3d.geometry.PointCloud): 비바닥 포인트 클라우드.
                eps (float): DBSCAN의 클러스터 반경.
                min_points (int): 클러스터를 형성하기 위한 최소 포인트 수.
                
            Returns:
                list of open3d.geometry.PointCloud: 각 클러스터에 해당하는 포인트 클라우드 리스트.
            """
            print("[DEBUG] Performing DBSCAN clustering...")
            labels = np.array(non_floor_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
            max_label = labels.max()
            print(f"[DEBUG] Number of clusters: {max_label + 1}")
            
            clusters = []
            for cluster_id in range(max_label + 1):
                cluster_indices = np.where(labels == cluster_id)[0]
                cluster_pcd = non_floor_pcd.select_by_index(cluster_indices)
                clusters.append(cluster_pcd)
            
            return clusters, labels

        # 4. 시각화 함수
        def visualize_point_clouds(pcd_list, window_name="ROR Visualization", point_size=0.5):
            # 단일 객체를 리스트로 변환
            if isinstance(pcd_list, o3d.geometry.PointCloud):
                pcd_list = [pcd_list]
                
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name)
            for pcd in pcd_list:
                vis.add_geometry(pcd)
            vis.get_render_option().point_size = point_size
            vis.run()
            vis.destroy_window()

        # # 메인 실행 코드
        # # file_path = "data/05_straight_duck_walk/pcd/pcd_000370.pcd"
        # # file_path = "data/01_straight_walk/pcd/pcd_000250.pcd"
        # file_path = "data/04_zigzag_walk/pcd/pcd_000250.pcd"
        # # file_path = "data/06_straight_crawl/pcd/pcd_000500.pcd"
        # # file_path = "data/02_straight_duck_walk/pcd/pcd_000500.pcd"
        # # file_path = "data/03_straight_crawl/pcd/pcd_000900.pcd"
        # # file_path = "data/07_straight_walk/pcd/pcd_000350.pcd"
        # original_pcd = o3d.io.read_point_cloud(file_path)
        print(f"[DEBUG] Original point cloud size: {len(pcd.points)}")

        # Voxel Downsampling
        voxel_size = 0.05
        downsample_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"[DEBUG] Downsampled point cloud size: {len(downsample_pcd.points)}")

        # Radius Outlier Removal (ROR)
        cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
        # ROR 후 남은 점 (inliers)
        ror_inliers_pcd = downsample_pcd.select_by_index(ind)
        ror_inliers_pcd.paint_uniform_color([0, 1, 0])  # 녹색 (남은 점)

        # ROR로 제거된 점 (outliers)
        ror_outliers_pcd = downsample_pcd.select_by_index(ind, invert=True)
        ror_outliers_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (제거된 점)

        ror_pcd = downsample_pcd.select_by_index(ind)
        print(f"[DEBUG] Point cloud size after ROR: {len(ror_pcd.points)}")

        # 평면 추정
        plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=2000)
        [a, b, c, d] = plane_model
        print(f"[DEBUG] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

        ror_points = np.asarray(ror_pcd.points)
        transformed_points = transform_to_plane_based_coordinates(ror_points, a, b, c, d)

        # 격자 생성
        grid_resolution = 0.3
        x_min, y_min = np.min(transformed_points[:, :2], axis=0)
        x_max, y_max = np.max(transformed_points[:, :2], axis=0)

        x_bins = np.arange(x_min, x_max, grid_resolution)
        y_bins = np.arange(y_min, y_max, grid_resolution)
        print(f"[DEBUG] Number of x_bins: {len(x_bins)}, Number of y_bins: {len(y_bins)}")

        # 높이 맵 계산 및 보완
        height_map = calculate_height_map(transformed_points, x_bins, y_bins, threshold = 0.05) # threshold : 주변 격자와의 높이차 

        # 비바닥 점 필터링
        threshold = 0.15 # threshold : 격자의 바닥 높이와의 차
        non_floor_indices = []
        # for idx, (x, y, z) in enumerate(projected_points):
        for idx, (x, y, z) in enumerate(transformed_points):
            x_idx = np.searchsorted(x_bins, x, side='right') - 1
            y_idx = np.searchsorted(y_bins, y, side='right') - 1
            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                smoothed_height = height_map.get((x_idx, y_idx), None)
                if smoothed_height is not None and abs(z - smoothed_height) > threshold:
                    non_floor_indices.append(idx)
        print(f"[DEBUG] Total non-floor points: {len(non_floor_indices)}")

        # Floor points count
        # floor_indices = set(range(len(projected_points))) - set(non_floor_indices)
        floor_indices = set(range(len(transformed_points))) - set(non_floor_indices)
        print(f"[DEBUG] Total floor points: {len(floor_indices)}")

        # 비바닥 및 바닥 포인트
        non_floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices)
        floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices, invert=True)

        # 비바닥 점 (non-floor) 개수
        num_non_floor_points = len(non_floor_pcd.points)
        print(f"Number of non-floor points: {num_non_floor_points}")

        # 바닥 점 (floor) 개수
        num_floor_points = len(floor_pcd.points)
        print(f"Number of floor points: {num_floor_points}")

        # 색상 설정
        floor_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
        non_floor_pcd.paint_uniform_color([0, 1, 0])  # 녹색

        # visualize_point_clouds([floor_pcd, non_floor_pcd], 
        #                        window_name="Floor (Red) and Non-Floor (Green) Points", point_size=1.0)

        # 클러스터링 실행
        eps = 0.3  # 클러스터 반경
        min_points = 30  # 클러스터 최소 크기
        clusters, cluster_labels = cluster_non_floor_points(non_floor_pcd, eps=eps, min_points=min_points)

        # 클러스터링 결과 시각화 (각 클러스터 다른 색상 적용)
        for i, cluster in enumerate(clusters):
            color = np.random.rand(3)  # 무작위 색상
            cluster.paint_uniform_color(color)
        return clusters

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
    view_control.set_zoom(0.5)  # 기존보다 더 가깝게
    
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
    pcd_files = pcd_files[50:100]
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
            clusters = detector.preprocess_pointcloud(pcd)
            
            # 클러스터링 결과로 새로운 포인트 클라우드 생성
            vis_pcd = o3d.geometry.PointCloud()
            for cluster in clusters:
                color = np.random.rand(3)  # 무작위 색상
                cluster.paint_uniform_color(color)
                vis_pcd += cluster  # 클러스터를 통합
            
            # # 보행자 클러스터 추가 (빨간색)
            # for cluster in pedestrians:
            #     cluster_pcd = o3d.geometry.PointCloud()
            #     cluster_pcd.points = o3d.utility.Vector3dVector(cluster)
            #     cluster_pcd.paint_uniform_color([1, 0, 0])  # 보행자는 빨간색
            #     vis_pcd.points.extend(cluster_pcd.points)
            #     vis_pcd.colors.extend(cluster_pcd.colors)
            
            # 시각화 업데이트
            vis.clear_geometries()
            vis.add_geometry(vis_pcd)
            
            # 렌더링 옵션 설정
            render_option = vis.get_render_option()
            render_option.point_size = point_size
            #render_option.background_color = np.array([0, 0, 0])  # 검은 배경
            
            # 첫 프레임에서만 카메라 뷰 초기화
            view_control = vis.get_view_control()
            # if saved_camera_params is None:
            #     saved_camera_params = set_camera_view(vis, initial_center, initial_size)
            # else:
            #     # 이후 프레임에서는 저장된 카메라 매개변수 복원
            #     view_control.convert_from_pinhole_camera_parameters(saved_camera_params)
            
            # 렌더링
            vis.poll_events()
            vis.update_renderer()
            
            # 프레임 간 대기
            #time.sleep(0.2)
            
        except Exception as e:
            print(f"프레임 처리 중 오류 발생: {e}")
            continue
    
    # 시각화 종료
    vis.destroy_window()

def main():
    try:
        print("\n연속 프레임 시각화 시작...")
        sequence_directory = "data/04_zigzag_walk/pcd"
        visualize_sequence(sequence_directory, point_size=1.0)
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        print("프로그램 종료")

if __name__ == "__main__":
    main()