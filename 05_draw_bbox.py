import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
#from scipy.spatial import cKDTree

# 1. 밀도 계산 함수
def compute_density(points, k=5):
    if len(points) < k:
        return np.array([0] * len(points))  # 점이 부족할 경우 기본값 반환
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    return 1 / distances.mean(axis=1)

# 2. PCA 분석 함수
def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    explained_variance_ratio = pca.explained_variance_ratio_
    return explained_variance_ratio

# 격자 내 포인트 PCA와 밀도 계산 함수
def calculate_grid_pca_and_density(points, x_bins, y_bins):
    x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
    y_indices = np.digitize(points[:, 1], bins=y_bins) - 1

    grid_pca = {}
    grid_density = {}

    for i, (x_idx, y_idx) in enumerate(zip(x_indices, y_indices)):
        if x_idx < 0 or y_idx < 0 or x_idx >= len(x_bins) - 1 or y_idx >= len(y_bins) - 1:
            continue

        key = (x_idx, y_idx)
        if key not in grid_pca:
            grid_pca[key] = []
            grid_density[key] = []

        grid_pca[key].append(points[i])
        grid_density[key].append(points[i])

    # PCA와 밀도 계산
    for key in list(grid_pca.keys()):
        grid_points = np.array(grid_pca[key])

        # PCA 계산
        if len(grid_points) >= 3:  # 최소 3개의 점이 필요
            grid_pca[key] = compute_pca(grid_points)
        else:
            grid_pca[key] = [0, 0, 0]  # 기본값 설정

        # 밀도 계산
        density_points = np.array(grid_density[key])
        grid_density[key] = compute_density(density_points, k=5).mean() if len(density_points) >= 5 else 0

    return grid_pca, grid_density

# 격자의 중심 좌표를 계산하는 함수
def get_grid_center(x_idx, y_idx, x_bins, y_bins):
    x_center = (x_bins[x_idx] + x_bins[x_idx + 1]) / 2
    y_center = (y_bins[y_idx] + y_bins[y_idx + 1]) / 2
    return np.array([x_center, y_center])

# 격자 간 변화 탐지 함수 (2m 거리 제한 추가)
def detect_grid_changes(prev_grid_pca, prev_grid_density, curr_grid_pca, curr_grid_density, 
                        x_bins, y_bins, pca_threshold=0.3, density_threshold=10, distance_threshold=2.0):
    changed_grids = []

    for key in curr_grid_pca.keys():
        if key in prev_grid_pca:
            # PCA 및 밀도 변화 계산
            pca_change = np.linalg.norm(np.array(curr_grid_pca[key]) - np.array(prev_grid_pca[key]))
            density_change = abs(curr_grid_density[key] - prev_grid_density[key])
            
            # 중심 좌표 간 거리 계산
            curr_center = get_grid_center(key[0], key[1], x_bins, y_bins)
            prev_center = get_grid_center(key[0], key[1], x_bins, y_bins)  # 이전 격자와 동일 키
            
            # 조건 충족 여부 확인
            if pca_change > pca_threshold or density_change > density_threshold:
                if np.linalg.norm(curr_center - prev_center) <= distance_threshold:
                    changed_grids.append(key)
        else:
            # 새로운 격자로 간주 (거리에 대한 조건 제외)
            changed_grids.append(key)

    return changed_grids

# Bounding Box 생성 함수 (포인트 개수 조건 및 2m 크기 제한 추가)
def create_bounding_boxes_for_changed_grids(points, labels, changed_grids, x_bins, y_bins):
    bounding_boxes = []

    for grid_key in changed_grids:
        x_min = x_bins[grid_key[0]]
        x_max = x_bins[grid_key[0] + 1]
        y_min = y_bins[grid_key[1]]
        y_max = y_bins[grid_key[1] + 1]

        grid_indices = [
            i for i, (x, y) in enumerate(points[:, :2]) if x_min <= x < x_max and y_min <= y < y_max
        ]

        cluster_indices = set(labels[grid_indices])
        for cluster_id in cluster_indices:
            if cluster_id == -1:
                continue  # 노이즈 제외
            
            # 클러스터에 속한 포인트 추출
            cluster_points = points[labels == cluster_id]

            # 클러스터의 포인트 개수 조건 확인
            if len(cluster_points) < 100:
                continue  # 포인트 개수가 70개 미만이면 건너뜀

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            
            # 바운딩 박스 크기 계산
            box_size = bbox.get_extent()  # [가로(x), 세로(y), 높이(z)]
            if all(dim <= 2.0 for dim in box_size):  # 모든 크기가 2m 이내인지 확인
                bbox.color = [1, 0, 0]  # 빨간색
                bounding_boxes.append(bbox)

    return bounding_boxes

# 4. 평면으로 투영하는 함수
def transform_to_plane_based_coordinates(points, a, b, c, d):
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm**2
    projected_points = points - distances[:, np.newaxis] * normal
    plane_equation_values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    new_z = plane_equation_values / normal_norm
    new_x = projected_points[:, 0]
    new_y = projected_points[:, 1]
    transformed_points = np.stack((new_x, new_y, new_z), axis=-1)
    return transformed_points

# 5. 높이 맵 계산 함수
def calculate_height_map(points, x_bins, y_bins, threshold=0.2):
    x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
    y_indices = np.digitize(points[:, 1], bins=y_bins) - 1
    valid_mask = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                 (y_indices >= 0) & (y_indices < len(y_bins) - 1)
    valid_points = points[valid_mask]
    x_indices, y_indices = x_indices[valid_mask], y_indices[valid_mask]
    height_map = {}
    for (z, i, j) in zip(valid_points[:, 2], x_indices, y_indices):
        key = (i, j)
        if key in height_map:
            height_map[key].append(z)
        else:
            height_map[key] = [z]
    for key, z_values in height_map.items():
        sorted_z_values = np.sort(z_values)
        index_5th_percentile = max(0, int(len(sorted_z_values) * 0.05))
        height_map[key] = sorted_z_values[index_5th_percentile]
    return height_map

# 6. 비바닥 클러스터링 함수
def cluster_non_floor_points(non_floor_pcd, eps=0.5, min_points=10):
    labels = np.array(non_floor_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    clusters = []
    for cluster_id in range(max_label + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_pcd = non_floor_pcd.select_by_index(cluster_indices)
        clusters.append(cluster_pcd)
    return clusters, labels

# Bounding Box와 Point Cloud를 함께 시각화
def visualize_with_bounding_boxes(pcd_list, bounding_boxes, window_name="Clusters with Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 클러스터 추가
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    
    # Bounding Box 추가
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    
    # 시각화 옵션 설정
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 7. 시각화 함수
def visualize_point_clouds(pcd_list, window_name="Clusters Visualization", point_size=0.5):
    if isinstance(pcd_list, o3d.geometry.PointCloud):
        pcd_list = [pcd_list]
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 메인 실행 코드
if __name__ == "__main__":
    file_paths = [
        "data/03_straight_crawl/pcd/pcd_000267.pcd",
        "data/03_straight_crawl/pcd/pcd_000268.pcd",
        "data/03_straight_crawl/pcd/pcd_000269.pcd",
        "data/03_straight_crawl/pcd/pcd_000270.pcd",
        "data/03_straight_crawl/pcd/pcd_000271.pcd"
    ]
    previous_grid_pca = None
    previous_grid_density = None
    
    for frame_idx, file_path in enumerate(file_paths):
        print(f"[DEBUG] Processing frame {frame_idx}: {file_path}")
        
        # PCD 파일 읽기
        original_pcd = o3d.io.read_point_cloud(file_path)
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=0.05)
        
        # Outlier 제거
        cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
        filtered_pcd = downsample_pcd.select_by_index(ind)
        
        # 평면 분할
        plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=2000)
        [a, b, c, d] = plane_model
        
        # Points 배열로 변환
        points = np.asarray(filtered_pcd.points)
        
        # 평면 좌표계로 변환
        transformed_points = transform_to_plane_based_coordinates(points, a, b, c, d)
        
        # X, Y 격자 설정
        x_min, y_min = np.min(transformed_points[:, :2], axis=0)
        x_max, y_max = np.max(transformed_points[:, :2], axis=0)
        x_bins = np.arange(x_min, x_max, 0.5)
        y_bins = np.arange(y_min, y_max, 0.5)
        
        # 높이 맵 계산
        height_map = calculate_height_map(transformed_points, x_bins, y_bins, threshold=0.05)
        
        # 비바닥 포인트 추출
        non_floor_indices = []
        for idx, (x, y, z) in enumerate(transformed_points):
            x_idx = np.searchsorted(x_bins, x, side='right') - 1
            y_idx = np.searchsorted(y_bins, y, side='right') - 1
            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                smoothed_height = height_map.get((x_idx, y_idx), None)
                if smoothed_height is not None and abs(z - smoothed_height) > 0.15:
                    non_floor_indices.append(idx)
        
        non_floor_pcd = filtered_pcd.select_by_index(non_floor_indices)
        
        # 클러스터링
        labels = np.array(non_floor_pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))
        
        # 격자 내 PCA와 밀도 계산
        current_grid_pca, current_grid_density = calculate_grid_pca_and_density(
            np.asarray(non_floor_pcd.points), x_bins, y_bins
        )
        
        # 변화 감지 (2m 거리 필터 포함)
        changed_grids = []
        if previous_grid_pca is not None:
            changed_grids = detect_grid_changes(
                previous_grid_pca, previous_grid_density,
                current_grid_pca, current_grid_density,
                x_bins, y_bins,
                pca_threshold=0.3, density_threshold=10, distance_threshold=2.0
            )
            print(f"[DEBUG] Changed grids in frame {frame_idx}: {changed_grids}")

        # Bounding Box 생성
        bounding_boxes = create_bounding_boxes_for_changed_grids(
            np.asarray(non_floor_pcd.points), labels, changed_grids, x_bins, y_bins
        )
        
        # 시각화
        visualize_with_bounding_boxes([non_floor_pcd], bounding_boxes, window_name=f"Frame {frame_idx} with Bounding Boxes")
        
        # 다음 프레임 준비
        previous_grid_pca = current_grid_pca
        previous_grid_density = current_grid_density