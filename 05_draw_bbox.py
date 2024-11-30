import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
#from scipy.spatial import cKDTree

# 1. 밀도 계산 함수
def compute_density(points, k=5):
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    return 1 / distances.mean(axis=1)

# 2. PCA 분석 함수
def compute_pca(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    explained_variance_ratio = pca.explained_variance_ratio_
    return explained_variance_ratio

def detect_cluster_changes(previous_clusters, current_clusters, pca_threshold=0.3, density_threshold=10):
    changed_clusters = []
    previous_stats = []
    current_stats = []
    change_values = []  # 각 클러스터의 변화값 저장
    
    for cluster in previous_clusters:
        points = np.asarray(cluster.points)
        pca_ratios = compute_pca(points)
        density = compute_density(points).mean()
        previous_stats.append((pca_ratios, density))
    
    for cluster in current_clusters:
        points = np.asarray(cluster.points)
        pca_ratios = compute_pca(points)
        density = compute_density(points).mean()
        current_stats.append((pca_ratios, density))
    
    for i, (current_pca, current_density) in enumerate(current_stats):
        for j, (prev_pca, prev_density) in enumerate(previous_stats):
            pca_change = np.linalg.norm(np.array(current_pca) - np.array(prev_pca))
            density_change = abs(current_density - prev_density)
            total_change = pca_change + density_change
            if pca_change > pca_threshold and density_change > density_threshold:
                changed_clusters.append(i)
                change_values.append({
                    "cluster_index": i,
                    "pca_change": pca_change,
                    "density_change": density_change,
                    "total_change": total_change
                })
                break
    
    return changed_clusters, change_values

# Bounding Box 생성 함수 (변화된 클러스터만 처리)
def create_bounding_boxes_for_changed_clusters(clusters, changed_indices):
    bounding_boxes = []
    for idx in changed_indices:
        cluster = clusters[idx]
        bbox = cluster.get_axis_aligned_bounding_box()
        bbox.color = [1, 0, 0]  # 빨간색으로 표시
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
        "data/04_zigzag_walk/pcd/pcd_000267.pcd",
        "data/04_zigzag_walk/pcd/pcd_000268.pcd",
        "data/04_zigzag_walk/pcd/pcd_000269.pcd",
        "data/04_zigzag_walk/pcd/pcd_000270.pcd",
        "data/04_zigzag_walk/pcd/pcd_000271.pcd"
    ]
    previous_clusters = None
    
    for frame_idx, file_path in enumerate(file_paths):
        print(f"[DEBUG] Processing frame {frame_idx}: {file_path}")
        original_pcd = o3d.io.read_point_cloud(file_path)
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=0.05)
        cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
        filtered_pcd = downsample_pcd.select_by_index(ind)
        plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=2000)
        [a, b, c, d] = plane_model
        points = np.asarray(filtered_pcd.points)
        transformed_points = transform_to_plane_based_coordinates(points, a, b, c, d)
        
        x_min, y_min = np.min(transformed_points[:, :2], axis=0)
        x_max, y_max = np.max(transformed_points[:, :2], axis=0)
        x_bins = np.arange(x_min, x_max, 0.3)
        y_bins = np.arange(y_min, y_max, 0.3)
        height_map = calculate_height_map(transformed_points, x_bins, y_bins, threshold=0.05)
        
        non_floor_indices = []
        for idx, (x, y, z) in enumerate(transformed_points):
            x_idx = np.searchsorted(x_bins, x, side='right') - 1
            y_idx = np.searchsorted(y_bins, y, side='right') - 1
            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                smoothed_height = height_map.get((x_idx, y_idx), None)
                if smoothed_height is not None and abs(z - smoothed_height) > 0.15:
                    non_floor_indices.append(idx)
        
        non_floor_pcd = filtered_pcd.select_by_index(non_floor_indices)
        clusters, cluster_labels = cluster_non_floor_points(non_floor_pcd, eps=0.3, min_points=30)
        
        # 프레임 간 변화 탐지
        changed_indices = []
        change_values = []
        if previous_clusters is not None:
            changed_indices, change_values = detect_cluster_changes(previous_clusters, clusters, pca_threshold=0.3, density_threshold=10)
            print(f"[DEBUG] Changed clusters in frame {frame_idx}: {changed_indices}")
            for change in change_values:
                print(f"Cluster {change['cluster_index']}: PCA Change = {change['pca_change']:.4f}, "
                      f"Density Change = {change['density_change']:.4f}, "
                      f"Total Change = {change['total_change']:.4f}")
        
        # 변화가 감지된 클러스터에 대해서만 Bounding Box 생성
        bounding_boxes = create_bounding_boxes_for_changed_clusters(clusters, changed_indices)
        
        # 시각화
        visualize_with_bounding_boxes(clusters, bounding_boxes, window_name=f"Frame {frame_idx} with Bounding Boxes", point_size=1.0)
        
        # 다음 프레임 준비
        previous_clusters = clusters
        