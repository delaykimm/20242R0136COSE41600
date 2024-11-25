# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

# 1. 평면으로 투영하는 함수
def project_to_new_plane(points, a, b, c, d):
    """
    평면 방정식을 새로운 XY 평면으로 간주하고,
    모든 포인트를 해당 평면으로 투영하여 새로운 (x, y, z) 좌표를 계산합니다.

    Args:
        points (numpy.ndarray): 입력 포인트 클라우드, 크기 (N, 3).
        a, b, c, d (float): 평면 방정식의 계수 (ax + by + cz + d = 0).

    Returns:
        numpy.ndarray: 변환된 포인트 클라우드, 크기 (N, 3).
    """
    print("[DEBUG] Projecting points onto the new plane...")
    print(f"[DEBUG] Input points shape: {points.shape}")
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm**2
    projection = points - distances[:, np.newaxis] * normal  # 투영된 좌표 계산
    print(f"[DEBUG] Projected points shape: {projection.shape}")
    return projection

# 2. 격자 크기 조정 함수
def adjust_grid_size(points, target_bin_count=50):
    print("[DEBUG] Adjusting grid size...")
    x_range = np.ptp(points[:, 0])
    y_range = np.ptp(points[:, 1])
    grid_size = min(x_range, y_range) / target_bin_count
    print(f"[DEBUG] Grid size: {grid_size}")
    return grid_size

# 3. 높이 맵 계산 함수
def calculate_height_map(points, x_bins, y_bins):
    print("[DEBUG] Calculating height map...")
    x_indices = np.digitize(points[:, 0], bins=x_bins) - 1
    y_indices = np.digitize(points[:, 1], bins=y_bins) - 1

    valid_mask = (x_indices >= 0) & (x_indices < len(x_bins) - 1) & \
                 (y_indices >= 0) & (y_indices < len(y_bins) - 1)
    x_indices, y_indices, valid_points = x_indices[valid_mask], y_indices[valid_mask], points[valid_mask]

    print(f"[DEBUG] Valid points count: {len(valid_points)}")
    
    height_map = {}
    for (x, y, z), i, j in zip(valid_points, x_indices, y_indices):
        key = (i, j)
        if key not in height_map:
            height_map[key] = []
        height_map[key].append(z)  # z 값만 추가

    for key in height_map:
        if len(height_map[key]) > 0:
            height_map[key] = np.median(height_map[key])  # 중앙값 계산
        else:
            height_map[key] = 0  # 기본값 설정 (빈 리스트일 경우)

    print(f"[DEBUG] Height map size: {len(height_map)}")
    return height_map

# 4. 빈 격자 보완 함수
def fill_empty_grids(height_map, x_bins, y_bins):
    print("[DEBUG] Filling empty grids...")
    grid_keys = np.array(list(height_map.keys()))
    grid_values = np.array(list(height_map.values()))
    print(f"[DEBUG] Number of height map entries before filling: {len(grid_keys)}")
    grid_centers = [((x_bins[i] + x_bins[i + 1]) / 2, (y_bins[j] + y_bins[j + 1]) / 2) for i, j in grid_keys]

    kdtree = cKDTree(grid_centers)
    filled_map = {}

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            key = (i, j)
            if key in height_map:
                filled_map[key] = height_map[key]
            else:
                _, idx = kdtree.query([(x_bins[i] + x_bins[i + 1]) / 2, (y_bins[j] + y_bins[j + 1]) / 2])
                filled_map[key] = grid_values[idx]

    print(f"[DEBUG] Number of height map entries after filling: {len(filled_map)}")
    return filled_map

# 메인 실행 코드
file_path = "data/05_straight_duck_walk/pcd/pcd_000370.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)
print(f"[DEBUG] Original point cloud size: {len(original_pcd.points)}")

# Voxel Downsampling
voxel_size = 0.01
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"[DEBUG] Downsampled point cloud size: {len(downsample_pcd.points)}")

# Radius Outlier Removal (ROR)
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
# ROR 후 남은 점 (inliers)
ror_inliers_pcd = downsample_pcd.select_by_index(ind)
ror_inliers_pcd.paint_uniform_color([0, 1, 0])  # 녹색 (남은 점)

# ROR로 제거된 점 (outliers)
ror_outliers_pcd = downsample_pcd.select_by_index(ind, invert=True)
ror_outliers_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (제거된 점)

# 시각화 함수
def visualize_point_clouds(pcd_list, window_name="ROR Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# ROR 시각화
visualize_point_clouds([ror_inliers_pcd, ror_outliers_pcd], 
                       window_name="ROR Visualization: Inliers (Green) & Outliers (Red)", point_size=2.0)

ror_pcd = downsample_pcd.select_by_index(ind)
print(f"[DEBUG] Point cloud size after ROR: {len(ror_pcd.points)}")

# 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.15, ransac_n=3, num_iterations=2000)
[a, b, c, d] = plane_model
print(f"[DEBUG] Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

ror_points = np.asarray(ror_pcd.points)
projected_points = project_to_new_plane(ror_points, a, b, c, d)

# 격자 생성
grid_resolution = adjust_grid_size(projected_points, target_bin_count=50)
x_min, y_min = np.min(projected_points[:, :2], axis=0)
x_max, y_max = np.max(projected_points[:, :2], axis=0)
x_bins = np.arange(x_min, x_max, grid_resolution)
y_bins = np.arange(y_min, y_max, grid_resolution)
print(f"[DEBUG] Number of x_bins: {len(x_bins)}, Number of y_bins: {len(y_bins)}")

# 높이 맵 계산 및 보완
height_map = calculate_height_map(projected_points, x_bins, y_bins)
height_map = fill_empty_grids(height_map, x_bins, y_bins)

# 격자들의 Z 값 출력
# print("[DEBUG] Height map values:")
# for key, z in height_map.items():
#     print(f"Grid {key}: Z = {z}")

# 비바닥 점 필터링
threshold = 0.2
non_floor_indices = []
for idx, (x, y, z) in enumerate(projected_points):
    x_idx = np.searchsorted(x_bins, x, side='right') - 1
    y_idx = np.searchsorted(y_bins, y, side='right') - 1
    if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
        smoothed_height = height_map.get((x_idx, y_idx), None)
        if smoothed_height is not None and abs(z - smoothed_height) > threshold:
            non_floor_indices.append(idx)
print(f"[DEBUG] Total non-floor points: {len(non_floor_indices)}")

# Floor points count
floor_indices = set(range(len(projected_points))) - set(non_floor_indices)
print(f"[DEBUG] Total floor points: {len(floor_indices)}")

# Open3D 객체 변환
projected_pcd = o3d.geometry.PointCloud()
projected_pcd.points = o3d.utility.Vector3dVector(projected_points)

# 비바닥 및 바닥 포인트
non_floor_pcd = projected_pcd.select_by_index(non_floor_indices)
road_pcd = projected_pcd.select_by_index(inliers)
non_road_pcd = projected_pcd.select_by_index(inliers, invert=True)

# 색상 설정
road_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
non_road_pcd.paint_uniform_color([0, 1, 0])  # 녹색

# 시각화
visualize_point_clouds([road_pcd, non_road_pcd], 
                       window_name="Road (Red) and Non-Road (Green) Points", point_size=2.0)