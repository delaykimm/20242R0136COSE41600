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

# 5. 시각화 함수
def visualize_point_clouds(pcd_list, window_name="ROR Visualization", point_size=1.0):
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

# 메인 실행 코드
#file_path = "data/05_straight_duck_walk/pcd/pcd_000370.pcd"
file_path = "data/01_straight_walk/pcd/pcd_000100.pcd"
original_pcd = o3d.io.read_point_cloud(file_path)
print(f"[DEBUG] Original point cloud size: {len(original_pcd.points)}")
print("[INFO] Visualizing original point cloud...")
visualize_point_clouds(original_pcd, window_name="Original Point Cloud", point_size=2.0)

# Voxel Downsampling
voxel_size = 0.01
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
print(f"[DEBUG] Downsampled point cloud size: {len(downsample_pcd.points)}")
print("[INFO] Visualizing downsampled point cloud...")
visualize_point_clouds(downsample_pcd, window_name="Downsampled Point Cloud", point_size=2.0)

# Radius Outlier Removal (ROR)
cl, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
# ROR 후 남은 점 (inliers)
ror_inliers_pcd = downsample_pcd.select_by_index(ind)
ror_inliers_pcd.paint_uniform_color([0, 1, 0])  # 녹색 (남은 점)

# ROR로 제거된 점 (outliers)
ror_outliers_pcd = downsample_pcd.select_by_index(ind, invert=True)
ror_outliers_pcd.paint_uniform_color([1, 0, 0])  # 빨간색 (제거된 점)

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
# projected_points = project_to_new_plane(ror_points, a, b, c, d)
transformed_points = transform_to_plane_based_coordinates(ror_points, a, b, c, d)

# 격자 생성
# grid_resolution = adjust_grid_size(projected_points, target_bin_count=50)
grid_resolution = adjust_grid_size(transformed_points, target_bin_count=50)
grid_resolution = 0.3

# x_min, y_min = np.min(projected_points[:, :2], axis=0)
# x_max, y_max = np.max(projected_points[:, :2], axis=0)
x_min, y_min = np.min(transformed_points[:, :2], axis=0)
x_max, y_max = np.max(transformed_points[:, :2], axis=0)

x_bins = np.arange(x_min, x_max, grid_resolution)
y_bins = np.arange(y_min, y_max, grid_resolution)
print(f"[DEBUG] Number of x_bins: {len(x_bins)}, Number of y_bins: {len(y_bins)}")

# 높이 맵 계산 및 보완
# height_map = calculate_height_map(projected_points, x_bins, y_bins)
height_map = calculate_height_map(transformed_points, x_bins, y_bins)
height_map = fill_empty_grids(height_map, x_bins, y_bins)

# 비바닥 점 필터링
threshold = 0.2
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

# print(non_floor_indices)

# Floor points count
# floor_indices = set(range(len(projected_points))) - set(non_floor_indices)
floor_indices = set(range(len(transformed_points))) - set(non_floor_indices)
print(f"[DEBUG] Total floor points: {len(floor_indices)}")

# print(floor_indices)

# Open3D 객체 변환
projected_pcd = o3d.geometry.PointCloud()
# projected_pcd.points = o3d.utility.Vector3dVector(projected_points)
projected_pcd.points = o3d.utility.Vector3dVector(transformed_points)

# 비바닥 및 바닥 포인트
non_floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices)
floor_pcd = ror_inliers_pcd.select_by_index(non_floor_indices, invert=True)
road_pcd = ror_inliers_pcd.select_by_index(inliers)
non_road_pcd = ror_inliers_pcd.select_by_index(inliers, invert=True)

# 비바닥 점 (non-floor) 개수
num_non_floor_points = len(non_floor_pcd.points)
print(f"Number of non-floor points: {num_non_floor_points}")

# 바닥 점 (floor) 개수
num_floor_points = len(floor_pcd.points)
print(f"Number of floor points: {num_floor_points}")

# 도로 점 (road) 개수
num_road_points = len(road_pcd.points)
print(f"Number of road points: {num_road_points}")

# 도로 아닌 점 (non-road) 개수
num_non_road_points = len(non_road_pcd.points)
print(f"Number of non-road points: {num_non_road_points}")

# 색상 설정
floor_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
non_floor_pcd.paint_uniform_color([0, 1, 0])  # 녹색

road_pcd.paint_uniform_color([1, 0, 0])  # 빨간색
non_road_pcd.paint_uniform_color([0, 1, 0])  # 녹색

# 시각화
# visualize_point_clouds([road_pcd, non_road_pcd], 
#                        window_name="Road (Red) and Non-Road (Green) Points", point_size=2.0)

visualize_point_clouds([floor_pcd, non_floor_pcd], 
                       window_name="Floor (Red) and Non-Floor (Green) Points", point_size=2.0)