import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

def transform_to_plane_based_coordinates(points, a, b, c, d):
    """
    평면을 새로운 XY 평면으로 간주하고,
    모든 포인트를 새로운 좌표계로 변환합니다.
    """
    print("[DEBUG] Transforming points to plane-based coordinate system...")
    print(f"[DEBUG] Input points shape: {points.shape}")

    # 평면의 법선 벡터 및 크기 계산
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)

    # 점들을 평면에 투영
    distances = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / normal_norm**2
    projected_points = points - distances[:, np.newaxis] * normal

    # 새로운 Z 값: 평면까지의 수직 거리 (부호 유지)
    plane_equation_values = a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d
    new_z = plane_equation_values / normal_norm

    # 새로운 좌표계
    new_x = projected_points[:, 0]
    new_y = projected_points[:, 1]
    transformed_points = np.stack((new_x, new_y, new_z), axis=-1)

    print(f"[DEBUG] Transformed points shape: {transformed_points.shape}")
    return transformed_points

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
    grid_keys = [key for key in height_map.keys() if height_map[key] is not None]
    grid_values = np.array([height_map[key] for key in grid_keys])

    # Precompute grid centers
    grid_centers = np.array([((x_bins[i] + x_bins[i + 1]) / 2, (y_bins[j] + y_bins[j + 1]) / 2) for i, j in grid_keys])
    kdtree = cKDTree(grid_centers)

    processed_height_map = {}
    search_radius = 1

    for (i, j), height in height_map.items():
        if height is None:
            continue

        center_x = (x_bins[i] + x_bins[i + 1]) / 2
        center_y = (y_bins[j] + y_bins[j + 1]) / 2

        indices = kdtree.query_ball_point([center_x, center_y], r=search_radius)
        neighbor_heights = [grid_values[idx] for idx in indices if grid_values[idx] is not None]

        if neighbor_heights:
            smallest_neighbor = min(neighbor_heights)
            if abs(height - smallest_neighbor) > threshold:
                processed_height_map[(i, j)] = smallest_neighbor
            else:
                processed_height_map[(i, j)] = height
        else:
            processed_height_map[(i, j)] = height

    print(f"[DEBUG] Processed height map size: {len(processed_height_map)}")
    return processed_height_map

def cluster_non_floor_points(non_floor_pcd, eps=0.5, min_points=10):
    """
    DBSCAN 클러스터링을 사용하여 비바닥(non-floor) 점을 군집화.
    """
    print("[DEBUG] Performing DBSCAN clustering...")
    print(f"입력 포인트 수: {len(non_floor_pcd.points)}")
    print(f"파라미터: eps={eps}, min_points={min_points}")
    
    # DBSCAN 수행
    labels = np.array(non_floor_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # 레이블 분석
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"고유 레이블: {unique_labels}")
    print(f"레이블별 포인트 수: {counts}")
    
    if len(labels) == 0:
        print("경고: 클러스터링 결과가 비어있습니다!")
        return [], labels
    
    if np.all(labels == -1):
        print("경고: 모든 포인트가 노이즈로 분류되었습니다!")
        print("eps 값을 늘리거나 min_points 값을 줄여보세요.")
        return [], labels
    
    max_label = labels.max()
    print(f"최대 레이블: {max_label}")
    
    # 클러스터 생성
    clusters = []
    for cluster_id in range(max_label + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) >= min_points:  # 최소 포인트 수 확인
            cluster_pcd = non_floor_pcd.select_by_index(cluster_indices)
            clusters.append(cluster_pcd)
            print(f"클러스터 {cluster_id}: {len(cluster_indices)} 포인트")
    
    return clusters, labels

def visualize_point_clouds(pcd_list, window_name="ROR Visualization", point_size=0.5):
    if isinstance(pcd_list, o3d.geometry.PointCloud):
        pcd_list = [pcd_list]
        
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

def process_point_cloud(pcd, voxel_size=0.3):
    """
    포인트 클라우드 전처리 파이프라인
    """
    # 1. 평면 추정 및 변환
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.15,
        ransac_n=3,
        num_iterations=3000
    )
    [a, b, c, d] = plane_model
    print(f"평면 모델 계수: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
    print(f"평면 인라이어 수: {len(inliers)}")
    
    # 포인트 변환
    points = np.asarray(pcd.points)
    transformed_points = transform_to_plane_based_coordinates(points, a, b, c, d)
    print(f"변환된 포인트 수: {len(transformed_points)}")
    
    # 2. 격자 생성 및 높이 맵 계산
    grid_resolution = 0.3
    x_min, y_min = np.min(transformed_points[:, :2], axis=0)
    x_max, y_max = np.max(transformed_points[:, :2], axis=0)
    print(f"포인트 클라우드 범위: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}]")
    
    x_bins = np.arange(x_min, x_max, grid_resolution)
    y_bins = np.arange(y_min, y_max, grid_resolution)
    height_map = calculate_height_map(transformed_points, x_bins, y_bins, threshold=0.05)
    print(f"높이 맵 크기: {len(height_map)}")
    
    # 3. 비바닥 점 필터링
    threshold = 0.15
    non_floor_indices = []
    for idx, (x, y, z) in enumerate(transformed_points):
        x_idx = np.searchsorted(x_bins, x, side='right') - 1
        y_idx = np.searchsorted(y_bins, y, side='right') - 1
        if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
            smoothed_height = height_map.get((x_idx, y_idx), None)
            if smoothed_height is not None and abs(z - smoothed_height) > threshold:
                non_floor_indices.append(idx)
    
    # Floor points count
    print(f"비바닥 점 수: {len(non_floor_indices)}")

    # 비바닥 및 바닥 포인트
    non_floor_pcd = pcd.select_by_index(non_floor_indices)
    floor_pcd = pcd.select_by_index(non_floor_indices, invert=True)

    # 포인트 수 확인
    print("\n=== 포인트 클라우드 정보 ===")
    print(f"비바닥 점 수: {len(non_floor_pcd.points):,}")
    print(f"바닥 점 수: {len(floor_pcd.points):,}")
    
    # 비바닥 점 좌표 범위 확인
    non_floor_points = np.asarray(non_floor_pcd.points)
    if len(non_floor_points) > 0:
        print("\n=== 비바닥 점 좌표 범위 ===")
        print(f"X 범위: [{non_floor_points[:,0].min():.3f}, {non_floor_points[:,0].max():.3f}]")
        print(f"Y 범위: [{non_floor_points[:,1].min():.3f}, {non_floor_points[:,1].max():.3f}]")
        print(f"Z 범위: [{non_floor_points[:,2].min():.3f}, {non_floor_points[:,2].max():.3f}]")
    
    # 포인트 클라우드 유효성 검사
    print("\n=== 포인트 클라우드 유효성 검사 ===")
    print(f"non_floor_pcd가 비어있나요? {len(non_floor_pcd.points) == 0}")
    print(f"non_floor_pcd가 None인가요? {non_floor_pcd is None}")
    
    try:
        # 포인트 클라우드 기본 속성 확인
        print("\n=== 포인트 클라우드 속성 ===")
        print(f"포인트 수: {len(non_floor_pcd.points):,}")
        print(f"포인트 타입: {type(non_floor_pcd.points)}")
        print(f"차원: {np.asarray(non_floor_pcd.points).shape}")
        
        # 첫 몇 개의 포인트 출력
        if len(non_floor_pcd.points) > 0:
            print("\n=== 첫 5개 포인트 샘플 ===")
            points_array = np.asarray(non_floor_pcd.points)
            for i in range(min(5, len(points_array))):
                print(f"Point {i+1}: {points_array[i]}")
    
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        return []

    # 비바닥 점 (non-floor) 개수
    num_non_floor_points = len(non_floor_pcd.points)
    print(f"Number of non-floor points: {num_non_floor_points}")

    # 바닥 점 (floor) 개수
    num_floor_points = len(floor_pcd.points)
    print(f"Number of floor points: {num_floor_points}")
    
    # 4. 클러스터링
    if len(non_floor_indices) < 10:
        print("경고: 비바닥 점이 너무 적습니다!")
        return []
        
    # 클러스터링 파라미터 조정
    eps = 0.4  # 더 큰 값으로 조정 (예: 0.5 -> 0.8)
    min_points = 20  # 더 작은 값으로 조정 (예: 30 -> 10)
    clusters, labels = cluster_non_floor_points(non_floor_pcd, eps=eps, min_points=min_points)

    if len(clusters) == 0:
        print("클러스터가 발견되지 않았습니다. 파라미터를 조정해보세요.")
        return []

    print(f"검출된 클러스터 수: {len(clusters)}")
    for i, cluster in enumerate(clusters):
        print(f"클러스터 {i+1} 포인트 수: {len(cluster.points)}")
    
    # 결과 반환 형식 변경: (클러스터 리스트, 비바닥 포인트 클라우드)
    return clusters, non_floor_pcd