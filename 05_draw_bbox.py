import open3d as o3d
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
#from scipy.spatial import cKDTree

class GridDatabase:
    def __init__(self):
        self.data = {}  # 격자 데이터 저장 (격자 키 -> PCA/밀도 정보 및 변화 기록)

    def update(self, grid_key, pca_values, density, frame_index):
        if grid_key not in self.data:
            self.data[grid_key] = {
                "pca": pca_values,
                "density": density,
                "last_updated": frame_index,
                "unchanged_frames": 0
            }
        else:
            self.data[grid_key]["pca"] = pca_values
            self.data[grid_key]["density"] = density
            self.data[grid_key]["last_updated"] = frame_index
            self.data[grid_key]["unchanged_frames"] = 0  # 업데이트 시 변화 감지로 간주

    def mark_unchanged(self, changed_grids):
        """
        변화가 감지되지 않은 격자의 `unchanged_frames`를 증가.
        `changed_grids`에 속한 격자는 `unchanged_frames`를 초기화.
        """
        for key in self.data:
            if key in changed_grids:
                self.data[key]["unchanged_frames"] = 0  # 변화 감지 격자 초기화
            else:
                self.data[key]["unchanged_frames"] += 1  # 변화 없는 격자 증가

    def clean_old_entries(self, threshold=3):
        """
        `unchanged_frames` 값이 `threshold` 이상인 격자 데이터를 삭제.
        """
        self.data = {
            k: v for k, v in self.data.items()
            if v["unchanged_frames"] < threshold
        }

    def get_pca_density(self, grid_key):
        """
        특정 격자의 PCA와 밀도 값 반환.
        """
        return self.data[grid_key]["pca"], self.data[grid_key]["density"] if grid_key in self.data else (None, None)

    def get_all_keys(self):
        """
        데이터베이스의 모든 격자 키 반환.
        """
        return list(self.data.keys())


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

# 첫 프레임에서 클러스터가 포함된 격자점의 PCA와 밀도 계산 함수
def calculate_cluster_grid_pca_and_density(clusters, x_bins, y_bins):
    grid_pca = {}
    grid_density = {}

    for cluster in clusters:
        cluster_points = np.asarray(cluster.points)

        # 클러스터의 길이 계산 (축 정렬 바운딩 박스를 사용)
        bounding_box = cluster.get_axis_aligned_bounding_box()
        cluster_length = np.linalg.norm(bounding_box.get_extent())  # 바운딩 박스의 대각선 길이

        # 길이가 2m 이내인 클러스터만 처리
        if cluster_length > 2.0:
            continue

        x_indices = np.digitize(cluster_points[:, 0], bins=x_bins) - 1
        y_indices = np.digitize(cluster_points[:, 1], bins=y_bins) - 1

        for i, (x_idx, y_idx) in enumerate(zip(x_indices, y_indices)):
            if x_idx < 0 or y_idx < 0 or x_idx >= len(x_bins) - 1 or y_idx >= len(y_bins) - 1:
                continue  # 격자 범위를 벗어난 경우 스킵

            # Z 값이 2m 이하인지 확인
            if cluster_points[i, 2] > 2.0:
                continue  # Z 값이 2m 초과인 경우 스킵

            key = (x_idx, y_idx)
            if key not in grid_pca:
                grid_pca[key] = []   # (x_index, y_index) : key, pca value : value
                grid_density[key] = []

            grid_pca[key].append(cluster_points[i])
            grid_density[key].append(cluster_points[i])

    # PCA와 밀도 계산
    for key in list(grid_pca.keys()):
        grid_points = np.array(grid_pca[key])

        # PCA 계산
        if len(grid_points) >= 3:
            pca = PCA(n_components=3)
            pca.fit(grid_points)
            grid_pca[key] = pca.explained_variance_ratio_
        else:
            grid_pca[key] = [0, 0, 0]

        # 밀도 계산
        density_points = np.array(grid_density[key])
        if len(density_points) >= 5:
            nbrs = NearestNeighbors(n_neighbors=5).fit(density_points)
            distances, _ = nbrs.kneighbors(density_points)
            grid_density[key] = 1 / distances.mean(axis=1).mean()
        else:
            grid_density[key] = 0

    return grid_pca, grid_density  # 대각선 2m 보다 작은 클러스터가 포함된 격자점들의 2m 아래의 위치한 포인트 클라우드들의 pca 및 밀도 분석

# 격자 PCA와 밀도 계산 함수
def calculate_database_grid_pca_and_density(database, points, x_bins, y_bins, height_threshold=2.0):
    # 높이 제한 적용
    filtered_points = points[points[:, 2] <= height_threshold]  # Z 좌표가 height_threshold 이하인 포인트만 사용
    
    x_indices = np.digitize(filtered_points[:, 0], bins=x_bins) - 1
    y_indices = np.digitize(filtered_points[:, 1], bins=y_bins) - 1

    grid_pca = {}
    grid_density = {}

    for grid_key in database.data.keys():
        x_min = x_bins[grid_key[0]]
        x_max = x_bins[grid_key[0] + 1]
        y_min = y_bins[grid_key[1]]
        y_max = y_bins[grid_key[1] + 1]

        # 격자 내 포인트 필터링
        grid_indices = [
            i for i, (x, y, z) in enumerate(filtered_points) 
            if x_min <= x < x_max and y_min <= y < y_max and z <= height_threshold
        ]

        grid_points = filtered_points[grid_indices]

        # PCA 계산
        if len(grid_points) >= 3:
            pca = PCA(n_components=3)
            pca.fit(grid_points)
            grid_pca[grid_key] = pca.explained_variance_ratio_
        else:
            grid_pca[grid_key] = [0, 0, 0]

        # 밀도 계산
        if len(grid_points) >= 5:
            nbrs = NearestNeighbors(n_neighbors=5).fit(grid_points)
            distances, _ = nbrs.kneighbors(grid_points)
            grid_density[grid_key] = 1 / distances.mean(axis=1).mean()
        else:
            grid_density[grid_key] = 0

    return grid_pca, grid_density ### 높이 2m 이하의 포인트들의 pca 및 밀도 분석


def detect_changes(database, current_pca, current_density, frame_index, points, x_bins, y_bins, pca_threshold=0.1, density_threshold=1, height_threshold=2.0):
    changed_grids = []   #변화하는 격자점들 모아놓음
    changed_grids_info = {}  # 딕셔너리로 선언

    for grid_key, data in database.data.items():
        prev_pca = data["pca"]
        prev_density = data["density"]

        if grid_key in current_pca:
            # 격자의 X, Y 범위 계산
            x_min = x_bins[grid_key[0]]
            x_max = x_bins[grid_key[0] + 1]
            y_min = y_bins[grid_key[1]]
            y_max = y_bins[grid_key[1] + 1]

            # 격자 내 높이 2m 이하의 포인트 필터링
            grid_indices = [
                i for i, (x, y, z) in enumerate(points) 
                if x_min <= x < x_max and y_min <= y < y_max and z <= height_threshold
            ]

            if len(grid_indices) < 1:  # 해당 격자에 유효 포인트가 없으면 건너뜀
                continue

            # PCA와 밀도 변화 계산
            pca_change = np.linalg.norm(np.array(current_pca[grid_key]) - np.array(prev_pca))
            density_change = abs(current_density[grid_key] - prev_density)

            # 변화 기준에 따라 변화 격자 추가
            if pca_change > pca_threshold or density_change > density_threshold:
                changed_grids.append(grid_key)

                # # 변화한 격자점의 정보를 저장
                # changed_grids_info[grid_key] = {  # 딕셔너리로 접근
                #     "pca_change": pca_change,
                #     "density_change": density_change,
                #     "current_pca": current_pca[grid_key],
                #     "current_density": current_density[grid_key]
                # }
            database.update(grid_key, current_pca[grid_key], current_density[grid_key], frame_index)

    # # 모든 변화 감지된 격자 처리 (최종적으로 변화 감지 완료 후 호출)
    # process_nearby_clusters(
    #     changed_grids=changed_grids,
    #     points=points,
    #     clusters=clusters,
    #     database=database,
    #     x_bins=x_bins,
    #     y_bins=y_bins,
    #     frame_index=frame_index,
    #     height_threshold=2.0
    # )

    return changed_grids

# 변화 감지된 격자 내 클러스터 라벨 및 추가 격자 처리
def process_changed_grids_and_update_db(
    changed_grids, clusters, labels, non_floor_points, x_bins, y_bins, grid_db, frame_index, height_threshold=2.0
):
    new_grids = {}  # 새롭게 추가될 격자 데이터를 저장
    
    for grid_key in changed_grids:
        x_min = x_bins[grid_key[0]]
        x_max = x_bins[grid_key[0] + 1]
        y_min = y_bins[grid_key[1]]
        y_max = y_bins[grid_key[1] + 1]

        # 해당 격자 내의 모든 포인트와 라벨 필터링
        grid_indices = [
            i for i, (x, y, z) in enumerate(non_floor_points)
            if x_min <= x < x_max and y_min <= y < y_max and z <= height_threshold
        ]
        grid_labels = labels[grid_indices]

        # 라벨 값으로 클러스터 가져오기
        cluster_ids = np.unique(grid_labels)
        cluster_ids = cluster_ids[cluster_ids != -1]  # 유효 클러스터만 사용

        for cluster_id in cluster_ids:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_points = non_floor_points[cluster_indices]

            # 클러스터가 포함된 격자 찾기
            x_indices = np.digitize(cluster_points[:, 0], bins=x_bins) - 1
            y_indices = np.digitize(cluster_points[:, 1], bins=y_bins) - 1
            cluster_grids = {(x_idx, y_idx) for x_idx, y_idx in zip(x_indices, y_indices)
                             if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1}

            for cluster_grid in cluster_grids:
                if cluster_grid not in grid_db.get_all_keys():  # 데이터베이스에 없는 격자
                    if cluster_grid not in new_grids:
                        new_grids[cluster_grid] = []

                    new_grids[cluster_grid].extend(cluster_points)

    # 새 격자 추가 및 PCA, 밀도 계산
    for new_grid, points in new_grids.items():
        grid_points = np.array(points)

        # PCA 계산
        if len(grid_points) >= 3:
            pca = PCA(n_components=3)
            pca.fit(grid_points)
            pca_values = pca.explained_variance_ratio_
        else:
            pca_values = [0, 0, 0]

        # 밀도 계산
        if len(grid_points) >= 5:
            nbrs = NearestNeighbors(n_neighbors=5).fit(grid_points)
            distances, _ = nbrs.kneighbors(grid_points)
            density = 1 / distances.mean(axis=1).mean()
        else:
            density = 0

        # 데이터베이스 업데이트
        grid_db.update(new_grid, pca_values, density, frame_index)

    print(f"[INFO] Updated database with {len(new_grids)} new grids.")

# 바운딩 박스 함수 (해당 클러스터에 바운딩 박스 그리기)
def draw_bounding_boxes_for_clusters(
    changed_grids, grid_db, clusters, labels, non_floor_points, height_threshold=2.0
):
    bounding_boxes = []

    # 변화 감지된 격자에 속한 클러스터를 찾고 바운딩 박스 생성 (빨간색)
    changed_cluster_ids = set()
    for grid_key in changed_grids:
        # 격자 내 포인트 필터링
        x_min = x_bins[grid_key[0]]
        x_max = x_bins[grid_key[0] + 1]
        y_min = y_bins[grid_key[1]]
        y_max = y_bins[grid_key[1] + 1]
        grid_indices = [
            i for i, (x, y, z) in enumerate(non_floor_points)
            if x_min <= x < x_max and y_min <= y < y_max and z <= height_threshold
        ]
        grid_labels = labels[grid_indices]
        changed_cluster_ids.update(np.unique(grid_labels[grid_labels != -1]))

    # 데이터베이스에 속한 모든 클러스터 가져오기 (changed_cluster_ids 제외)
    all_cluster_ids = set()
    for grid_key in grid_db.get_all_keys():
        x_min = x_bins[grid_key[0]]
        x_max = x_bins[grid_key[0] + 1]
        y_min = y_bins[grid_key[1]]
        y_max = y_bins[grid_key[1] + 1]
        grid_indices = [
            i for i, (x, y, z) in enumerate(non_floor_points)
            if x_min <= x < x_max and y_min <= y < y_max and z <= height_threshold
        ]
        grid_labels = labels[grid_indices]
        all_cluster_ids.update(np.unique(grid_labels[grid_labels != -1]))

    # `changed_cluster_ids`를 제외한 데이터베이스 클러스터
    database_cluster_ids = all_cluster_ids - changed_cluster_ids

    # 바운딩 박스 생성
    for cluster_id in changed_cluster_ids.union(database_cluster_ids):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_points = non_floor_points[cluster_indices]

        if len(cluster_points) > 0:
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)

            bbox = cluster_pcd.get_axis_aligned_bounding_box()
            if cluster_id in changed_cluster_ids:
                bbox.color = (1, 0, 0)  # 빨간색 (변화 감지된 클러스터)
            else:
                bbox.color = (0, 1, 0)  # 초록색 (기존 데이터베이스 클러스터)
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
def visualize_with_bounding_boxes(pcd_list, bounding_boxes, labels, window_name="Clusters with Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    
    # 클러스터에 고유 색상 적용
    if labels is not None and len(labels) > 0:
        max_label = labels.max()
        cluster_colors = np.random.rand(max_label + 1, 3)  # 각 클러스터에 고유 색상 생성
        colors = np.zeros((len(labels), 3))  # 포인트 수 만큼 색상 초기화
        for i in range(len(labels)):
            if labels[i] != -1:  # 노이즈는 색상을 설정하지 않음
                colors[i] = cluster_colors[labels[i]]
        
        for pcd in pcd_list:
            pcd.colors = o3d.utility.Vector3dVector(colors)  # 클러스터 색상 적용

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

# 메인 실행 코드
if __name__ == "__main__":
    file_paths = [
        "data/04_zigzag_walk/pcd/pcd_000267.pcd",
        "data/04_zigzag_walk/pcd/pcd_000268.pcd",
        "data/04_zigzag_walk/pcd/pcd_000269.pcd",
        "data/04_zigzag_walk/pcd/pcd_000270.pcd",
        "data/04_zigzag_walk/pcd/pcd_000271.pcd",
        "data/04_zigzag_walk/pcd/pcd_000272.pcd",
        "data/04_zigzag_walk/pcd/pcd_000273.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000274.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000275.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000276.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000277.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000278.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000279.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000280.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000281.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000282.pcd",
        # "data/04_zigzag_walk/pcd/pcd_000283.pcd",
    ]
    grid_db = GridDatabase()
    
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
        threshold = 0.15 # threshold : 격자의 바닥 높이와의 차
        non_floor_indices = []
        for idx, (x, y, z) in enumerate(transformed_points):
            x_idx = np.searchsorted(x_bins, x, side='right') - 1
            y_idx = np.searchsorted(y_bins, y, side='right') - 1
            if 0 <= x_idx < len(x_bins) - 1 and 0 <= y_idx < len(y_bins) - 1:
                smoothed_height = height_map.get((x_idx, y_idx), None)
                if smoothed_height is not None and abs(z - smoothed_height) > threshold:
                    non_floor_indices.append(idx)
        
        non_floor_pcd = filtered_pcd.select_by_index(non_floor_indices)
        non_floor_points = np.asarray(non_floor_pcd.points)  # numpy 배열로 변환
        
        # 클러스터링
        labels = np.array(non_floor_pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))    #### 파라미터 조정 필요
        unique_labels = np.unique(labels)
        clusters = [non_floor_pcd.select_by_index(np.where(labels == cluster_id)[0]) for cluster_id in unique_labels if cluster_id != -1]

        # 첫 프레임 처리
        if frame_idx == 0:
            print("[DEBUG] Processing first frame for initial cluster grid analysis.")
            pca, density = calculate_cluster_grid_pca_and_density(clusters, x_bins, y_bins)
            for key in pca:
                grid_db.update(key, pca[key], density[key], frame_idx)
                
            # 데이터베이스 상태 출력
            print(f"[INFO] Number of grid points in database after frame {frame_idx}: {len(grid_db.get_all_keys())}")
            continue

       # 데이터베이스에 존재하는 격자 대해서만 PCA 및 밀도 계산 (현재)
        current_pca, current_density = calculate_database_grid_pca_and_density(
            database=grid_db,
            points=non_floor_points,
            x_bins=x_bins,
            y_bins=y_bins,
            height_threshold=2.0  # Z 좌표 제한하여 2m 이하의 포인트들의 pca 및 밀도 분석
        ) 
        # 현재 프레임에서 db에 존재하는 격자들에 대해 pca 및 밀도 계산
        
        # 변화 감지
        changed_grids = detect_changes(  ##변화 감지된 모든 격자점 인덱스들 모아놓음
            database=grid_db,
            current_pca=current_pca,
            current_density=current_density,
            frame_index=frame_idx,
            points=non_floor_points,  # 평면 변환된 포인트
            x_bins=x_bins,
            y_bins=y_bins,
            height_threshold=2.0  # 높이 제한 (2m 이하로 제한)
        )
        print(f"[DEBUG] Changed grids: {changed_grids}")

        # 변화 감지된 격자 처리 및 DB 업데이트
        process_changed_grids_and_update_db(
            changed_grids=changed_grids,
            clusters=clusters,
            labels=labels,
            non_floor_points=non_floor_points,
            x_bins=x_bins,
            y_bins=y_bins,
            grid_db=grid_db,
            frame_index=frame_idx,
            height_threshold=2.0
        )

        # 데이터베이스 상태 출력
        print(f"[INFO] Number of grid points in database after frame {frame_idx}: {len(grid_db.get_all_keys())}")
        
        # 변화 감지된 격자 및 데이터베이스 격자의 클러스터 바운딩 박스 생성
        bounding_boxes = draw_bounding_boxes_for_clusters(
            changed_grids=changed_grids,
            grid_db=grid_db,
            clusters=clusters,
            labels=labels,
            non_floor_points=non_floor_points,
            height_threshold=2.0
        )

        # # 시각화
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(non_floor_pcd)  # 포인트 클라우드 추가
        # for bbox in bounding_boxes:
        #     vis.add_geometry(bbox)  # 바운딩 박스 추가
        # vis.run()
        # vis.destroy_window()
        
        # 모든 격자 업데이트
        grid_db.mark_unchanged(changed_grids)  # 변화 없는 격자 마킹
        grid_db.clean_old_entries(threshold=3)  # 3 프레임 이상 변화 없는 격자 삭제
        
        # # 데이터베이스 상태 출력
        # print(f"[INFO] Number of grid points in database after frame {frame_idx}: {len(grid_db.get_all_keys())}")
        
        # 시각화
        visualize_with_bounding_boxes(
            pcd_list=[non_floor_pcd],
            bounding_boxes=bounding_boxes,
            labels=labels,
            window_name=f"Frame {frame_idx}: Clusters with Bounding Boxes",
            point_size=1.0
        )