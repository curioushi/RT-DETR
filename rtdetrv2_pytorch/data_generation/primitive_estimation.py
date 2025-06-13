import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') # 显式设置后端
import matplotlib.pyplot as plt

def generate_ellipse_point_cloud(num_points=100, center_range=(-5, 5), angle_range=(0, 2 * np.pi), scale_range=(0.5, 3.0)):
    """
    生成一个具有随机位置、旋转和缩放的二维椭圆点云。

    参数:
    num_points (int): 点云中的点数。
    center_range (tuple): 椭圆中心坐标的随机范围 (min, max)。
    angle_range (tuple): 椭圆旋转角度的随机范围 (min, max) (弧度)。
    scale_range (tuple): 椭圆半轴长度缩放因子的随机范围 (min, max)。

    返回:
    np.ndarray: 形状为 (num_points, 2) 的点云数组。
    """

    # 1. 生成随机参数
    center_x = np.random.uniform(center_range[0], center_range[1])
    center_y = np.random.uniform(center_range[0], center_range[1])
    center = np.array([center_x, center_y])

    angle = np.random.uniform(angle_range[0], angle_range[1])

    # 确保半轴长度为正
    scale_a = np.random.uniform(scale_range[0], scale_range[1])
    scale_b = np.random.uniform(scale_range[0], scale_range[1])
    while scale_a <= 0:
        scale_a = np.random.uniform(scale_range[0], scale_range[1])
    while scale_b <= 0:
        scale_b = np.random.uniform(scale_range[0], scale_range[1])


    # 2. 在标准椭圆内部随机生成点 (中心在原点, 无旋转)
    # 首先在单位圆内部生成点
    random_angles = np.random.uniform(0, 2 * np.pi, num_points)
    # 为了确保点在圆内均匀分布，半径 r 的平方 r^2 应在 [0, R^2] 上均匀分布
    # 对于单位圆 R=1, 所以 r = sqrt(u) where u is uniform in [0,1]
    random_radii_sqrt = np.sqrt(np.random.uniform(0, 1, num_points))

    points = np.stack((
        random_radii_sqrt * np.cos(random_angles),
        random_radii_sqrt * np.sin(random_angles)
    ), axis=-1) # (num_points, 2), points are now inside a unit circle

    # 3. 应用缩放
    # 注意：这里我们先用单位圆生成点，然后根据 scale_a 和 scale_b 缩放
    points[:, 0] *= scale_a
    points[:, 1] *= scale_b

    # 4. 应用旋转
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    # points 是 (num_points, 2)，rotation_matrix 是 (2, 2)
    # 我们需要 (points @ rotation_matrix.T) 或者 (rotation_matrix @ points.T).T
    rotated_points = points @ rotation_matrix.T

    # 5. 应用平移
    translated_points = rotated_points + center

    return translated_points

def generate_rectangle_point_cloud(num_points=1000, center_range=(-5, 5), angle_range=(0, 2 * np.pi), scale_range=(0.99, 1.01)):
    """
    生成一个具有随机位置、旋转和缩放的二维实心长方形点云。

    参数:
    num_points (int): 点云中的点数。
    center_range (tuple): 长方形中心坐标的随机范围 (min, max) (作用于 x, y)。
    angle_range (tuple): 长方形旋转角度的随机范围 (min, max) (弧度)。
    scale_range (tuple): 长方形两个轴长度（宽度和高度）缩放因子的随机范围 (min, max)。

    返回:
    np.ndarray: 形状为 (num_points, 2) 的点云数组。
    """

    # 1. 生成随机参数
    center_x = np.random.uniform(center_range[0], center_range[1])
    center_y = np.random.uniform(center_range[0], center_range[1])
    center = np.array([center_x, center_y])

    # 旋转角度
    theta = np.random.uniform(angle_range[0], angle_range[1])

    # 确保尺度为正
    scale_width = np.random.uniform(scale_range[0], scale_range[1])
    scale_height = np.random.uniform(scale_range[0], scale_range[1])
    while scale_width <= 0:
        scale_width = np.random.uniform(scale_range[0], scale_range[1])
    while scale_height <= 0:
        scale_height = np.random.uniform(scale_range[0], scale_range[1])
    scales = np.array([scale_width, scale_height])

    # 2. 在单位长方形内部随机生成点 (中心在原点, 范围 [-0.5, 0.5] x [-0.5, 0.5])
    points = np.random.uniform(-0.5, 0.5, (num_points, 2)) # (num_points, 2)

    # 3. 应用缩放
    scaled_points = points * scales # Element-wise multiplication

    # 4. 应用旋转
    # 构建2D旋转矩阵
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # points 是 (num_points, 2)，rotation_matrix 是 (2, 2)
    rotated_points = scaled_points @ rotation_matrix.T

    # 5. 应用平移
    translated_points = rotated_points + center

    return translated_points, center, theta, scales

if __name__ == "__main__":
    # 生成点云
    num_points = 2000
    # cloud = generate_ellipse_point_cloud(num_points=num_points)
    cloud, center, theta, scales = generate_rectangle_point_cloud(num_points=num_points)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    scales = np.diag(scales)/4
    covariance_matrix2 = R @ (scales ** 2) @ R.T

    # 1. 计算均值 (估计的中心)
    estimated_center = np.mean(cloud, axis=0)

    # 2. 中心化点云
    centered_cloud = cloud - estimated_center

    # 3. 计算协方差矩阵
    cov_before_reduce = centered_cloud.reshape(-1, 2, 1) * centered_cloud.reshape(-1, 1, 2)
    covariance_matrix = np.mean(cov_before_reduce, axis=0) / 12 * 9
    print(covariance_matrix)
    print(covariance_matrix2)

    # 4. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 特征向量是列向量，eigenvectors[:, i] 对应 eigenvalues[i]
    # 我们希望最大的特征值对应第一主轴
    sort_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    # 估计的旋转角度 (第一主轴与x轴的夹角)
    estimated_angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    # 估计的半轴长度 (通常用特征值的平方根的倍数，例如2倍标准差)
    # 这里的 eigenvalues 是方差，所以 sqrt(eigenvalues) 是标准差
    estimated_scale_a = 2 * np.sqrt(eigenvalues[0]) # 对应第一主轴
    estimated_scale_b = 2 * np.sqrt(eigenvalues[1]) # 对应第二主轴


    # 可视化
    plt.figure(figsize=(10, 10))
    plt.scatter(cloud[:, 0], cloud[:, 1], s=10, label=f'{num_points} Points', alpha=0.6)

    # 绘制估计的中心
    plt.scatter(estimated_center[0], estimated_center[1], color='red', s=100, marker='x', label='Estimated Center')

    # 绘制估计的主轴
    # 主轴起点是估计的中心，方向是特征向量，长度是估计的半轴长度
    # 第一主轴
    axis1_direction = eigenvectors[:, 0]
    axis1_end = estimated_center + axis1_direction * estimated_scale_a
    axis1_start = estimated_center - axis1_direction * estimated_scale_a
    plt.plot([axis1_start[0], axis1_end[0]], [axis1_start[1], axis1_end[1]], color='red', linewidth=2, label=f'Estimated Major Axis (Length: {estimated_scale_a:.2f})')

    # 第二主轴
    axis2_direction = eigenvectors[:, 1]
    axis2_end = estimated_center + axis2_direction * estimated_scale_b
    axis2_start = estimated_center - axis2_direction * estimated_scale_b
    plt.plot([axis2_start[0], axis2_end[0]], [axis2_start[1], axis2_end[1]], color='green', linewidth=2, label=f'Estimated Minor Axis (Length: {estimated_scale_b:.2f})')


    plt.title("Random 2D Rectangle Point Cloud with PCA Estimation")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.axhline(0, color='grey', lw=0.5)
    plt.axvline(0, color='grey', lw=0.5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.axis('equal') # 确保 x 和 y 轴的比例相同，以正确显示长方形形状
    plt.show() 