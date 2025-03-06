import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise):
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, state_transition_matrix, control_input):
        # 状态预测
        self.state = np.dot(state_transition_matrix, self.state) + control_input
        # 协方差预测
        self.covariance = np.dot(state_transition_matrix, np.dot(self.covariance, state_transition_matrix.T)) + self.process_noise

    def update(self, measurement, observation_matrix):
        # 卡尔曼增益计算
        kalman_gain = np.dot(self.covariance, np.dot(observation_matrix.T, np.linalg.inv(np.dot(observation_matrix, np.dot(self.covariance, observation_matrix.T)) + self.measurement_noise)))
        # 状态更新
        self.state = self.state + np.dot(kalman_gain, (measurement - np.dot(observation_matrix, self.state)))
        # 协方差更新
        self.covariance = np.dot((np.eye(len(self.state)) - np.dot(kalman_gain, observation_matrix)), self.covariance)
