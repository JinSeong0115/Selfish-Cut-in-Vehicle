import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import optuna  # Optuna 추가

# 데이터 로드
file_path = "/home/kim_js/car_project/yolov9/YOLOv9-DeepSORT-Object-Tracking/lane_change/data/merged_final_revised.csv"
labeling_file_path = "/home/kim_js/car_project/yolov9/YOLOv9-DeepSORT-Object-Tracking/lane_change/data/label_revised.csv"

df = pd.read_csv(file_path)
labeling_df = pd.read_csv(labeling_file_path)

# Dataset 클래스 정의
class LaneChangeDataset(Dataset):
    def __init__(self, grouped_data, label_col):
        self.data = grouped_data
        self.label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = self.data.iloc[idx][['speed', 'acceleration', 'angle', 'center_x', 'center_y']].values
        features = features.astype(float)
        label = self.data.iloc[idx][self.label_col]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Transformer 모델 설정
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, dim_feedforward, dropout, num_classes=2):
        super(TransformerModel, self).__init__()
        # 입력을 d_model 차원으로 맞추기 위한 선형 변환 레이어 추가
        self.input_fc = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=1, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_fc(x)  # 입력 차원을 d_model로 변환
        x = self.transformer_encoder(x)
        x = self.fc(x)
        return x

# 데이터 전처리 함수
def prepare_data(df, labeling_df, label_col, case_number, valid_ids):
    if case_number == 1:
        df_case = df[(df['center_x'] >= 3420) & (df['center_x'] <= 3840)]
    elif case_number == 2:
        df_case = df[(df['center_x'] >= 3083) & (df['center_x'] <= 3420)]
    elif case_number == 3:
        df_case = df[(df['center_x'] >= 2747) & (df['center_x'] <= 3083)]

    # 주어진 valid_ids만 유지
    df_case = df_case[df_case['id'].isin(valid_ids)]

    labeled_data = labeling_df[labeling_df['id'].isin(df_case['id'].unique())]
    merged_data = pd.merge(df_case, labeled_data[['id', label_col]], on='id', how='left')
    merged_data[label_col] = merged_data[label_col].fillna(0)

    return merged_data

# 공통된 ID만 필터링
def filter_valid_ids(df):
    ids_3420_3840 = df[(df['center_x'] >= 3420) & (df['center_x'] <= 3840)]['id'].unique()
    ids_3083_3420 = df[(df['center_x'] >= 3083) & (df['center_x'] <= 3420)]['id'].unique()
    ids_2747_3083 = df[(df['center_x'] >= 2747) & (df['center_x'] <= 3083)]['id'].unique()

    # 세 구간의 공통 ID들 추출
    common_ids = set(ids_3420_3840).intersection(ids_3083_3420).intersection(ids_2747_3083)
    
    return list(common_ids)

# Train/Test Split 함수 (차선 변경 여부별로 나눔)
def split_train_test_by_id(data, label_col, test_size=0.3):
    # 차선 변경하지 않는 차량 (label = 0)과 차선 변경 차량 (label = 1)의 ID 추출
    no_lane_change_ids = data[data[label_col] == 0]['id'].unique()
    lane_change_ids = data[data[label_col] == 1]['id'].unique()

    # 7:3으로 Train/Test로 나눔
    no_lane_change_train_ids, no_lane_change_test_ids = train_test_split(no_lane_change_ids, test_size=test_size, random_state=42)
    lane_change_train_ids, lane_change_test_ids = train_test_split(lane_change_ids, test_size=test_size, random_state=42)

    # ID를 기준으로 Train/Test 데이터셋 구성
    train_set = data[data['id'].isin(no_lane_change_train_ids) | data['id'].isin(lane_change_train_ids)]
    test_set = data[data['id'].isin(no_lane_change_test_ids) | data['id'].isin(lane_change_test_ids)]

    return train_set, test_set

# 모델 학습 함수
def train_model(df_grouped, label_col, d_model, dim_feedforward, dropout, lr, input_dim=5, epochs=50):
    dataset = LaneChangeDataset(df_grouped, label_col)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerModel(input_dim=input_dim, d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for features, labels in dataloader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model

# 모델 평가 함수
def evaluate_model(model, test_df, labeling_df, label_col, case_number):
    predictions = []
    true_labels = []
    labeling_filtered = labeling_df[labeling_df[label_col] == 1]

    for vehicle_id in test_df['id'].unique():
        vehicle_data = test_df[test_df['id'] == vehicle_id]
        features = vehicle_data[['speed', 'acceleration', 'angle', 'center_x', 'center_y']].values
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        try:
            with torch.no_grad():
                output = model(features)
                pred = torch.argmax(output, dim=-1)
                most_common_pred = torch.mode(pred).values.item()
                predictions.append(most_common_pred)
        except Exception as e:
            print(f"Model prediction error for vehicle_id {vehicle_id}: {e}")
            continue

        true_label = 1 if vehicle_id in labeling_filtered['id'].values else 0
        true_labels.append(true_label)

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    return accuracy

# Optuna의 objective 함수 정의
def objective(trial):
    # 탐색할 하이퍼파라미터 정의
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    d_model = trial.suggest_categorical('d_model', [32, 64, 128])
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [128, 256, 512])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # 학습 및 평가 실행 (1구간 학습 및 평가로 설정)
    valid_ids = set(filter_valid_ids(df))
    df_case_1 = prepare_data(df, labeling_df, 'label_1', 1, valid_ids)
    train_df_case_1, test_df_case_1 = split_train_test_by_id(df_case_1, 'label_1')

    # 모델 학습 및 평가
    model = train_model(train_df_case_1, 'label_1', d_model, dim_feedforward, dropout, lr)
    accuracy = evaluate_model(model, test_df_case_1, labeling_df, 'label_1', 1)

    return accuracy

# Optuna를 사용해 최적의 파라미터 탐색
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)

# 최적 파라미터 출력
print("Best trial:")
trial = study.best_trial
print(f"  Accuracy: {trial.value}")
print(f"  Best hyperparameters: {trial.params}")

# 학습 및 평가 실행
valid_ids = set(filter_valid_ids(df))

# Case 1: 구간 1 (3420~3840) 학습
df_case_1 = prepare_data(df, labeling_df, 'label_1', 1, valid_ids)
train_df_case_1, test_df_case_1 = split_train_test_by_id(df_case_1, 'label_1')

# Case 1에서 LC가 1인 ID 제외
exclude_ids_case_1 = set(df_case_1[df_case_1['label_1'] == 1]['id'])

# Case 2: 구간 2 (3083~3420) 학습, Case 1에서 LC=1인 ID 제외
valid_ids_case_2 = valid_ids.difference(exclude_ids_case_1)
df_case_2 = prepare_data(df, labeling_df, 'label_2', 2, valid_ids_case_2)
train_df_case_2, test_df_case_2 = split_train_test_by_id(df_case_2, 'label_2')

# Case 2에서 LC가 1인 ID 제외
exclude_ids_case_2 = set(df_case_2[df_case_2['label_2'] == 1]['id'])

# Case 3: 구간 3 (2747~3083) 학습, Case 1과 Case 2에서 LC=1인 ID 제외
valid_ids_case_3 = valid_ids_case_2.difference(exclude_ids_case_2)
df_case_3 = prepare_data(df, labeling_df, 'label_3', 3, valid_ids_case_3)
train_df_case_3, test_df_case_3 = split_train_test_by_id(df_case_3, 'label_3')

# 각 구간에서 학습 및 평가
for case, train_df, test_df, label_col in [(1, train_df_case_1, test_df_case_1, 'label_1'), 
                                           (2, train_df_case_2, test_df_case_2, 'label_2'), 
                                           (3, train_df_case_3, test_df_case_3, 'label_3')]:
    if len(train_df) == 0:
        print(f"No data available for case {case} and {label_col}.")
    else:
        # 학습 데이터 정보 출력
        num_ids = train_df['id'].nunique()
        num_samples = len(train_df)
        label_1_count = train_df[train_df[label_col] == 1].shape[0]
        label_0_count = train_df[train_df[label_col] == 0].shape[0]

        # Train 데이터에서 LC(1)와 No LC(0)의 ID 개수
        train_lane_change_ids = train_df[train_df[label_col] == 1]['id'].nunique()
        train_no_lane_change_ids = train_df[train_df[label_col] == 0]['id'].nunique()

        print(f"Training model for case {case} and {label_col}:")
        print(f" - Number of IDs in train set: {num_ids}")
        print(f" - Number of samples in train set: {num_samples}")
        print(f" - Number of label=1 samples (lane change): {label_1_count}")
        print(f" - Number of label=0 samples (no lane change): {label_0_count}")
        print(f" - Number of lane change IDs in train set: {train_lane_change_ids}")
        print(f" - Number of no lane change IDs in train set: {train_no_lane_change_ids}")

        # 모델 학습
        model = train_model(train_df, label_col, trial.params['d_model'], trial.params['dim_feedforward'], trial.params['dropout'], trial.params['lr'])

        # 평가 데이터 정보 출력
        num_test_ids = test_df['id'].nunique()
        num_test_samples = len(test_df)
        test_label_1_count = test_df[test_df[label_col] == 1].shape[0]
        test_label_0_count = test_df[test_df[label_col] == 0].shape[0]

        # Test 데이터에서 LC(1)와 No LC(0)의 ID 개수
        test_lane_change_ids = test_df[test_df[label_col] == 1]['id'].nunique()
        test_no_lane_change_ids = test_df[test_df[label_col] == 0]['id'].nunique()

        print(f"Evaluating model for case {case} and {label_col}:")
        print(f" - Number of IDs in test set: {num_test_ids}")
        print(f" - Number of samples in test set: {num_test_samples}")
        print(f" - Number of label=1 samples in test set: {test_label_1_count}")
        print(f" - Number of label=0 samples in test set: {test_label_0_count}")
        print(f" - Number of lane change IDs in test set: {test_lane_change_ids}")
        print(f" - Number of no lane change IDs in test set: {test_no_lane_change_ids}")

        # 모델 평가
        evaluate_model(model, test_df, labeling_df, label_col, case)

# 공통으로 존재하는 ID 출력
print(f"4개 구간에 공통으로 존재하는 ID 개수: {len(valid_ids)}")
