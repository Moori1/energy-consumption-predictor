import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import seaborn as sns
import torch.optim as optim
import joblib

df  = pd.read_csv("./powerconsumption.csv")
df.head() #Show the first lines of the dataframe
df['Datetime']=pd.to_datetime(df.Datetime)
df.sort_values(by='Datetime', ascending=True, inplace=True)

chronological_order = df['Datetime'].is_monotonic_increasing

time_diffs = df['Datetime'].diff()
equidistant_timestamps = time_diffs.nunique() == 1


def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['day'] = df.index.month
    df['year'] = df.index.year
    df['season'] = df['month'] % 12 // 3 + 1
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Additional features
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
    df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
    df['is_quarter_start'] = (df['dayofmonth'] == 1) & (df['month'] % 3 == 1).astype(int)
    df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max'))

    # Additional features
    df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
    df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
    df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)

    # Minute-level features
    df['minute_of_day'] = df['hour'] * 60 + df['minute']
    df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']

    return df.astype(float)




# Debugging
# df['Datetime']=pd.to_datetime(df.Datetime)
# df.sort_values(by='Datetime', ascending=True, inplace=True)
#
# chronological_order = df['Datetime'].is_monotonic_increasing
#
# time_diffs = df['Datetime'].diff()
# equidistant_timestamps = time_diffs.nunique() == 1
#
# print(chronological_order, equidistant_timestamps)


# Debugging
# print(df.isna().sum())

df = df.set_index('Datetime')
df = df.drop('GeneralDiffuseFlows', axis=1)
df = df.drop('DiffuseFlows', axis=1)
df = create_features(df)
df.to_csv('powerconsumption-expanded.csv', index=False)

# Debugging
# print(df[[ 'year', 'month', 'day','minute', 'dayofyear', 'weekofyear', 'quarter', 'season']].head())


# Debugging 1
# # Calculate correlation matrix
# correlation_matrix = df[['Temperature', 'Humidity', 'WindSpeed', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']].corr()
# # Create a heatmap of the correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
# plt.title('Correlation Heatmap')
# plt.show()


# Debugging 2
# sns.pairplot(df[['Temperature', 'Humidity', 'WindSpeed', 'PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']])
# plt.show()



# Debugging 3
# # Time series plot for PowerConsumption
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='Datetime', y='PowerConsumption_Zone1', data=df, label='Zone 1')
# plt.xlabel('Datetime')
# plt.ylabel('Power Consumption')
# plt.title('Power Consumption Over Time')
# plt.show()


# Debugging 4
# # Time series plot for PowerConsumption
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='Datetime', y='PowerConsumption_Zone2', data=df, label='Zone 2')
# plt.xlabel('Datetime')
# plt.ylabel('Power Consumption')
# plt.title('Power Consumption Over Time')
# plt.show()

# Debugging 5
# # Time series plot for PowerConsumption
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='Datetime', y='PowerConsumption_Zone3', data=df, label='Zone 3')
# plt.xlabel('Datetime')
# plt.ylabel('Power Consumption')
# plt.title('Power Consumption Over Time')
# plt.show()


# Separate the input features (X) and target variables (y)
X = df.drop(['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3'], axis=1)
y = df[['PowerConsumption_Zone1', 'PowerConsumption_Zone2', 'PowerConsumption_Zone3']]

# Initialize StandardScaler for y
scaler_y = StandardScaler()

# Fit and transform  y
y_scaled = scaler_y.fit_transform(y)

joblib.dump(scaler_y, "scaler_y.pkl")  # Save the scaler


X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.25, shuffle=False)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# Define MLP model
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define CNN model
class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=2)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 12, 50)  # Adjust dimensions as needed
        self.fc2 = nn.Linear(50, 3)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# Define LSTM model
class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, 50, batch_first=True)
        self.fc = nn.Linear(50, 3)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return x

# Training function
def train_model(model, criterion, optimizer, train_loader, epochs):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    return model



# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.append(outputs)
            actuals.append(y_batch)
    predictions = torch.cat(predictions, dim=0)
    actuals = torch.cat(actuals, dim=0)
    return predictions, actuals

# Train and evaluate MLP
mlp_model = MLP(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.0003)
mlp_model = train_model(mlp_model, criterion, optimizer, train_loader, epochs=40)
predictions, actuals = evaluate_model(mlp_model, test_loader)

# Calculate metrics
mse = torch.mean((predictions - actuals) ** 2).item()
mae = torch.mean(torch.abs(predictions - actuals)).item()
print("MLP - Mean squared error on test set: {:.4f}".format(mse))
print("MLP - Mean absolute error on test set: {:.4f}".format(mae))

# Save the model to a .pth file
torch.save(mlp_model.state_dict(), "energy_model.pth")
