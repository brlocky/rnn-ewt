from utils import predict
import seaborn as sns
from tensorflow.keras.models import load_model
from utils import prepare_training_data
from sklearn.preprocessing import StandardScaler
from utils import enrich_data


# Read the csv file
df, df_for_training = enrich_data('data/NASDAQ.csv')
# print(df_for_training)


# LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

# print(df_for_training_scaled)  # 7 columns, including the Date.


trainX, trainY = prepare_training_data(df_for_training_scaled, df_for_training)

# Load model
model = load_model('a_model')
print("Model loaded successfully.")


original, df_forecast = predict(model, df, df_for_training, scaler, trainX)

sns.lineplot(data=original, x='Date', y='Open')
sns.lineplot(data=df_forecast, x='Date', y='Open')
