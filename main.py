
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bank-additional-full.csv", delimiter=';')

df.describe()

plt.figure(figsize=(6, 4))
sns.countplot(x='y', data=df)
plt.title('Distribution of Target Variable')
plt.show()


numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
    plt.show()


for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='y', y=feature, data=df)
    plt.title(f'{feature} vs. Target Variable')
    plt.show()


categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=feature, hue='y', data=df)
    plt.title(f'{feature} vs. Target Variable')
    plt.xticks(rotation=45)
    plt.show()


df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)



df_encoded['interaction_feature'] = df_encoded['age'] * df_encoded['campaign']


df.dropna(inplace=True)

import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
X=df_encoded.drop('y',axis=1)
y=df_encoded['y']
y = (y == 'yes').astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import seaborn as sns
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import tensorflow.keras as keras

print('X_train:',X_train.shape)
print('y_train:',y_train.shape)
print('X_test:',X_test.shape)
print('y_test:',y_test.shape)


from tensorflow.keras import layers 
from tensorflow.keras.optimizers import Adam
from tensorflow. keras.models import Sequential
model = Sequential()
model.add (layers.Dense(32,activation='sigmoid', input_shape=(X_train.shape[1],)))
model.add(layers.Dense(1, activation='linear'))
model.summary()

model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60)

optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_acc}')


y_pred = (model.predict(X_test) > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
