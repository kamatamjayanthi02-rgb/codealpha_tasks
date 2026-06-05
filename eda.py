import pandas as pd
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("titanic.csv")

# Basic Information
print("First 5 Rows:")
print(data.head())

print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns)

print("\nStatistics:")
print(data.describe())

# Survival Rate by Gender
print("\nSurvival Rate by Gender:")
print(data.groupby("Sex")["Survived"].mean())

# Survival Rate by Passenger Class
print("\nSurvival Rate by Passenger Class:")
print(data.groupby("Pclass")["Survived"].mean())

# Graph 1: Survival Count
data["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.xlabel("0 = Did Not Survive, 1 = Survived")
plt.ylabel("Number of Passengers")
plt.savefig("survival_count.png")
plt.show()

# Graph 2: Survival Rate by Gender
gender_survival = data.groupby("Sex")["Survived"].mean()
gender_survival.plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.savefig("gender_survival.png")
plt.show()

# Graph 3: Survival Rate by Passenger Class
class_survival = data.groupby("Pclass")["Survived"].mean()
class_survival.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.savefig("class_survival.png")
plt.show()

# Graph 4: Age Distribution
data["Age"].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.savefig("age_distribution.png")
plt.show()
