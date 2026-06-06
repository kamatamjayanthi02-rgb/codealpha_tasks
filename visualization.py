import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("titanic.csv")

# Survival Count
data["Survived"].value_counts().plot(kind="bar")
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()
# Survival Rate by Gender
gender_survival = data.groupby("Sex")["Survived"].mean()

gender_survival.plot(kind="bar")
plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.savefig("gender_survival.png")
plt.show()
# Survival Rate by Passenger Class
class_survival = data.groupby("Pclass")["Survived"].mean()

class_survival.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.savefig("class_survival.png")
plt.show()
# Age Distribution
data["Age"].hist(bins=20)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.savefig("age_distribution.png")
plt.show()