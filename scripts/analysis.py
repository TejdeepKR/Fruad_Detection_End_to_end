import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/preprocessed_data.csv")

print("First 5 rows ::", df.head(5))
print("Data Summmary ::" , df.describe())
print(" Value Count ::", df["label"].value_counts())

#plot

sns.countplot(data=df ,x="label")
plt.title("Label Distribution")
plt.savefig("data/class_distribution.png")

print(" Image saved successfully")

