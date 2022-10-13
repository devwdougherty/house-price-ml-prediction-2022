import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('HousePricePrediction.csv')

# Printing first 5 records of the dataset
print(dataset.head())
# Dimensions of the dataset
print(dataset.shape)

# Data Preprocessing
# Lets categorize the features depending on their datatype (int, float, object) and then calculate the number of them
obj = (dataset.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:", len(object_cols))

int_ = (dataset.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:", len(num_cols))

float_ = (dataset.dtypes == 'float')
float_cols = list(float_[float_].index)
print("Float variables:", len(float_cols))

# Heatmap plot using seaborn library
plt.figure(figsize=(12, 6))

sns.heatmap(dataset.corr(),
            cmap = 'BrBG',
            fmt = '.2f',
            linewidths = 2,
            annot = True)

# Barplot
unique_values = []

for col in object_cols:
    unique_values.append(dataset[col].unique().size)

plt.figure(figsize=(10,6))
plt.title('No. Unique values of Categorical Features')
plt.xticks(rotation=90)
sns.barplot(x=object_cols, y=unique_values)


