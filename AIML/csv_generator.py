import pandas as pd
import numpy as np

# Create random car data
np.random.seed(42)
data = {
    "ProductID": [201, 202, 203, 204],
    "ProductName": ["Laptop", "Phone", "Tablet", "Headphones"],
    "Price": [800, 600, 300, 100]
}

df = pd.DataFrame(data)

# Save as CSV
file_path = "E:\Python/products.csv"
df.to_csv(file_path, index=False)

file_path
