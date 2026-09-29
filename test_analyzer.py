import pandas as pd
from analyzer import analyze_dataset

df = pd.DataFrame({
    "age": [25, 30, None, 45],
    "city": ["Pune", "Delhi", "Pune", "Mumbai"],
    "name": ["A", "B", "C", "D"],
    "passed": [True, False, True, True],
})

result = analyze_dataset(df)

for key, value in result.items():
    print("----", key)
    print(value)