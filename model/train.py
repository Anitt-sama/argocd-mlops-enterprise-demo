from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)

# Synthetic training data
area = rng.uniform(500, 3000, 200)
bedrooms = rng.integers(1, 6, 200)

X = np.column_stack((area, bedrooms))
y = 150000 + 180 * area + 25000 * bedrooms

model = LinearRegression()
model.fit(X, y)

output = Path(__file__).resolve().parent / "house_price_model.joblib"
joblib.dump(model, output)

print(f"Model trained and saved to {output}")
