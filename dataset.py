# Import the necessary library
import pandas as pd
import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# 1. Define the size of our synthetic dataset
num_samples = 200  # Let's create 200 synthetic samples

# 2. Generate random concentrations for each compound within the ranges from your files
# Using the min and max values from 'types of tulsi.xlsx' and 'phytochemical table.xlsx'
data = {
    'eugenol': np.random.uniform(0.5, 3.2, num_samples),    # Range from Rama/Krishna leaf %
    'rosmarinic_acid': np.random.uniform(0.2, 1.0, num_samples), # Range from Krishna/Vana %
    'ursolic_acid': np.random.uniform(0.1, 0.7, num_samples),    # Range from Krishna %
    'apigenin': np.random.uniform(0.05, 0.3, num_samples),       # Range from Krishna %
    'carvacrol': np.random.uniform(0.2, 1.8, num_samples)        # Range from phytochemical table
}

# Create the DataFrame for our Features (X)
df = pd.DataFrame(data)

# 3. Now, create the Target Outputs (y) based on LOGICAL RULES
# We use the chemical values to calculate a base score and then clip it to our 0-5 scale.

# PUNGENT: Driven by Eugenol and Carvacrol
df['pungent'] = (0.7 * df['eugenol'] + 0.8 * df['carvacrol']) / 2
df['pungent'] = np.clip(df['pungent'], 0, 5)  # Ensure score stays between 0 and 5

# BITTER: Driven by Ursolic Acid and Apigenin
df['bitter'] = (1.2 * df['ursolic_acid'] + 1.0 * df['apigenin'])
df['bitter'] = np.clip(df['bitter'], 0, 5)

# ASTRINGENT: Driven by Rosmarinic Acid
df['astringent'] = (1.5 * df['rosmarinic_acid'])
df['astringent'] = np.clip(df['astringent'], 0, 5)

# SWEET: Not present in these Tulsi compounds. Mostly 0, with some noise.
df['sweet'] = np.random.choice([0, 0, 0, 0, 1], size=num_samples) # Mostly zeros, occasional 1

# 4. Let's look at the first 5 rows of our complete, labeled dataset
print("First 5 rows of our synthetic labeled dataset:")
print(df.head().round(2))

# 5. Save this dataset to a CSV file for training
df.to_csv('tulsi_taste_training_dataset.csv', index=False)
print("\nDataset saved as 'tulsi_taste_training_dataset.csv'")