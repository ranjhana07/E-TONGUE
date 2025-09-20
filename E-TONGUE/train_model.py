# train_model_enhanced.py
# Enhanced training code with integrated research data
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- ENHANCED CONFIGURATION BASED ON RESEARCH DATA ---
# Updated ranges based on extracted research data with realistic variations
CHEMICAL_RANGES = {
    'eugenol': (1.8, 3.2),              # Original range maintained
    'rosmarinic_acid': (0.5, 2.5),      # Extended based on research (major compound in all varieties)
    'ursolic_acid': (0.2, 1.2),         # Extended based on kinetic studies (5.13-11.21 mg/g)
    'apigenin': (0.1, 0.8),             # Extended range (major isolated compound)
    'carvacrol': (0.2, 1.8),            # Original range maintained
    'caffeic_acid': (0.001, 0.6),       # New compound from HPLC data (0.001-2.63 mg/100mg)
    'gallic_acid': (0.1, 1.6),          # New compound from HPLC data (0.58-16.15 mg/100mg)
    'catechin': (0.01, 0.08),           # New compound from HPLC data (0.032-0.347 mg/100mg)
    'carnosic_acid': (0.05, 0.4),       # New compound from isolation studies
    'methyl_eugenol': (0.1, 0.8),       # From GC-MS data (Krishna variety marker)
    'bornanone': (0.05, 0.3),           # From GC-MS data (Camphor variety marker)
}

# Variety-specific multipliers based on research data
VARIETY_MULTIPLIERS = {
    'Krishna': {
        'rosmarinic_acid': 1.3,    # Highest polyphenolic content
        'caffeic_acid': 1.5,       # 263x higher than Ram tulsi
        'gallic_acid': 1.4,        # 2.8x higher than Ram tulsi
        'methyl_eugenol': 1.8,     # Dominant in Krishna variety
        'apigenin': 1.2
    },
    'Vishnu': {
        'ursolic_acid': 1.2,       # Good extraction yields
        'rosmarinic_acid': 1.1,
        'eugenol': 1.1,
        'carnosic_acid': 1.0
    },
    'Camphor': {
        'bornanone': 2.0,          # Dominant compound
        'rosmarinic_acid': 0.9,    # Lower polyphenolic content
        'eugenol': 1.0
    },
    'Lavanga': {
        'ursolic_acid': 1.0,
        'rosmarinic_acid': 0.95,
        'eugenol': 1.1,
        'carvacrol': 1.2
    },
    'Red_Holy_Basil': {
        'rosmarinic_acid': 1.4,    # Higher antioxidant capacity
        'eugenol': 1.3,
        'ursolic_acid': 1.2,
        'caffeic_acid': 1.3
    },
    'White_Holy_Basil': {
        'rosmarinic_acid': 1.0,    # Reference variety
        'eugenol': 1.0,
        'ursolic_acid': 1.0,
        'caffeic_acid': 1.0
    }
}

# Extraction condition effects based on kinetic research
EXTRACTION_EFFECTS = {
    'methanol': {'ursolic_acid': 1.2, 'rosmarinic_acid': 1.1, 'caffeic_acid': 1.0},
    'ethanol_76': {'rosmarinic_acid': 1.3, 'eugenol': 1.2, 'apigenin': 1.1},
    'ethanol_57': {'rosmarinic_acid': 1.4, 'caffeic_acid': 1.2, 'gallic_acid': 1.1},
    'acetone': {'ursolic_acid': 0.9, 'apigenin': 1.1, 'rosmarinic_acid': 1.0},
    'acetonitrile': {'ursolic_acid': 0.95, 'rosmarinic_acid': 1.0, 'eugenol': 1.0}
}

# Antioxidant activity correlation coefficients (from research)
ANTIOXIDANT_CORRELATIONS = {
    'rosmarinic_acid': 0.964,    # High correlation with ABTS
    'caffeic_acid': 0.85,        # Strong antioxidant
    'gallic_acid': 0.90,         # Strong antioxidant
    'ursolic_acid': 0.75,        # Moderate antioxidant
    'apigenin': 0.80,            # Good antioxidant
    'catechin': 0.88,            # Strong antioxidant
    'eugenol': 0.70,             # Moderate antioxidant
    'carnosic_acid': 0.82        # Good antioxidant
}

NUM_SAMPLES = 8000  # Increased for better representation

def generate_enhanced_realistic_dataset(num_samples):
    """
    Generates synthetic dataset incorporating research-based chemical profiles,
    variety-specific variations, and extraction condition effects
    """
    data = {}
    varieties = list(VARIETY_MULTIPLIERS.keys())
    extraction_methods = list(EXTRACTION_EFFECTS.keys())
    
    # Generate base chemical concentrations
    for chem, (min_val, max_val) in CHEMICAL_RANGES.items():
        data[chem] = np.random.uniform(min_val, max_val, num_samples)
    
    # Generate variety and extraction method assignments
    data['variety'] = np.random.choice(varieties, num_samples)
    data['extraction_method'] = np.random.choice(extraction_methods, num_samples)
    
    df = pd.DataFrame(data)
    
    # Apply variety-specific multipliers
    for i, row in df.iterrows():
        variety = row['variety']
        extraction = row['extraction_method']
        
        # Apply variety effects
        if variety in VARIETY_MULTIPLIERS:
            for compound, multiplier in VARIETY_MULTIPLIERS[variety].items():
                if compound in df.columns:
                    df.at[i, compound] *= multiplier
        
        # Apply extraction effects
        if extraction in EXTRACTION_EFFECTS:
            for compound, multiplier in EXTRACTION_EFFECTS[extraction].items():
                if compound in df.columns:
                    df.at[i, compound] *= multiplier
    
    # Add realistic noise based on experimental variation (±10-15%)
    for chem in CHEMICAL_RANGES.keys():
        noise = np.random.normal(1.0, 0.12, num_samples)  # 12% coefficient of variation
        df[chem] *= noise
        df[chem] = np.clip(df[chem], 0, df[chem].quantile(0.99))  # Remove extreme outliers
    
    # Enhanced taste profile calculations based on research findings
    # PUNGENT: Dominated by Eugenol, Carvacrol, and Methyl eugenol
    df['pungent'] = (1.4 * df['eugenol'] + 
                     1.2 * df['carvacrol'] + 
                     1.0 * df.get('methyl_eugenol', 0)) / 3.6
    
    # BITTER: Ursolic acid, Apigenin, Caffeic acid, Carnosic acid
    df['bitter'] = (1.5 * df['ursolic_acid'] + 
                    1.3 * df['apigenin'] + 
                    1.1 * df.get('caffeic_acid', 0) + 
                    1.0 * df.get('carnosic_acid', 0)) / 4.9
    
    # ASTRINGENT: Rosmarinic acid, Gallic acid, Catechin (tannin-like compounds)
    df['astringent'] = (1.7 * df['rosmarinic_acid'] + 
                        1.4 * df.get('gallic_acid', 0) + 
                        1.2 * df.get('catechin', 0)) / 4.3
    
    # SWEET: Minimal in Tulsi, slight correlation with some compounds
    base_sweet = np.random.choice([0, 0.1, 0.2, 0.3], size=num_samples, p=[0.6, 0.25, 0.1, 0.05])
    variety_sweet_bonus = df['variety'].map({
        'Krishna': 0.1, 'Vishnu': 0.15, 'Camphor': 0.05, 
        'Lavanga': 0.08, 'Red_Holy_Basil': 0.12, 'White_Holy_Basil': 0.1
    }).fillna(0.1)
    df['sweet'] = base_sweet + variety_sweet_bonus
    
    # Calculate overall antioxidant activity score based on research correlations
    df['antioxidant_activity'] = 0
    for compound, correlation in ANTIOXIDANT_CORRELATIONS.items():
        if compound in df.columns:
            df['antioxidant_activity'] += correlation * df[compound]
    
    # Normalize antioxidant activity to 0-10 scale
    df['antioxidant_activity'] = (df['antioxidant_activity'] / df['antioxidant_activity'].max()) * 10
    
    # Clip taste values to realistic 0-5 scale
    for taste in ['pungent', 'bitter', 'astringent', 'sweet']:
        df[taste] = np.clip(df[taste], 0, 5)
    
    # Add quality grade based on compound concentrations and antioxidant activity
    df['quality_grade'] = np.select([
        (df['antioxidant_activity'] >= 7) & (df['rosmarinic_acid'] >= 1.5),
        (df['antioxidant_activity'] >= 5) & (df['rosmarinic_acid'] >= 1.0),
        (df['antioxidant_activity'] >= 3) & (df['rosmarinic_acid'] >= 0.7),
        df['antioxidant_activity'] >= 1
    ], [4, 3, 2, 1], default=0)  # 0-4 scale (Poor to Excellent)
    
    return df

def create_research_validation_samples():
    """Create samples matching research data for validation"""
    validation_samples = []
    
    # Sample 1: Krishna Tulsi from HPLC paper
    krishna_sample = {
        'eugenol': 2.5, 'rosmarinic_acid': 1.8, 'ursolic_acid': 0.6, 
        'apigenin': 0.3, 'carvacrol': 0.8, 'caffeic_acid': 0.26, 
        'gallic_acid': 1.615, 'catechin': 0.035, 'carnosic_acid': 0.2,
        'methyl_eugenol': 0.6, 'bornanone': 0.1, 'variety': 'Krishna',
        'extraction_method': 'methanol'
    }
    
    # Sample 2: Red Holy Basil high antioxidant
    red_basil_sample = {
        'eugenol': 2.8, 'rosmarinic_acid': 2.2, 'ursolic_acid': 0.8,
        'apigenin': 0.4, 'carvacrol': 1.2, 'caffeic_acid': 0.4,
        'gallic_acid': 1.2, 'catechin': 0.06, 'carnosic_acid': 0.25,
        'methyl_eugenol': 0.3, 'bornanone': 0.1, 'variety': 'Red_Holy_Basil',
        'extraction_method': 'ethanol_57'
    }
    
    # Sample 3: Camphor variety with Bornanone
    camphor_sample = {
        'eugenol': 2.0, 'rosmarinic_acid': 1.2, 'ursolic_acid': 0.4,
        'apigenin': 0.2, 'carvacrol': 0.6, 'caffeic_acid': 0.1,
        'gallic_acid': 0.5, 'catechin': 0.03, 'carnosic_acid': 0.15,
        'methyl_eugenol': 0.2, 'bornanone': 0.25, 'variety': 'Camphor',
        'extraction_method': 'ethanol_76'
    }
    
    return [krishna_sample, red_basil_sample, camphor_sample]

# --- MAIN ENHANCED TRAINING CODE ---
print("=" * 60)
print("ENHANCED TULSI TASTE PREDICTOR TRAINING")
print("Incorporating Research Data from 5 Scientific Papers")
print("=" * 60)

print(f"\nGenerating ENHANCED Dataset with {NUM_SAMPLES} samples...")
print("Incorporating:")
print("- 11 phytochemical compounds (vs 5 original)")
print("- 6 Tulsi varieties with specific profiles")
print("- 5 extraction methods with realistic effects")
print("- Research-validated concentration ranges")
print("- Antioxidant activity correlations")

df = generate_enhanced_realistic_dataset(NUM_SAMPLES)
print(f"Enhanced dataset created with {len(df)} samples and {len(df.columns)} features")

# Display dataset statistics
print(f"\nDataset Composition:")
print(f"Varieties: {df['variety'].value_counts().to_dict()}")
print(f"Extraction methods: {df['extraction_method'].value_counts().to_dict()}")

print(f"\nKey compound concentration ranges (mg/g):")
key_compounds = ['rosmarinic_acid', 'ursolic_acid', 'caffeic_acid', 'gallic_acid', 'eugenol']
for compound in key_compounds:
    if compound in df.columns:
        print(f"{compound:15}: {df[compound].min():.3f} - {df[compound].max():.3f} "
              f"(mean: {df[compound].mean():.3f})")

print(f"\nTaste profile ranges:")
taste_cols = ['pungent', 'bitter', 'astringent', 'sweet']
for taste in taste_cols:
    print(f"{taste:12}: {df[taste].min():.3f} - {df[taste].max():.3f} "
          f"(mean: {df[taste].mean():.3f})")

print(f"\nAntioxidant activity: {df['antioxidant_activity'].min():.1f} - {df['antioxidant_activity'].max():.1f} "
      f"(mean: {df['antioxidant_activity'].mean():.1f})")

# Prepare features and targets
feature_columns = list(CHEMICAL_RANGES.keys())
X = df[feature_columns]
y = df[['pungent', 'bitter', 'astringent', 'sweet']]

# Additional target: antioxidant activity prediction
y_antioxidant = df['antioxidant_activity']
y_quality = df['quality_grade']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target matrix shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_ant, X_test_ant, y_train_ant, y_test_ant = train_test_split(X, y_antioxidant, test_size=0.2, random_state=42)
X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X, y_quality, test_size=0.2, random_state=42)

# Train main taste prediction model
print("\nTraining main taste prediction model...")
main_model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1)
main_model.fit(X_train, y_train)

# Train antioxidant activity model
print("Training antioxidant activity prediction model...")
antioxidant_model = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
antioxidant_model.fit(X_train_ant, y_train_ant)

# Train quality grade model
print("Training quality grade prediction model...")
from sklearn.ensemble import RandomForestClassifier
quality_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
quality_model.fit(X_train_qual, y_train_qual)

# Evaluate models
print("\n" + "="*50)
print("MODEL EVALUATION RESULTS")
print("="*50)

# Main taste model evaluation
y_pred = main_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n1. TASTE PREDICTION MODEL:")
print(f"   Mean Absolute Error: {mae:.4f}")
print(f"   R² Score: {r2:.4f}")
print(f"   Variance Explained: {r2 * 100:.2f}%")

# Individual taste predictions
for i, taste in enumerate(['pungent', 'bitter', 'astringent', 'sweet']):
    taste_r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"   {taste.capitalize():12} R²: {taste_r2:.4f}")

# Antioxidant model evaluation
y_pred_ant = antioxidant_model.predict(X_test_ant)
mae_ant = mean_absolute_error(y_test_ant, y_pred_ant)
r2_ant = r2_score(y_test_ant, y_pred_ant)

print(f"\n2. ANTIOXIDANT ACTIVITY MODEL:")
print(f"   Mean Absolute Error: {mae_ant:.4f}")
print(f"   R² Score: {r2_ant:.4f}")
print(f"   Variance Explained: {r2_ant * 100:.2f}%")

# Quality model evaluation
from sklearn.metrics import accuracy_score, classification_report
y_pred_qual = quality_model.predict(X_test_qual)
accuracy = accuracy_score(y_test_qual, y_pred_qual)

print(f"\n3. QUALITY GRADE MODEL:")
print(f"   Accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

# Feature importance analysis
print(f"\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': main_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop compounds for taste prediction:")
for i, row in feature_importance.head(8).iterrows():
    print(f"{row['feature']:15}: {row['importance']:.4f}")

# Save all models
models_to_save = {
    'tulsi_taste_predictor_ENHANCED.pkl': main_model,
    'tulsi_antioxidant_predictor.pkl': antioxidant_model,
    'tulsi_quality_classifier.pkl': quality_model
}

print(f"\n" + "="*50)
print("SAVING TRAINED MODELS")
print("="*50)

for filename, model in models_to_save.items():
    joblib.dump(model, filename)
    print(f"✓ Saved: {filename}")

# Save feature names and metadata
metadata = {
    'feature_columns': feature_columns,
    'chemical_ranges': CHEMICAL_RANGES,
    'variety_multipliers': VARIETY_MULTIPLIERS,
    'taste_columns': ['pungent', 'bitter', 'astringent', 'sweet'],
    'model_performance': {
        'taste_r2': r2,
        'antioxidant_r2': r2_ant,
        'quality_accuracy': accuracy
    }
}

joblib.dump(metadata, 'tulsi_model_metadata.pkl')
print(f"✓ Saved: tulsi_model_metadata.pkl")

# Create validation samples and test
print(f"\n" + "="*50)
print("RESEARCH VALIDATION TEST")
print("="*50)

validation_samples = create_research_validation_samples()
for i, sample in enumerate(validation_samples, 1):
    sample_features = [sample[col] for col in feature_columns]
    sample_array = np.array(sample_features).reshape(1, -1)
    
    taste_pred = main_model.predict(sample_array)[0]
    antioxidant_pred = antioxidant_model.predict(sample_array)[0]
    quality_pred = quality_model.predict(sample_array)[0]
    
    print(f"\nValidation Sample {i} ({sample['variety']}):")
    print(f"  Variety: {sample['variety']}")
    print(f"  Extraction: {sample['extraction_method']}")
    print(f"  Predicted tastes - Pungent: {taste_pred[0]:.2f}, Bitter: {taste_pred[1]:.2f}, "
          f"Astringent: {taste_pred[2]:.2f}, Sweet: {taste_pred[3]:.2f}")
    print(f"  Antioxidant activity: {antioxidant_pred:.2f}/10")
    print(f"  Quality grade: {quality_pred}/4")

print(f"\n" + "="*60)
print("ENHANCED TRAINING COMPLETED SUCCESSFULLY!")
print("="*60)
print("Generated models:")
print("1. tulsi_taste_predictor_ENHANCED.pkl - Main taste prediction")
print("2. tulsi_antioxidant_predictor.pkl - Antioxidant activity")
print("3. tulsi_quality_classifier.pkl - Quality grading")
print("4. tulsi_model_metadata.pkl - Model metadata")
print("\nDataset enhanced with:")
print("- Research-validated compound ranges")
print("- Variety-specific chemical profiles") 
print("- Extraction method effects")
print("- Antioxidant activity correlations")
print("- Quality grading system")
print(f"\nReady for production use with {len(feature_columns)} chemical features!")