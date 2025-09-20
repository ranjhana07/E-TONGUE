
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Set to True to use simulated sensor values, False for real sensors
USE_SIMULATION = True

# --- CONFIGURATION BASED ON YOUR EXCEL DATA ---
CHEMICAL_RANGES = {
    'eugenol': (1.8, 3.2),
    'rosmarinic_acid': (0.5, 1.0),
    'ursolic_acid': (0.2, 0.7),
    'apigenin': (0.1, 0.4),
    'carvacrol': (0.2, 1.8)
}

CONFIG = {
    'model_load_path': 'tulsi_taste_predictor_REALISTIC.pkl',
    'taste_threshold': 1.8,
    'noticeable_threshold': 0.8,
    # REVISED POTENCY THRESHOLDS for Indian Market
    'high_potency_threshold': 2.2,  # Lowered from 2.5. A score of 2.2/5 is very good for common Tulsi.
    'low_potency_threshold': 1.6,   # Lowered from 1.8. This is the baseline for "Good".
    'optimal_ph_center': 6.0,
    'tds_reference': 1000,
    # REVISED MINIMUM ACCEPTABLE LEVELS for Indian Market
    'minimum_acceptable': {
        'eugenol': 1.9,        # Slightly lowered. Many decent samples hover around 2.0%.
        'rosmarinic_acid': 0.5, # Significantly lowered from 0.6%. This is a common point of failure for the old threshold.
        'ursolic_acid': 0.3,    # Lowered from 0.4%. Many acceptable samples are in the 0.3-0.4% range.
        'carvacrol': 0.7        
    }
}

# ---------------- SENSOR INTEGRATION SECTION ----------------

def read_ph_sensor():
    if USE_SIMULATION:
        # Simulate sensor voltage output (e.g. 2.45V for demo)
        return 2.45
    else:
        # TODO: Replace with actual sensor code (serial, USB, API, etc)
        raise NotImplementedError("Real pH sensor reading not implemented.")


def read_tds_sensor():
    if USE_SIMULATION:
        # Simulate EC reading in μS/cm (e.g. 1550μS/cm for demo)
        return 1550
    else:
        # TODO: Replace with actual sensor code (serial, USB, API, etc)
        raise NotImplementedError("Real TDS sensor reading not implemented.")

def calibrate_ph(sensor_output):
    # Example calibration (real devices will have different curves)
    # pH = 7.0 - ((sensor_output - 2.5) * 3.5)
    return 7.0 - ((sensor_output - 2.5) * 3.5)

def calibrate_tds(sensor_output_ec):
    # Typical conversion factor for TDS meters: 0.64
    return sensor_output_ec * 0.64

def get_physicochemical_input_from_sensors():
    ph_sensor = read_ph_sensor()
    tds_sensor = read_tds_sensor()
    ph_value = calibrate_ph(ph_sensor)
    tds_value = calibrate_tds(tds_sensor)
    print(f"\nSensor Output: pH={ph_sensor:.2f}V, TDS={tds_sensor}μS/cm")
    print(f"Calibrated: pH={ph_value:.2f} | TDS={tds_value:.1f} ppm")
    return ph_value, tds_value

# ---------------- CHEMICAL DATA GENERATION SECTION ----------------
def generate_realistic_sample():
    """Generate a new random Tulsi sample with realistic concentrations"""
    return pd.DataFrame({
        'eugenol': [round(np.random.uniform(CHEMICAL_RANGES['eugenol'][0], CHEMICAL_RANGES['eugenol'][1]), 2)],
        'rosmarinic_acid': [round(np.random.uniform(CHEMICAL_RANGES['rosmarinic_acid'][0], CHEMICAL_RANGES['rosmarinic_acid'][1]), 2)],
        'ursolic_acid': [round(np.random.uniform(CHEMICAL_RANGES['ursolic_acid'][0], CHEMICAL_RANGES['ursolic_acid'][1]), 2)],
        'apigenin': [round(np.random.uniform(CHEMICAL_RANGES['apigenin'][0], CHEMICAL_RANGES['apigenin'][1]), 2)],
        'carvacrol': [round(np.random.uniform(CHEMICAL_RANGES['carvacrol'][0], CHEMICAL_RANGES['carvacrol'][1]), 2)]
    })

# ---------------- GRADING AND INTERPRETATION SECTION ----------------
def predict_and_interpret(model, new_sample, taste_names, taste_threshold, noticeable_threshold, ph_factor, tds_factor):
    predicted_taste = model.predict(new_sample)[0] * ph_factor * tds_factor
    print("\nAdjusted Taste Profile for Sensor-Corrected Sample:")
    print("---------------------------------------")
    for name, value in zip(taste_names, predicted_taste):
        print(f"{name}: {value:.2f}/5")
    dominant_taste_index = np.argmax(predicted_taste)
    dominant_taste_name = taste_names[dominant_taste_index]
    dominant_taste_score = predicted_taste[dominant_taste_index]
    if dominant_taste_score >= taste_threshold:
        print(f"-> Dominant Taste: {dominant_taste_name} ({dominant_taste_score:.2f}/5)")
    else:
        if all(score < 1.0 for score in predicted_taste):
            print("-> Overall Impression: Bland / Neutral")
        else:
            print("-> Overall Impression: Balanced profile, no single dominant taste.")
    noticeable_tastes = [name for name, value in zip(taste_names, predicted_taste) if value >= noticeable_threshold]
    if noticeable_tastes:
        print(f"-> Noticeable Tastes: {', '.join(noticeable_tastes)}")
    else:
        print("-> No strongly noticeable tastes.")
    return predicted_taste, dominant_taste_name, dominant_taste_score

def assess_potency_quality(predicted_taste, chemical_data, taste_names):
    print("\n" + "="*50)
    print("COMPREHENSIVE QUALITY ASSESSMENT:")
    print("="*50)

    compound_assessment = {
        'Pungent': {
            'compound': 'Eugenol', 
            'value': chemical_data['eugenol'][0],
            'min_acceptable': CONFIG['minimum_acceptable']['eugenol'],
            'importance': 'Primary antimicrobial compound'
        },
        'Bitter': {
            'compound': 'Ursolic Acid', 
            'value': chemical_data['ursolic_acid'][0],
            'min_acceptable': CONFIG['minimum_acceptable']['ursolic_acid'],
            'importance': 'Antioxidant properties'
        },
        'Astringent': {
            'compound': 'Rosmarinic Acid', 
            'value': chemical_data['rosmarinic_acid'][0],
            'min_acceptable': CONFIG['minimum_acceptable']['rosmarinic_acid'],
            'importance': 'Antiviral properties'
        }
    }

    print("KEY COMPOUND ANALYSIS:")
    print("-" * 30)
    acceptable_compounds = 0
    total_compounds = 0

    for taste, info in compound_assessment.items():
        taste_score = predicted_taste[taste_names.index(taste)]
        is_acceptable = info['value'] >= info['min_acceptable']
        status = "✓" if is_acceptable else "⚠"
        print(f"{status} {info['compound']}: {info['value']:.2f}% (Min: {info['min_acceptable']}%)")
        if is_acceptable:
            acceptable_compounds += 1
        total_compounds += 1

    acceptability_percentage = (acceptable_compounds / total_compounds) * 100
    noticeable_indices = [i for i, score in enumerate(predicted_taste) if score > CONFIG['noticeable_threshold'] and taste_names[i] != 'Sweet']

    if noticeable_indices:
        overall_quality = np.mean([predicted_taste[i] for i in noticeable_indices])
        print(f"\nOVERALL QUALITY SCORE: {overall_quality:.2f}/5")
        print(f"COMPOUND ACCEPTABILITY: {acceptability_percentage:.0f}% ({acceptable_compounds}/{total_compounds})")
        if overall_quality >= CONFIG['high_potency_threshold'] and acceptability_percentage >= 80:
            print("→ OVERALL QUALITY: EXCELLENT ★★★★")
            print("→ Premium quality - Suitable for therapeutic use")
        elif overall_quality >= CONFIG['low_potency_threshold'] and acceptability_percentage >= 60:
            print("→ OVERALL QUALITY: GOOD ★★★☆")
            print("→ Good quality - Suitable for general medicinal use")
        elif acceptability_percentage >= 40:
            print("→ OVERALL QUALITY: ACCEPTABLE ★★☆☆")
            print("→ Acceptable quality - Suitable for daily wellness")
        else:
            print("→ OVERALL QUALITY: POOR ★☆☆☆")
            print("→ Substandard - Not recommended for regular use")
    else:
        print("→ OVERALL QUALITY: POOR ★☆☆☆")
        print("→ Very low potency - may be stale, old, or poorly stored")

    print("\nDETAILED ASSESSMENT:")
    print("-" * 20)
    for taste, info in compound_assessment.items():
        if info['value'] < info['min_acceptable']:
            print(f"→ {info['compound']}: Below minimum acceptable level ({info['value']:.2f}% < {info['min_acceptable']}%)")
        else:
            print(f"→ {info['compound']}: Acceptable level ({info['value']:.2f}%)")

    if acceptability_percentage >= 80 and overall_quality >= 2.5:
        print(f"\nRECOMMENDATION: Therapeutic use (Premium quality)")
    elif acceptability_percentage >= 60:
        print(f"\nRECOMMENDATION: General medicinal use (Good quality)")
    elif acceptability_percentage >= 40:
        print(f"\nRECOMMENDATION: Daily wellness (Acceptable quality)")
    else:
        print(f"\nRECOMMENDATION: Not recommended for regular use")

def main():
    print("TULSI TASTE PREDICTION & QUALITY AUDITOR")
    print("========================================")
    print("Sensor-Integrated Query Mode\n")

    try:
        if not Path(CONFIG['model_load_path']).exists():
            print(f"❌ Error: Trained model not found at '{CONFIG['model_load_path']}'")
            print("Please train the model first with 'generate_realistic_dataset()'")
            return

        print("Loading pre-trained model...")
        model = joblib.load(CONFIG['model_load_path'])
        print("✓ Model loaded successfully!")
        taste_names = ['Pungent', 'Bitter', 'Astringent', 'Sweet']

        while True:
            user_input = input("\nAnalyze new sample with sensor data? (y/n/q): ").lower().strip()
            if user_input == 'q':
                print("Exiting quality auditor. Goodbye!")
                break
            elif user_input in ['y', 'yes', '']:
                print("\nSensor input activated. Insert sensors and calibrate if needed.")
                ph_value, tds_value = get_physicochemical_input_from_sensors()

                # Generate a random realistic tulsi sample
                new_sample = generate_realistic_sample()
                print("\n" + "=" * 50)
                print("SAMPLE CHEMICAL COMPOSITION (Based on Experimental Ranges):")
                print("=" * 50)
                for col in new_sample.columns:
                    print(f"{col:15}: {new_sample[col].values[0]:.2f}%")

                # Calculate adjustment factors
                ph_penalty = abs(ph_value - CONFIG['optimal_ph_center'])
                ph_factor = max(0.5, 1 - (ph_penalty / 3))
                clamped_tds = max(400, min(tds_value, 1600))
                tds_factor = clamped_tds / CONFIG['tds_reference']

                # Predict, interpret, AND assess quality
                taste_predictions, dominant_name, dominant_score = predict_and_interpret(
                    model, new_sample, taste_names,
                    CONFIG['taste_threshold'], CONFIG['noticeable_threshold'],
                    ph_factor, tds_factor
                )
                assess_potency_quality(taste_predictions, new_sample, taste_names)

                print("\n" + "=" * 50)
            elif user_input in ['n', 'no']:
                continue
            else:
                print("Please enter 'y' for yes, 'n' for no, or 'q' to quit.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
