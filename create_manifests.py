import pandas as pd
import os

# Load the master manifest (which has image paths)
master_manifest_path = 'outputs/manifeste.csv'
if not os.path.exists(master_manifest_path):
    print(f"Error: {master_manifest_path} not found. Run prepare_dataset.py first.")
    exit(1)

df_master = pd.read_csv(master_manifest_path)

# Load clinical data for comments
clinical_path = 'resultats_patients.csv'
df_clinical = pd.read_csv(clinical_path)
df_clinical.columns = [c.strip() for c in df_clinical.columns]

# Create a mapping from Patient to Clinical Comments
clinical_dict = df_clinical.set_index('Patient')['Diagnostic Clinique (commentaire)'].to_dict()

def normalize_diag(diag):
    if pd.isna(diag):
        return ""
    return " ".join(str(diag).upper().split())

# 1. Manifeste Original
df_master.to_csv('manifeste_original.csv', index=False)
print(f"1. manifeste_original.csv created: {len(df_master)} rows")

# 2. Manifeste Corrige (ASC-US -> Anormal + errors)
def get_corrected_label(row):
    pid = str(row['patient_id']).strip()
    label = row['label'] # 0 or 1
    
    diag_raw = clinical_dict.get(pid, "")
    diag = normalize_diag(diag_raw)

    # SPECIAL CASE: buth_208, 242, 329 are visually NORMAL NILM (user feedback)
    if pid in ['buth_208', 'buth_242', 'buth_329']:
        return 0
    
    # Explicit NILM/Atrophy errors in Anormal (from analyse_dataset.txt)
    if pid == 'buth_191': # NILM
        return 0
    if pid == 'buth_340': # ATROPHIC SMEAR
        return 0

    # ASC-US is considered Abnormal for screening purposes
    if "ASCUS" in diag or "ASC-US" in diag:
        return 1

    # Additional HSIL catches (shouldn't hit our 3 special cases now)
    if label == 0 and ("HSIL" in diag or "CIN II" in diag or "CIN III" in diag):
        return 1
    
    return label

df_corrige = df_master.copy()
df_corrige['label'] = df_corrige.apply(get_corrected_label, axis=1)
df_corrige.to_csv('manifeste_corrige.csv', index=False)
print(f"2. manifeste_corrige.csv created: {len(df_corrige)} rows")
print(f"   Counts: {df_corrige['label'].value_counts().to_dict()}")

# 3. Manifeste Strict (Exclut ASC-US, Atrophie, et Inadequate)
def is_ambiguous(row):
    pid = str(row['patient_id']).strip()
    diag_raw = clinical_dict.get(pid, "")
    diag = normalize_diag(diag_raw)

    # These 3 cases have junk text but are visually NORMAL. Keep them.
    if pid in ['buth_208', 'buth_242', 'buth_329']:
        return False

    # Exclude technical failures
    if any(term in diag for term in ["NO SLIDE", "INADEQUATE", "POOR SLIDE"]):
        return True
    
    # Exclude ASC-US (ambiguous)
    if "ASCUS" in diag or "ASC-US" in diag:
        return True
    
    # Exclude Atrophy (mimic)
    if "ATROPHIC" in diag or "ATROPHIE" in diag:
        return True

    return False

df_strict = df_master[~df_master.apply(is_ambiguous, axis=1)].copy()
df_strict['label'] = df_strict.apply(get_corrected_label, axis=1)
df_strict.to_csv('manifeste_strict.csv', index=False)
print(f"3. manifeste_strict.csv created: {len(df_strict)} rows")
print(f"   Counts: {df_strict['label'].value_counts().to_dict()}")
