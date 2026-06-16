import os
import pandas as pd
from pathlib import Path

def analyser_dataset(chemin_dataset):
    """
    Parcourt la structure de dossiers et extrait les informations des patients.
    
    Args:
        chemin_dataset: Chemin vers le dossier racine contenant 01_Normal et 02_Abnormal
    """
    donnees = []
    
    # Parcourir les deux dossiers principaux
    for dossier_label in ['01_Normal', '02_Abnormal']:
        chemin_label = os.path.join(chemin_dataset, dossier_label)
        
        if not os.path.exists(chemin_label):
            print(f"Attention: {chemin_label} n'existe pas")
            continue
        
        # Déterminer le label actuel
        label_actuel = 'Normal' if dossier_label == '01_Normal' else 'Anormal'
        
        # Parcourir tous les sous-dossiers des patients
        for patient_dir in os.listdir(chemin_label):
            chemin_patient = os.path.join(chemin_label, patient_dir)
            
            # Vérifier que c'est bien un dossier et qu'il commence par "buth_"
            if os.path.isdir(chemin_patient) and patient_dir.startswith('buth_'):
                
                # Chemin du fichier texte
                fichier_texte = os.path.join(chemin_patient, f"{patient_dir}.txt")
                
                # Lire le commentaire s'il existe
                commentaire = ""
                if os.path.exists(fichier_texte):
                    try:
                        with open(fichier_texte, 'r', encoding='utf-8') as f:
                            commentaire = f.read().strip()
                    except Exception as e:
                        commentaire = f"Erreur de lecture: {e}"
                else:
                    commentaire = "Aucun commentaire"
                
                # Ajouter les données
                donnees.append({
                    'Patient': patient_dir,
                    'Label Actuel': label_actuel,
                    'Diagnostic Clinique (commentaire)': commentaire
                })
    
    return donnees

def sauvegarder_resultats(donnees, fichier_sortie='resultats_patients.xlsx'):
    """
    Sauvegarde les résultats dans un fichier Excel et CSV.
    """
    df = pd.DataFrame(donnees)
    
    # Sauvegarde en Excel
    df.to_excel(fichier_sortie, index=False)
    print(f"✓ Résultats sauvegardés dans {fichier_sortie}")
    
    # Sauvegarde aussi en CSV pour compatibilité
    fichier_csv = fichier_sortie.replace('.xlsx', '.csv')
    df.to_csv(fichier_csv, index=False, encoding='utf-8-sig')
    print(f"✓ Résultats sauvegardés dans {fichier_csv}")
    
    return df

def afficher_resume(df):
    """
    Affiche un résumé statistique.
    """
    print("\n" + "="*50)
    print("RÉSUMÉ DE L'ANALYSE")
    print("="*50)
    print(f"Nombre total de patients: {len(df)}")
    print(f"Patients normaux: {len(df[df['Label Actuel'] == 'Normal'])}")
    print(f"Patients anormaux: {len(df[df['Label Actuel'] == 'Anormal'])}")
    print(f"Patients avec commentaire: {len(df[df['Diagnostic Clinique (commentaire)'] != 'Aucun commentaire'])}")
    print(f"Patients sans commentaire: {len(df[df['Diagnostic Clinique (commentaire)'] == 'Aucun commentaire'])}")
    print("="*50)

# Utilisation du script
if __name__ == "__main__":
    # Spécifiez le chemin vers votre dataset
    #chemin_dataset = input("Entrez le chemin vers votre dataset: ").strip()
    
    # Chemin par défaut : dataset dans le nouveau pipeline
    chemin_dataset = "/workspace/dataset"
    
    if os.path.exists(chemin_dataset):
        print("Analyse en cours...")
        donnees = analyser_dataset(chemin_dataset)
        
        if donnees:
            df = sauvegarder_resultats(donnees)
            afficher_resume(df)
            
            # Afficher un aperçu des 5 premiers patients
            print("\nAPERÇU DES 5 PREMIERS PATIENTS:")
            print(df.head().to_string())
        else:
            print("Aucune donnée trouvée dans la structure spécifiée.")
    else:
        print(f"Le chemin {chemin_dataset} n'existe pas.")