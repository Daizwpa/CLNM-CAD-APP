import pandas as pd
import pyreadstat
import os
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import json
from easydict import EasyDict as edict

def load_sav_to_dataframe(file_path):
    """
    Load a .sav file and return its content as a pandas DataFrame with metadata.

    Parameters:
        file_path (str): The path to the .sav file.

    Returns:
        pd.DataFrame: The content of the .sav file as a pandas DataFrame.
        metadata: Metadata about the .sav file.

    Raises:
        ValueError: If the file does not have a .sav extension.
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.endswith('.sav'):
        raise ValueError("The file must have a .sav extension.")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    
    # Load the .sav file using pyreadstat
    df, meta = pyreadstat.read_sav(file_path)
    df = __prepare_data(df)
    return df, meta


def __prepare_data(data):

    data["DOSE_CUMULEE_IODE"] = data["DOSE_CUMULEE_IODE"].fillna(0)
    data["DOSE_CUMULEE_IODE"] = pd.Categorical(data["DOSE_CUMULEE_IODE"],ordered=True)
    data["DOSE_CUMULEE_IODE"] = data["DOSE_CUMULEE_IODE"].cat.rename_categories({
        0:0,
        1:30,
        2:50,
        3:100,
        4:150,
        5:200,
        6:250, 
        7:300,
        160:160,
        300:300,
        320:320,
        330:330,
        350:350,
        400:400,
        450:450
    })
    data["DOSE_CUMULEE_IODE"] = pd.cut(data["DOSE_CUMULEE_IODE"], bins=[-np.inf, 0, 30, 150, + np.inf], labels=["0","≤30", "30-150",">150"])
   #data["DOSE_CUMULEE_IODE"]  = data["DOSE_CUMULEE_IODE"].astype("int")
    
    data["INTERVAL_DERNIERE_TSH"] = data["INTERVAL_DERNIERE_TSH"].astype("Int64")
    data["INTERVAL_DERNIERE_TSH"] = pd.Categorical(data["INTERVAL_DERNIERE_TSH"], ordered=True)
    data["INTERVAL_DERNIERE_TSH"] = data["INTERVAL_DERNIERE_TSH"].cat.rename_categories({
        1: '<0.1',
        2: '0.1-0.5',
        3: '0.5-2',
        4: '>2'
    })
    
    data["Interval_Derniere_TSH"] = pd.cut(data["DERNIERE_TSH"], bins=[-np.inf, 0.1, 0.5, 2, np.inf], labels=['<0.1', '0.1-0.5', '0.5-2', '>2'])
    data["Interval_AVANT_DERNIERE_TSH"] = pd.cut(data["AVANT_DERNIERE_TSH"], bins=[-np.inf, 0.1, 0.5, 2, np.inf], labels=['<0.1', '0.1-0.5', '0.5-2', '>2'])

    data["Interval_DOSE_LEVOTHYROX"] = pd.cut(data["DOSE_LEVOTHYROX"], bins=[0, 101, 201, +np.inf], labels=["100≥", "100-200", ">200"], ordered=True)

    data["REVENU_ANNUEL"]  = pd.Categorical(data["REVENU_ANNUEL"], ordered=True)
    data["REVENU_ANNUEL"] = data["REVENU_ANNUEL"].cat.rename_categories({
        1: '< 150 000',
        2: '1150 000 - 250 000',
        3: '2250 000 - 500 000',
        4: '3500 000 -700 000',
        5: '>700000 DA',
        6: 'Refus',
        7: 'NSP'
    })
    
    
    data.rename({"SEXE2": "SEXE"},  axis=1, inplace=True)
    data["SEXE"].name
    data["SEXE"] = data["SEXE"].astype("category")
    data["SEXE"] = data["SEXE"].cat.rename_categories({
        1: "Femme",
        2: "Homme"
    })
    
    
    data["STATUT_MATRIMONIAL"] = data["STATUT_MATRIMONIAL"].astype("category")
    data["STATUT_MATRIMONIAL"] = data["STATUT_MATRIMONIAL"].cat.rename_categories({
        1: 'Marié',
        2: 'Célibataire',
        3: 'Divorcé',
        4: 'Veuf',
        5: 'Séparé'
    })
    
    
    data.loc[data["ACTIVITE_POFESSIONNELLE"] == 1,"ACTIVITE_POFESSIONNELLE"] =2 
    data["ACTIVITE_POFESSIONNELLE"] = data["ACTIVITE_POFESSIONNELLE"].astype("category")
    data["ACTIVITE_POFESSIONNELLE"] = data["ACTIVITE_POFESSIONNELLE"].cat.rename_categories({
        0: 'Refuse',
        2: 'Employé',
        3: 'Indépendant', 
        4: 'Bénévole',
        5: 'Étudiant',
        6: 'Maître (sse) de maison',
        7: 'Retraité(e)',
        8: 'Chômeur (se)',
        9: 'Invalide'
    })

    
    data["TYPE_HISTOLOGIQUE"] = data["TYPE_HISTOLOGIQUE"].astype("category")
    data["TYPE_HISTOLOGIQUE"] = data["TYPE_HISTOLOGIQUE"].cat.rename_categories({
        1: 'NIFT',
        2: 'Tumeur vesiculaire à potentiel de malignité incertain',
        3: 'Papillaire',
        4: 'Vésiculaire',
        5: 'Peu différencié',
        6: 'Anaplasique',
        7: 'Medullaire'
    })
    data["Papillaire"] = np.where(data["TYPE_HISTOLOGIQUE"]=="Papillaire", "Yes", "No")
    data["Papillaire"] = data["Papillaire"].astype("category")
    
    data["Medullaire"] = np.where(data["TYPE_HISTOLOGIQUE"]=="Medullaire", "Yes", "No")
    data["Medullaire"] = data["Medullaire"].astype("category")
    
    data["Vésiculaire"] = np.where(data["TYPE_HISTOLOGIQUE"]=="Vésiculaire", "Yes", "No")
    data["Vésiculaire"] = data["Vésiculaire"].astype("category")



    data.loc[data["T"]==0, "T"] = np.nan
    data["T"] = pd.Categorical(data["T"], ordered=True)
    data["T"] = data["T"].cat.rename_categories({
        1: 'T1',
        2: 'T2',
        3: 'T3',
        4: 'T4'
    })
    data["T"]
    
    data["N"] = pd.Categorical(data["N"], ordered=True)
    data["N"] = data["N"].cat.rename_categories({
        0: 'N0/Nx',
        1: 'N1'
    })


    data["M"] = pd.Categorical(data["M"], ordered=True)
    data["M"] = data["M"].cat.rename_categories({
        0: 'Mx/M0',
        1: 'M1'
    })
    
    
    data["INVASION_VASCULAIRE"] = data["INVASION_VASCULAIRE"].astype("category")
    data["INVASION_VASCULAIRE"] = data["INVASION_VASCULAIRE"].cat.rename_categories({
        0: "Absence d'emboles vasculaires",
        1: "Présence d'emboles vasculaires"
    })
    
    
    data["ETE"] = data["ETE"].astype("category")
    data["ETE"] = data["ETE"].cat.rename_categories({
        0: 'Non',
        1: 'Minime',
        3: 'grosse',
        4: 'Non précisé',
        5: 'Présente non spécifiée'
    })
    data["b_ETE"] = np.where(data["ETE"] == "Non", 0, 1)
    
    
    data["MULTIFOCALITE"] = data["MULTIFOCALITE"].astype("category")
    data["MULTIFOCALITE"] = data["MULTIFOCALITE"].cat.rename_categories({
        0: 'Non',
        1: 'Oui'
    })

    data.loc[data["RISK_AJCC8"]==7, "RISK_AJCC8"] = np.nan
    data["RISK_AJCC8"] = pd.Categorical(data["RISK_AJCC8"], ordered=True)
    data["RISK_AJCC8"] = data["RISK_AJCC8"].cat.rename_categories({
        1: 'Stade I',
        2: 'Stade II',
        3: 'Stade III',
        4: 'Stade IVA',
        5: 'Stade IVB',
        6: 'Stade IVC',
    })
    
    data.loc[data["AJCC8"]==5, "AJCC8"] = np.nan
    data["AJCC8"] = pd.Categorical(data["AJCC8"], ordered=True)
    data["AJCC8"] = data["AJCC8"].cat.rename_categories({
        1: 'Stade I',
        2: 'Stade II',
        3: 'Stade III',
        4: 'Stade IV',

    })

    data["RISQUE_RECIDIVE_ATA"] = pd.Categorical(data["RISQUE_RECIDIVE_ATA"], ordered=True)
    data["RISQUE_RECIDIVE_ATA"] = data["RISQUE_RECIDIVE_ATA"].cat.rename_categories({
        1: 'Faible',
        2: 'Intermédiaire',
        3: 'Elevé'
    })


    data["NIVEAU_INSTRUC"] = data["NIVEAU_INSTRUC"].astype("category")
    data["NIVEAU_INSTRUC"] = data["NIVEAU_INSTRUC"].cat.rename_categories({
        1: 'Instruit',
        2: 'Analphabète'
    })


    data["RISK_DYNAMIQ"] = data["RISK_DYNAMIQ"].astype("category")
    data["RISK_DYNAMIQ"] = data["RISK_DYNAMIQ"].cat.rename_categories({
        1: 'Excellente réponse',
        2: 'Réponse indeterminée',
        3: 'Réponse biologique incomplète',
        4: 'Réponse radiologique incomplète',
        5: 'Inconnu ou données insuffisantes',
        6: 'En cours'
    })


    data["MALADIE_CV_CONNUE"] = data["MALADIE_CV_CONNUE"].astype("category")
    data["MALADIE_CV_CONNUE"] = data["MALADIE_CV_CONNUE"].cat.rename_categories({
        0: 'Non',
        1: 'cardiopathie ischémique',
        2: 'Insuffisance cardiaque',
        3: 'Maladie rythmique',
        4: 'AVC',
        5: 'AOMI',
        6: 'Maladie rythmique+ IC',
        7: 'TVP'
    })
    data["MALADIE_CV"] = np.where(data["MALADIE_CV_CONNUE"] == "Non", "Non", "Oui")
    data["MALADIE_CV"] = data["MALADIE_CV"].astype("category")

    data["META"] = data["META"].astype("category")
    data["META"] = data["META"].cat.rename_categories({
        0: 'Non',
        1: 'Oui'
    })


    data["SECURITE_SOCIALE"] = data["SECURITE_SOCIALE"].astype("category")
    data["SECURITE_SOCIALE"] = data["SECURITE_SOCIALE"].cat.rename_categories({
        0: 'Non',
        1: 'Oui',
        2: 'Ne sait pas'
    })


    data["TABAC_STAT"] = data["TABAC_STAT"].astype("category")
    data["TABAC_STAT"] = data["TABAC_STAT"].cat.rename_categories({
        0: 'Jamais',
        1: 'Actif',
        2: 'Ancien'
    })
    data["Tabagisme"] = np.where(data["TABAC_STAT"]=="Jamais", "Non", "Oui")
    data["Tabagisme"] = data["Tabagisme"].astype("category")


    data["ALCOOL_STATUS"] = data["ALCOOL_STATUS"].astype("category")
    data["ALCOOL_STATUS"] = data["ALCOOL_STATUS"].cat.rename_categories({
        0: 'Jamais',
        1: 'Actif',
        2: 'Ancien'
    })
    data["Alcoolisme"] = np.where(data["ALCOOL_STATUS"]=="Jamais", "Non", "Oui")
    data["Alcoolisme"] = data["Alcoolisme"].astype("category")

    data["MCV_FAM"] = data["MCV_FAM"].astype("category")
    data["MCV_FAM"] = data["MCV_FAM"].cat.rename_categories({
        1: 'Oui',
        2: 'Non'
    })


    data["ATCD_PER_KC"] = data["ATCD_PER_KC"].astype("category")
    data["ATCD_PER_KC"] = data["ATCD_PER_KC"].cat.rename_categories({
        0: 'Non',
        1: 'OUi'
    })


    data["ACTIVITE_VIGOUREUSE"] = data["ACTIVITE_VIGOUREUSE"].astype("category")
    data["ACTIVITE_VIGOUREUSE"] = data["ACTIVITE_VIGOUREUSE"].cat.rename_categories({
        0: 'Non',
        1: 'OUi'
    })


    data["YEAR_CHIRURGIE"] = data["YEAR_CHIRURGIE"].astype("category")
    data["YEAR_CHIRURGIE"] = data["YEAR_CHIRURGIE"].cat.rename_categories({
        1: '<2000',
        2: '2000-2009',
        3: '2010-2019',
        4: '≥2020'
    })


    data["PARTS_F_L_PLUS_5"] = data["PARTS_F_L_PLUS_5"].astype("category")
    data["PARTS_F_L_PLUS_5"] = data["PARTS_F_L_PLUS_5"].cat.rename_categories({
        0: '<5 Portions',
        1: '≥5 Portions'
    })


    data.drop([
        "DIAGN_AGE_LESS_4O",
        "TRANCHE_AGE_DIAGNOSTIC",
        "DOSE_CUMULEE_IODE1",
        "TRANCHE_AGE",
        "PARTS_F_L_PLUS_5",
        "ACTIVITE_VIGOUREUSE",
        "NIVEAU_INSTRUC",
        "SECURITE_SOCIALE",
        "YEAR_CHIRURGIE",
        "REVENU_ANNUEL",
        "ACTIVITE_POFESSIONNELLE",
        "STATUT_MATRIMONIAL",
        "DUREE_EVOLUTION"
    
    ], axis=1, inplace=True)

    
    return data



def CalculatePerformance(y_true, y_predicted, proba):
    try:
        matrics = [accuracy_score, recall_score, precision_score, f1_score]
        dic = {}
        for matric in matrics:
            output = round(matric(y_true=y_true, y_pred=y_predicted), 2)
            dic[matric.__name__.replace("_score", "").title()] = output
        output = float(round(roc_auc_score(y_true=y_true, y_score=proba), 2))
        dic[roc_auc_score.__name__.replace("_score", "").title()] = output
        return dic
    except:
        raise


def calculate_Iqr_Upper_bound_lower_bound(data:pd.Series):
    # finding the 1st quartile
    q1 = data.quantile(0.25)
    
    # finding the 3rd quartile
    q3 = data.quantile(0.75)
    
    # finding the iqr region
    iqr = q3-q1
    
    # finding upper and lower whiskers
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)
    return iqr, upper_bound, lower_bound


def clip_dataset(data, columns):
    for column in columns:
        print("Column:", column)
        iqr, upper_bound, lower_bound = calculate_Iqr_Upper_bound_lower_bound(data[column])
        print(f"iqr:{iqr}, upper bound:{upper_bound}, lower bound:{lower_bound}")
        data_without_outliers =data.loc[(data[column]<=upper_bound) & (data[column]>=lower_bound)]
        print(f"Romvoed count outliers:{len(data)-len(data_without_outliers)}")
        data = data_without_outliers
    return data_without_outliers    


def load_settings(settings_path:str='./../../preprocessing_settings.json', remove_possible_targets=False)-> edict:
    """
    Load settings from a JSON file.

    Parameters:
        settings_path (str): The path to the settings JSON file.
        remove_possible_targets (bool): If True, removes columns that are possible targets from the dataset.

    Returns:
        dict: The settings loaded from the JSON file.
    """
    try:
        # Open and read the JSON file
        with open(settings_path, 'r', encoding="utf-8") as file:
            settings = json.load(file)
            settings = edict(settings)
        if remove_possible_targets:
            settings.ignore_columns.extend([col for col in settings.possible_targets if col not in settings.ignore_columns])
        return settings
    except:
        raise