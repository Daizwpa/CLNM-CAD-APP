from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class PatientModel:
    # Attributes:
    age: int = 45
    sexe: str = "Female"
    alcool: str = "No"
    tabac: str = "No"
    mcv_fam: str = "No"
    atcd_kc: str = "No"
    invasion: str = "No"
    multifocalite: str = "No"
    ete: str = "No"
    t_stage: str = "T1"
    meta: str = "M0"
    histology_type: str = "Papillary"
    ajcc8: str = "Stade I"
    risque_ata: str = "Low"
    dose_iode: str = "0"
    levothyrox: float = 120.0


    def yes_no_encoder(self, value: str) -> str:
        return "Oui" if value == "Yes" else "Non"


    def histology_encoding(self):

        return {
            "Papillary":  "Yes" if self.histology_type == "Papillary" else  "No",
            "Medullary":  "Yes" if self.histology_type == "Medullary" else  "No",
            "Follicular": "Yes" if self.histology_type == "Follicular" else "No",
        }
    def RISQUE_RECIDIVE_ATA_encoder(self):
            if self.risque_ata == "High":  
                return "Elevé",
            elif self.risque_ata == "Intermidate":
                return "Intermédiaire"
            else:
                return "Faible"
        

    def to_model_input(self) -> pd.DataFrame:

        histology = self.histology_encoding()

        data = {
            'SEXE': "Homme" if self.sexe == "Male" else "Femme",
            'ATCD_PER_KC': self.yes_no_encoder(self.atcd_kc),
            'MCV_FAM': self.yes_no_encoder(self.mcv_fam),
            'AGE_AU_DIAGNOSTIC': self.age,
            'T': self.t_stage,
            'INVASION_VASCULAIRE': "Absence d'emboles vasculaires" if self.invasion == "No" else "Présence d'emboles vasculaires",
            'MULTIFOCALITE': self.yes_no_encoder(self.multifocalite),
            'AJCC8': self.ajcc8,
            'RISQUE_RECIDIVE_ATA': self.RISQUE_RECIDIVE_ATA_encoder(),
            'DOSE_LEVOTHYROX': self.levothyrox,
            'DOSE_CUMULEE_IODE': self.dose_iode,
            'META': "Oui" if self.meta == "M1" else "Non",
            'Papillaire': histology["Papillary"],
            'Medullaire': histology["Medullary"],
            'Vésiculaire': histology["Follicular"],
            'b_ETE': 1 if self.ete == "Yes" else 0,
            'Alcoolisme': self.yes_no_encoder(self.alcool),
            'Tabagisme': self.yes_no_encoder(self.tabac),
        }

        return pd.DataFrame([data])
