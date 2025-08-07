import json
from pathlib import Path

import pandas as pd


current_dir = Path(__file__).parent

"""
Monthly payroll data in German 
Ecach individual has multiple: income sources/variables
across a whole time period 
get rid of all the zero rows 
columns are consistent across all users

specific to this csv, 
are the descriptions/payroll types consistent across all users? 
"""

# Translate headers
column_translation_dict = {
    "Firma": "Company",
    "Persnr.": "Personnel No.",
    "ID": "ID",
    "Name": "Name",
    "Geschlecht": "Gender",
    "Lohnart": "Payroll Type",
    "Beschreibung": "Description",
    "E/B": "Earnings/Deductions",
    "Jänner_2022": "January_2022",
    "Februar_2022": "February_2022",
    "März_2022": "March_2022",
    "April_2022": "April_2022",
    "Mai_2022": "May_2022",
    "Juni_2022": "June_2022",
    "Juli_2022": "July_2022",
    "August_2022": "August_2022",
    "Sept._2022": "September_2022",
    "Oktober_2022": "October_2022",
    "Nov._2022": "November_2022",
    "Dez._2022": "December_2022",
    "Jänner_2023": "January_2023",
    "Februar_2023": "February_2023",
    "März_2023": "March_2023",
    "April_2023": "April_2023",
    "Mai_2023": "May_2023",
    "Juni_2023": "June_2023",
    "Juli_2023": "July_2023",
    "August_2023": "August_2023",
    "Sept._2023": "September_2023",
    "Oktober_2023": "October_2023",
    "Nov._2023": "November_2023",
    "Dez._2023": "December_2023",
    "Jänner_2024": "January_2024",
    "Februar_2024": "February_2024",
    "März_2024": "March_2024",
    "April_2024": "April_2024",
    "Mai_2024": "May_2024",
    "Juni_2024": "June_2024",
    "Juli_2024": "July_2024",
    "August_2024": "August_2024",
    "Sept._2024": "September_2024",
    "Oktober_2024": "October_2024",
    "Nov._2024": "November_2024",
    "Dez._2024": "December_2024",
    "Jänner_2025": "January_2025",
    "Februar_2025": "February_2025",
    "März_2025": "March_2025",
    "April_2025": "April_2025",
    "Mai_2025": "May_2025",
    "Juni_2025": "June_2025",
    "Gesamtsumme": "Total Amount",
}


german_to_english = {
    "** AUSZAHLUNG **": "** Payout **",
    "** BRUTTO **": "** Gross **",
    "* § 68 (215)": "§68 (215) exemption",
    "* §67/3-7": "Tax under §67/3–7 (bonus/exempt allowances)",
    "* ABZ. SV WEITERV.": "Social insurance deduction continuation",
    "* ABZÜGE": "* Deductions",
    "* ABZÜGE DIV": "Various deductions",
    "* ANR. LST (260)": "Deducted income tax (260)",
    "* BEM §67/1,2": "Assessment base §67/1,2",
    "* BEM. SV LFD.": "Assessment base social insurance ongoing",
    "* BEM. SV SZ": "Assessment base social insurance special payment",
    "* BRUTTO (210)": "Gross (210)",
    "* LST PFL. (245)": "Income tax liability (245)",
    "§ 68/1 FREI": "Exempt under §68/1",
    "§ 68/2 FREI": "Exempt under §68/2",
    "Abfertigung ges. (Auszahlung)": "Severance payment (payout)",
    "BASIS J/6": "Base for J/6 calculation",
    "BEM §67/3,4": "Tax base §67/3,4",
    "BEM.MV-KASSE GESAMT": "Assessment base company pension fund",
    "BEM. LST TAB.": "Income tax assessment base",
    "BRU": "Gross pay",
    "BRUTTO": "Gross",
    "BRUTTOSUMME": "Gross total",
    "DB BEMESSUNG": "Employer contribution assessment base",
    "DB BERECHNET (FB BERÜCK)": "Employer contribution calculated (tax-free amount considered)",
    "DZ BERECHNET (FB BERÜCK)": "Supplementary contribution calculated (tax-free considered)",
    "E-CARD GEBÜHR": "e-Card fee (Austrian health insurance)",
    "Ersatzl. Urlaub 67/8d SZ": "Compensated vacation §67/8d (special payment)",
    "Ersatzl. Urlaub 67/8d lfd.": "Compensated vacation §67/8d ongoing",
    "FREIBETRAG": "Tax-free allowance",
    "Gehalt": "Salary",
    "Homeoffice": "Home office",
    "J/6 BER.": "One-sixth calculation (bonus proportion)",
    "J/6 ÜBERHANG": "One-sixth overhang",
    "KOMM.ST.BEM.": "Municipal tax assessment base",
    "KOMM.ST.BERECHNET (FB ERÜCK)": "Municipal tax calculated (exemptions applied)",
    "Kontroll6tel": "Control sixth (bonus eligibility check)",
    "Kostenübern. Par. 26 Z 5 lit b": "Cost coverage §26 section 5 lit b",
    "LST 67/1,2": "Income tax §67/1,2",
    "LST LFD": "Income tax ongoing",
    "LST LFD.": "Income tax (ongoing)",
    "LST §67/1-2": "Income tax §67/1–2",
    "LST §67/1,2": "Income tax §67/1,2",
    "LST §67/3-7": "Income tax §67/3–7",
    "MV-KASSE GESAMT": "Company pension fund total",
    "Prämie (laufend)": "Bonus (ongoing)",
    "Prämie SZ": "Bonus (special payment)",
    "Sachbezug": "Non-cash benefit",
    "STEUERFR. § 26": "Tax-free under §26",
    "SV J/6 ÜBERHANG": "Social insurance overhang (J/6)",
    "SV LFD.": "Ongoing social insurance",
    "SV SZ": "Social insurance special payment",
    "SV-DG ANTEIL": "Employer's social insurance contribution",
    "SV-EINBEH. UE/KE": "Social insurance deduction UE/KE",
    "SV-LFD. UE/KE": "Social insurance ongoing UE/KE",
    "SV-SZ UE/KE": "Social insurance special payment UE/KE",
    "SV-Tage für UE/UA/EL f. Url.": "Insurance days for leave/parental/comp. for vacation",
    "SV.§67/1,2(225)": "Social insurance §67/1,2 (225)",
    "SZ 67/1,2 IN J/6": "Special payment §67/1,2 in annual sixth",
    "Tage Homeoffice": "Days working from home",
    "ÜStd. 50 %  § (68/2)": "Overtime 50% §68/2",
    "ÜStd. Grundvergütung": "Overtime base pay",
    "ÜStd. Zuschl. 100 % (§ 68/1)": "Overtime bonus 100% (§68/1)",
    "U-BAHNSTEUER": "Subway tax (Vienna only)",
    "Urlaubszuschuss Aliqu. 688": "Pro-rated vacation bonus 688",
    "Weihnachtsremuneration Al. 688": "Christmas bonus per agreement 688",
    "*** ALV-% LFD RÜCKVERR.": "Unemployment contribution % ongoing (retro adjustment)",
    "*** ALV-% SZ RÜCKKVERR.": "Unemployment contribution % special payment (retro adjustment)",
}


if __name__ == "__main__":

    # Load with proper encoding and delimiter
    df = pd.read_csv(f"{current_dir}/Lohnkonto2022-2025.csv", encoding="ISO-8859-1", sep=";")
    df.rename(columns=column_translation_dict, inplace=True)

    unique_names = df["Name"].unique().tolist()
    keys = {}
    all_descriptions = set()

    for person in unique_names:
        df_person = df[df["Name"] == person].copy()
        keys[person] = {}

        for _, row in df_person.iterrows():
            payroll_type = row["Payroll Type"]
            description = row["Description"].strip()
            # Initialize an empty list if not already there
            if payroll_type not in keys[person]:
                keys[person][payroll_type] = []

            # Append the new description
            keys[person][payroll_type].append(description)
            all_descriptions.add(description)

        keys[person][payroll_type] = sorted(keys[person][payroll_type])

    count_descriptions = {description: [] for description in all_descriptions}
    for person in unique_names:
        df_person = df[df["Name"] == person].copy()

        for _, row in df_person.iterrows():
            description = row["Description"].strip()
            count_descriptions[description].append(person)
    for key, list_people in count_descriptions.items():
        count_descriptions[key] = (len(list_people) / len(unique_names), list_people)
    sorted_dict = dict(sorted(count_descriptions.items(), key=lambda item: (item[0], item[1][0] if item[1] else "")))
    print(all_descriptions)

    # Save to a readable JSON file
    with open(f"{current_dir}/payroll_keys.json", "w", encoding="utf-8") as f:
        json.dump(keys, f, indent=2, ensure_ascii=False)
    with open(f"{current_dir}/descriptions_count.json", "w", encoding="utf-8") as f:
        json.dump(sorted_dict, f, indent=2, ensure_ascii=False)

    if "Description" in df.columns:
        df["Description"] = df["Description"].str.strip()
        df["Description"] = df["Description"].map(german_to_english).fillna(df["Description"])

    # Save translated CSV
    df.to_csv(f"{current_dir}/Lohnkonto2022-2025_english.csv", index=False, encoding="utf-8")
