import pandas as pd
import numpy as np
import datetime as dt
import argparse
import os
from tqdm import tqdm
import random

# argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessing for Knowledge Analysis")
    
    parser.add_argument("--data_folder", type=str, default="raw_dataset", help="Path to the input JSON file")
    # parser.add_argument("--num_generate_person", type=int, default=64000, help="Number of data points to generate")
    parser.add_argument("--num_generate_person", type=int, default=64000, help="Number of data points to generate")
    parser.add_argument("--num_name", type=int, default=900, help="Number of names to generate")
    parser.add_argument("--num_surname", type=int, default=1000, help="Number of surnames to generate")
    parser.add_argument("--num_country", type=int, default=800, help="Number of countries to generate")
    parser.add_argument("--num_university", type=int, default=200, help="Number of universities to generate")
    parser.add_argument("--num_major", type=int, default=100, help="Number of majors to generate")
    parser.add_argument("--num_company", type=int, default=500, help="Number of companies to generate")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    args = parse_args()
    
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(os.path.join("data", f"bio_data_{args.num_generate_person}_{args.seed}.json")):
        print(f"Data file already exists: data/bio_data_{args.num_generate_person}_{args.seed}.json")
        return
    
    set_seed(args.seed)
    
    # Load and process first names
    print("\n\n==============\nNAME\n==============\n")
    first_name_db = pd.read_csv(os.path.join(args.data_folder, "yob2010.txt"), header=None, names=["name", "gender", "count"], index_col=False, dtype={"name": str, "count": int})
    print("First names loaded:", first_name_db.shape)
    print(first_name_db.head())
    
    name_pool = first_name_db["name"][:args.num_name].tolist()

    print("First names processed:", len(name_pool))
    
    # Load and process surnames
    print("\n\n==============\nSURNAME\n==============\n")
    last_name_db = pd.read_excel(os.path.join(args.data_folder, "2010_surname.xlsx"), header=2)
    print("Last names loaded:", last_name_db.shape)
    print(last_name_db.head())
    
    surname_pool = last_name_db["SURNAME"][:args.num_surname].tolist()
    print("Last names processed:", len(surname_pool))
    
    # Load and process countries
    print("\n\n==============\nCOUNTRY\n==============\n")
    country_db = pd.read_csv(os.path.join(args.data_folder, "cities15000.txt"), header=None, 
                             names=["geonameid", "name", "asciiname", "alternatenames", "latitude", "longitude", 
                                    "feature_class", "feature_code", "country_code", "cc2", 
                                    "admin1_code", "admin2_code", "admin3_code", "admin4_code", 
                                    "population", "elevation", "dem", "timezone", "modification_date"],
                             sep="\t", dtype={"geonameid": int, "name": str, "asciiname": str, 
                                              "alternatenames": str, "latitude": float, "longitude": float, 
                                              "feature_class": str, "feature_code": str, 
                                              "country_code": str, "cc2": str, 
                                              "admin1_code": str, "admin2_code": str, "admin3_code": str, 
                                              "admin4_code": str, "population": int, 
                                              "elevation": float, "dem": float, "timezone": str,
                                              "modification_date": str})
    country_db = country_db[["name", "population"]]
    
    # sort by population and select top countries
    country_db = country_db.sort_values(by="population", ascending=False).reset_index(drop=True)
    
    print("Countries loaded:", country_db.shape)
    print(country_db.head())
    
    country_pool = country_db["name"][:args.num_country].tolist()
    print("Countries processed:", len(country_pool))
    
    # Load and process universities
    print("\n\n==============\nUNIVERSITY\n==============\n")
    university_db = pd.read_excel(os.path.join(args.data_folder, "cwts_leiden_ranking_2024.xlsx"), sheet_name="Results")
    mask = (university_db["Period"] == "2019–2022") & (university_db["Field"] == "All sciences")
    subset = university_db.loc[mask]
    subset = subset[["University", "impact_P"]]
    university_db = subset.sort_values("impact_P", ascending=False).reset_index(drop=True)  
    print("Universities loaded:", university_db.shape)
    print(university_db.head())
    university_pool = university_db["University"][:args.num_university].tolist()
    print("Universities processed:", len(university_pool))
    
    # Load and process majors
    print("\n\n==============\nMAJOR\n==============\n")
    major_db = pd.read_csv(os.path.join(args.data_folder, "cip2020.csv"), dtype={"major": str})
    mask_4digit = major_db["CIPCode"].astype(str).str.fullmatch(r"\=\"\d{2}\.\d{2}\"")
    major_db = major_db[mask_4digit].reset_index(drop=True)
    print("Majors loaded:", major_db.shape)
    print(major_db.head())
    major_pool = major_db["CIPTitle"].sample(n=args.num_major, random_state=args.seed).reset_index(drop=True).tolist()
    print("Majors processed:", len(major_pool))
    print(major_pool[:5])  # Print first 5 majors for verification
    
    # Load and process companies
    print("\n\n==============\nCOMPANY\n==============\n")
    company_db = pd.read_csv(os.path.join(args.data_folder, "companiesmarketcap.csv"), index_col=False)
    company_db = company_db.dropna(subset=["Company Names"])
    company_db = company_db[["Rank", "Company Names"]]
    print("Companies loaded:", company_db.shape)
    print(company_db.head())
    company_pool = company_db["Company Names"][:args.num_company].tolist()
    print("Companies processed:", len(company_pool))
    
    full_name_pool = []
    bio_dataset = []
    
    for i in tqdm(range(args.num_generate_person)):
        while True:
            first_name = random.choice(name_pool).capitalize()
            middle_name = random.choice(name_pool).capitalize()
            last_name = random.choice(surname_pool).capitalize()
            
            name = f"{first_name} {middle_name} {last_name}"
            
            if name not in full_name_pool:
                full_name_pool.append(name)
                break
        
        one_bio = {"name": name}
        
        one_bio["birth_date"] = {
            "year": np.random.randint(1900, 2101),
            "month": np.random.randint(1, 13),
            "day": np.random.randint(1, 32)
        }
        one_bio["birth_date"] = f"{one_bio['birth_date']['year']}-{one_bio['birth_date']['month']:02d}-{one_bio['birth_date']['day']:02d}"
        # one_bio["birth_place"] = np.random.choice(country_pool)
        # one_bio["university"] = np.random.choice(university_pool)
        # one_bio["major"] = np.random.choice(major_pool)
        # one_bio["location"] = np.random.choice(country_pool)
        # one_bio["company"] = np.random.choice(company_pool)
        
        one_bio["birth_place"] = random.choice(country_pool)
        one_bio["university"] = random.choice(university_pool)
        one_bio["major"] = random.choice(major_pool)
        one_bio["location"] = random.choice(country_pool)
        one_bio["company"] = random.choice(company_pool)    
           
        # 6 list of 25 values of 0 or 1 and 1 is random choiced 20 positions 
        one_bio["train_tamplates"] = {
            "birth_date_template": [1] * 20 + [0] * 5,  # 25 templates for birth date
            "birth_place_template": [1] * 20 + [0] * 5,  # 25 templates for birth place
            "university_template": [1] * 20 + [0] * 5,  # 25 templates for university
            "major_template": [1] * 20 + [0] * 5,  # 25 templates for major
            "company_template": [1] * 20 + [0] * 5,  # 25 templates for company
            "location_template": [1] * 20 + [0] * 5   # 25 templates for location
        }
        
        for key in one_bio["train_tamplates"]:
            # Shuffle the list and ensure the first 20 are 1s
            random.shuffle(one_bio["train_tamplates"][key])
         
        bio_dataset.append(one_bio)
    
    # Save the generated dataset to a JSON file
    output_file = os.path.join("data", f"bio_data_{args.num_generate_person}_{args.seed}.json")
    with open(output_file, "w") as f:
        import json
        json.dump(bio_dataset, f, indent=4) 
    

main()  # Call the main function to execute the script