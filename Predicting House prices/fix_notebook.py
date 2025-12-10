import json

notebook_path = r"c:\Users\pc\OneDrive\Documents\GitHub\Machine learning projects\Predicting House prices\Data_Cleaning.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Helper to find cell by execution count or source content
def find_cell_by_source(nb, keyword):
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if keyword in source:
                return i, cell
    return -1, None

# 1. Modify Cell 5 (approximate location, looking for nominal_columns)
idx5, cell5 = find_cell_by_source(nb, 'nominal_columns = [')
if cell5:
    print(f"Found Cell 5 at index {idx5}")
    new_source = [
        "nominal_columns = [\n",
        "    \"Isıtma_Tipi\",'Tapu_Durumu', \"Kullanım_Durumu\",\n",
        "    \"Takas\",\n",
        "    \"Yatırıma_Uygunluk\",\n",
        "    \"Eşya_Durumu\",\n",
        "    \"Şehir\"\n",
        "]# make a list of nominal columns to get dummies\n",
        "\n",
        "# Handle rare cities before encoding\n",
        "city_counts = df['Şehir'].value_counts()\n",
        "rare_cities = city_counts[city_counts < 20].index # Threshold of 20 houses\n",
        "df['Şehir'] = df['Şehir'].apply(lambda x: 'Other' if x in rare_cities else x)\n",
        "\n",
        "df= pd.get_dummies(df, columns=nominal_columns,drop_first=True) # getting dummies for nominal columns\n",
        "print(df.head())\n",
        "# dictionary to apply ordinal encoding on age column\n",
        "age_rules = {\n",
        "    '0 (Yeni)': 0,\n",
        "    '1': 1,\n",
        "    '2': 2,\n",
        "    '3': 3,\n",
        "    '4': 4,\n",
        "    '5-10': 5,\n",
        "    '11-15': 6,\n",
        "    '16-20': 7,\n",
        "    '21 Ve Üzeri': 8\n",
        "}\n",
        "\n",
        "df['Binanın_Yaşı'] = df['Binanın_Yaşı'].map(age_rules) # map the rules to the column to apply ordinal encoding\n",
        "\n",
        "\n",
        "print(df['Binanın_Yaşı'].head())\n",
        "df.info() # check the info of the dataframe"
    ]
    cell5['source'] = new_source
else:
    print("Cell 5 not found!")

# 2. Modify Cell 11 (looking for rare_cities logic)
idx11, cell11 = find_cell_by_source(nb, "rare_cities = city_counts[city_counts < 20].index")
if cell11:
    print(f"Found Cell 11 at index {idx11}")
    new_source = [
        "# Rare city handling moved to Cell 5 to avoid KeyError\n",
        "# city_counts = df['Şehir'].value_counts()\n",
        "# rare_cities = city_counts[city_counts < 20].index # Threshold of 20 houses\n",
        "# df['Şehir'] = df['Şehir'].apply(lambda x: 'Other' if x in rare_cities else x)"
    ]
    cell11['source'] = new_source
else:
    print("Cell 11 not found!")

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
