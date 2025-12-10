import pandas as pd
try:
    df = pd.read_csv('cleaned_data.csv')
    scale_cols=['Net_Metrekare', 'Brüt_Metrekare', 'Oda_Sayısı', 'Bulunduğu_Kat', 
                  'Binanın_Yaşı', 'Binanın_Kat_Sayısı', 'Banyo_Sayısı']
    missing = [col for col in scale_cols if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
    else:
        print("All scale_cols present")
except Exception as e:
    print(f"Error: {e}")
