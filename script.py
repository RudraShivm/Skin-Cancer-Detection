import pandas as pd
df = pd.read_csv("data/isic-2024-challenge/train-metadata.csv", low_memory=False)
print(df.head())
my_image_meta = df[df["isic_id"] == "ISIC_0015670"]
print(my_image_meta)
print(my_image_meta[["age_approx", "sex", "anatom_site_general"]])
