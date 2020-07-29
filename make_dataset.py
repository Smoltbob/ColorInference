import pandas as pd
df = pd.read_csv("satfaces.txt", names = ["r", "g", "b", "color"], index_col = False, delimiter = " ")
colors = ["black", "white", "yellow", "orange", "red", "pink", "purple", "blue", "green", "grey", "brown"]
df = df[df["color"].isin(colors)]
df = df.sample(n=20000)
df.to_csv("satfaces_lite.csv", index=False)
