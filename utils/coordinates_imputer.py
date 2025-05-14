def coordinate_imputer(df, column, map_lat, map_long):
    df[f"{column}Latitude"] = df[column].map(map_lat)
    df[f"{column}Longitude"] = df[column].map(map_long)
    return df