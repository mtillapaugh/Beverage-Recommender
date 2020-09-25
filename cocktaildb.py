# Set Up Environment
import pandas as pd
import requests
import json
from pandas.io.json import json_normalize
#from sqlalchemy import create_engine

# List of parameters to search DB
search_parameters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

url = 'https://www.thecocktaildb.com/api/json/v1/1/search.php?f='
json_list = []

for i in search_parameters:
    r = requests.get(url + str(i))
    # This presumes your JSON response is a dict, if the response is a list, use extend instead of append
    json_list.append(r.json())

df = pd.json_normalize(json_list, 'drinks')

# Filter out
df_new = df[['idDrink', 'strDrink', 'strCategory', 'strIBA', 'strAlcoholic',
         'strGlass', 'strInstructions', 'strDrinkThumb', 'strIngredient1',
         'strIngredient2', 'strIngredient3', 'strIngredient4', 'strIngredient5',
         'strIngredient6', 'strIngredient7', 'strIngredient8', 'strIngredient9',
         'strIngredient10', 'strMeasure1', 'strMeasure2', 'strMeasure3', 'strMeasure4',
         'strMeasure5', 'strMeasure6', 'strMeasure7', 'strMeasure8', 'strMeasure9', 'strMeasure10']]

# Create all ingredients column
df_new['all_ingredients'] = df_new[df_new.columns[8:17]].apply(
    lambda x: ','.join(x.dropna().astype(str)), axis=1)

# Create soup column
df_new['soup'] = df_new[['all_ingredients', 'strInstructions']].apply(lambda x: ','.join(x), axis=1)

# Save new dataframe as csv
df_new.to_csv(r'./static/cocktaildb.csv', index=True, header=True)