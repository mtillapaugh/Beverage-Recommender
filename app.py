import pandas as pd
import json
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialization
app = Flask(__name__)

# Homepage
@app.route('/')
def homepage():
    return render_template('home.html')

# Similar Page
@app.route('/similar', methods=['GET', 'POST'])
def similar_page():
    df = pd.read_csv('./static/cocktaildb.csv')
    drink_list = pd.Series([x for x in df['strDrink'].str.replace("'", "")])
    return render_template("similar.html", drink_list=drink_list)

# Quiz Page
@app.route('/quiz', methods=['GET', 'POST'])
def quiz_page():
    yes_alcohol = ['alcoholic', 'non-alcoholic']
    glass_type = ['none', 'cocktail glass', 'highball glass', 'collins glass', 'old-fashioned glass', 'shot glass', 'whiskey glass']
    alcohol_list = ['none', 'rum', 'gin', 'vodka', 'whiskey', 'scotch', 'tequila', 'brandy', 'bourbon']
    minor_alcohol_list = ['none', 'sweet vermouth', 'orange liqueur', 'bailey', 'kahlua', 'triple sec', 'Schnapps', 'bitters', 'malibu', 'grenadine']
    fruit_list = ['none', 'pineapple', 'maraschino cherry', 'cranberry', 'raspberry', 'apricot']
    citrus_list = ['none', 'lime', 'lemon', 'orange']
    other_list = ['none', 'tonic water', 'coca-cola', 'sugar', 'cream', 'syrup', 'coffee', 'ginger', 'vanilla', 'mint']
    return render_template('quiz.html',
                           yes_alcohol=yes_alcohol,
                           glass_type=glass_type,
                           alcohol_list=alcohol_list,
                           minor_alcohol_list=minor_alcohol_list,
                           fruit_list=fruit_list,
                           citrus_list=citrus_list,
                           other_list=other_list)

# Results Page
@app.route('/results', methods=['GET', 'POST'])
def results_page():
    if request.method == "POST":
        if request.form.get("drink_name"):
            drink_name = request.form.get("drink_name", None)
            if drink_name!=None:
                rec_list = get_recommendations(drink_name)
                return render_template('results.html', drink_name=drink_name, rec_list=rec_list)

        else:
            a = request.form.get("alc")
            b = request.form.get("glass")
            c = request.form.get("alcohollist")
            d = request.form.get("minoralcohol")
            e = request.form.get("fruit")
            f = request.form.get("citrus")
            g = request.form.get("other")
            ingredient_list = a + ', ' + b + ', ' + c + ', ' + d + ', ' + e + ', ' + f + ', ' + g
            new_drink = {'strDrink': 'New Drink', 'soup': ingredient_list}
            df = pd.read_csv('./static/cocktaildb.csv')
            df = df.append(new_drink, ignore_index=True)

            # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
            tfidf = TfidfVectorizer(stop_words='english')
            # Construct the required TF-IDF matrix by fitting and transforming the data
            tfidf_matrix = tfidf.fit_transform(df['soup'])
            # Compute the cosine similarity matrix
            cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
            # Construct a reverse map of indices and movie titles
            indices = pd.Series(df.index, index=df['strDrink']).drop_duplicates()

            rec_list = get_recommendations('New Drink', cosine_sim, indices)
            return render_template('results.html', rec_list=rec_list, df=df)
            # a = str(request.form['Vodka'])
            # b = str(request.form['Gin'])
            # s = [a, b]
            # ingredient_list = ' '.join([str(x) for x in s])
            # new_drink = {'strDrink': 'New Drink', 'soup': ingredient_list}
            # df = df.append(new_drink, ignore_index=True)
            # rec_list = get_recommendations('New Drink')
            #return render_template('results.html', drink_name=drink_name, rec_list=rec_list)

    return render_template('results.html', drink_name=drink_name)

# Test
@app.route('/quiz', methods=['GET', 'POST'])
def add_ingredient(ingredient_list):
    if request.method == "POST":
        ingredient_list.append(request.form.get("clicked_btn"))
    return ingredient_list

# Test
@app.route('/remove', methods=['GET', 'POST'])
def remove_ingredient(ingredient_list):
    if request.method == "POST":
        ingredient_list.remove(request.form.get("clicked_btn"))
    return ingredient_list


# Prepare TF-IDF
df = pd.read_csv('./static/cocktaildb.csv')
# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['soup'])
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['strDrink']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim, indices=indices):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the drink indices
    drink_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df.iloc[drink_indices]


if __name__ == '__main__':
    app.run(debug=True)