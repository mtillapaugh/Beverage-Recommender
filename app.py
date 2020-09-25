import pandas as pd
import numpy as np
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
@app.route('/quiz')
def quiz_page():
    return render_template('quiz.html')

# Results Page
@app.route('/results', methods=['GET', 'POST'])
def results_page():
    if request.method == "POST":
        drink_name = request.form.get("drink_name", None)
        if drink_name!=None:
            rec_list = get_recommendations(drink_name)
            return render_template('results.html', drink_name=drink_name, rec_list=rec_list)
    return render_template('results.html', drink_name=drink_name)

# Prepare TF-IDF
df = pd.read_csv('./static/cocktaildb.csv')
# Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')
# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df['all_ingredients'])
# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Construct a reverse map of indices and movie titles
indices = pd.Series(df.index, index=df['strDrink']).drop_duplicates()


def get_recommendations(title, cosine_sim=cosine_sim):
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






# Similar Page
#@app.route('/similar')
#def similar_page():
#    drink_list = ['option 1', 'option 2', 'option 3', 'option 4']
#    return render_template('similar.html', drink_list=drink_list)