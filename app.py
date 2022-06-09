"""
@Description of folders :

pickled  : It consist of 2 pickled files. 
venv  :  It consist of all the installed libraries required for this project.

@Description of app.py

The main goal of this file is to create a streamlit app which which implemets the 
recommender model into a nice frontend.

The flow of the app.py is as follows : 
1. Importing the required libraries.
2. Unpickling the pickled files.
3. Model fitting.
4. Creating recommeder function.
5. Implementing model into web page.

"""

#--------------Importing the required libraries--------------

import pandas as pd
import numpy as np
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st

#--------------Unpickling the pickled files-----------------

rating_table = pickle.load(open("pickled/rating_table.pkl","rb"))
books_image_data = pickle.load(open("pickled/books_image_data.pkl","rb"))

#----------------------Model fitting------------------------

sparse_matrix = csr_matrix(rating_table)
model = NearestNeighbors(algorithm='brute')
model.fit(sparse_matrix)

#-------------------Creating recommeder function---------------

#This function takes a book name >><str> and returns a list of 5 books and their image link.
def recommend(book_name):
  """
  @param -- <str> book_name.
  @returns -- Two <list> of <str> 
  """
  recommended_books = []
  image_url = []
  #Extract the index of input book.
  book_index = np.where(rating_table.index==book_name)[0][0]
  distances , suggestions = model.kneighbors(rating_table.iloc[book_index,:].values.reshape(1,-1),n_neighbors=5)

  #Convert suggested 2d array into 1d array.
  suggestions = np.ravel(suggestions, order='C')

  #Get recommended books name.
  for i in suggestions:
    recommended_books.append(rating_table.index[i])

  #Get image link of those recommended books.
  for i in recommended_books:
    image_url.append(books_image_data[books_image_data["title"] == i ].image.to_string(index=False))
    
  return recommended_books,image_url

#---------------Implementing model into web page-----------------

#Refer streamlit documentation for frontend.

st.subheader("COLLABORATIVE FILTERING BASED BOOKS RECOMMENDER") #Title

#Extracting the books name from the loaded pickled rating table
books_name = rating_table.index.to_list()
#Dropdown select menu
selected_book = st.selectbox(
     'Search Your Book Here',
     books_name)

if st.button('Search'):
    books,images = recommend(selected_book) 

    container1 =st.container()
    container1.subheader("You Searched For:")
    container1.markdown(books[0])
    container1.image(images[0])

    st.subheader("Users Also Liked:")
    col1, col2, col3,col4 = st.columns(4)

    with col1:
        st.text(books[1])
        st.image(images[1])
    with col2:
        st.text(books[2])
        st.image(images[2])
    with col3:
        st.text(books[3])
        st.image(images[3])
    with col4:
        st.text(books[4])
        st.image(images[4])




