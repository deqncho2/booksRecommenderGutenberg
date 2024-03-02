from flask import Flask, render_template, redirect, url_for, abort, request
import json
from flask_caching import Cache
import pandas as pd
import cohere
from annoy import AnnoyIndex

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'SimpleCache'})
cache.init_app(app)

titles_idx_path = 'gutendex_titles_idx.json'
authors_idx_path = 'gutendex_authors_idx.json'
books_descriptions = pd.read_csv('ids_descriptions.csv')
model_name = "embed-english-v3.0"
api_key = "wRJ57aIZtgcoIVdzMayvIqLLIQU009zqMhRDWmGy"
input_type_embed = "search_document"

# Now we'll set up the cohere client.
co = cohere.Client(api_key)

f = 1024  # Length of item vector that will be indexed
search_index = AnnoyIndex(f, 'angular')
search_index.load('description_embeddings.ann')  # Path to the Annoy index file

query = "Give me a science fiction novel with an interesting plot twist"
input_type_query = "search_query"

with open(titles_idx_path, 'r') as file:
    titles_idx = json.load(file)

with open(authors_idx_path, 'r') as file:
    authors_idx = json.load(file)

with open('gutendex_books.json', 'r') as f:
    books = json.load(f)

# Sort books by title
books.sort(key=lambda x: x['title'])

with open('gutendex_books_idx.json', 'r') as f:
    books_idx = json.load(f)

# Function to categorize books by first letter
def categorize_books(books):
    categorized = {}
    for book in books:
        book['id'] = str(book['id'])
        first_letter = book['title'][0].upper()
        if first_letter not in categorized:
            categorized[first_letter] = []
        categorized[first_letter].append(book)
    return categorized

categorized_books = categorize_books(books)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/books/<letter>', defaults={'page': 1})
@app.route('/books/<letter>/<int:page>')
def books_by_letter(letter, page):
    if letter in categorized_books:
        books_per_page = 99  # Set the number of books per page to 100
        start = (page - 1) * books_per_page
        end = start + books_per_page
        books_to_display = categorized_books[letter][start:end]
        total_pages = len(categorized_books[letter]) // books_per_page + (1 if len(categorized_books[letter]) % books_per_page > 0 else 0)
        return render_template('letter_page.html', books=books_to_display, letter=letter, page=page, total_pages=total_pages)
    else:
        return "No books found for this letter", 404

@app.route('/book/<int:id>')
def book_page(id):
    if str(id) not in books_idx:
        abort(404)  # Book not found
    book_info = books_idx[str(id)]
    return render_template('book_page.html', book=book_info)

@app.route('/search_by_title')
def search_by_title():
    return render_template('search_by_title.html')

@app.route('/search_by_title_results')
def search_by_title_results():
    query = request.args.get('query')
    page = request.args.get('page', 1, type=int)
    books_per_page = 10

    if query:
        cache_key = f'search_by_title_results_{query}_{page}'
        cached_results = cache.get(cache_key)

        if cached_results is not None:
            books_to_display, total_pages = cached_results
        else:
            results = {title: id for title, id in titles_idx.items() if
                       all(word.lower() in title.lower() for word in query.split())}

            # results = {title: id for title, id in titles_idx.items() if query.lower() in title.lower()}
            books = [books_idx[str(id)] for title, id in results.items() if str(id) in books_idx]
            total_books = len(books)
            total_pages = (total_books + books_per_page - 1) // books_per_page

            start = (page - 1) * books_per_page
            end = start + books_per_page
            books_to_display = books[start:end]
            cache.set(cache_key, (books_to_display, total_pages), timeout=5*60)

        return render_template('search_by_title_results.html', books=books_to_display, total_pages=total_pages, current_page=page, query=query)
    return redirect(url_for('search_by_title'))

@app.route('/search_by_author')
def search_by_author():
    return render_template('search_by_author.html')


@app.route('/search_by_author_results')
def search_by_author_results():
    query = request.args.get('query')
    page = request.args.get('page', 1, type=int)
    books_per_page = 10

    if query:
        cache_key = f'search_by_author_results_{query}_{page}'
        cached_results = cache.get(cache_key)

        if cached_results is not None:
            books_to_display, total_pages = cached_results
        else:
            results = {author: lst_ids for author, lst_ids in authors_idx.items() if
                       all(word.lower() in author.lower() for word in query.split())}

            books = []
            for a in results:
                for b in results[a]:
                    if str(b) in books_idx:
                        books.append(books_idx[str(b)])
            # books = [books_idx[str(id)] for title, id in results.items() if str(id) in books_idx]
            total_books = len(books)
            total_pages = (total_books + books_per_page - 1) // books_per_page

            start = (page - 1) * books_per_page
            end = start + books_per_page
            books_to_display = books[start:end]
            cache.set(cache_key, (books_to_display, total_pages), timeout=5*60)

        return render_template('search_by_author_results.html', books=books_to_display, total_pages=total_pages, current_page=page, query=query)
    return redirect(url_for('search_by_author'))


@app.route('/search_by_query')
def search_by_query():
    return render_template('search_by_query.html')


@app.route('/search_by_query_results')
def search_by_query_results():
    query = request.args.get('query')
    page = request.args.get('page', 1, type=int)
    books_per_page = 10

    if query:
        cache_key = f'search_by_query_results_{query}_{page}'
        cached_results = cache.get(cache_key)

        if cached_results is not None:
            books_to_display, total_pages = cached_results
        else:
            # Get the query's embedding
            query_embed = co.embed(texts=[query],
                                   model=model_name,
                                   input_type=input_type_query).embeddings

            # Retrieve the nearest neighbors
            similar_item_ids = search_index.get_nns_by_vector(query_embed[0], 100, include_distances=True)
            # Format the results
            # query_results = pd.DataFrame(data={'label': books_descriptions.iloc[similar_item_ids[0]]['text'],
            #                                    'distance': similar_item_ids[1]})

            results = list(books_descriptions.iloc[similar_item_ids[0]]['id'])
            # results = {author: lst_ids for author, lst_ids in authors_idx.items() if
            #            all(word.lower() in author.lower() for word in query.split())}

            books = []
            for book_id in results:
                books.append(books_idx[str(book_id)])
            # books = [books_idx[str(id)] for title, id in results.items() if str(id) in books_idx]
            total_books = len(books)
            total_pages = (total_books + books_per_page - 1) // books_per_page

            start = (page - 1) * books_per_page
            end = start + books_per_page
            books_to_display = books[start:end]
            cache.set(cache_key, (books_to_display, total_pages), timeout=5*60)

        return render_template('search_by_query_results.html', books=books_to_display, total_pages=total_pages, current_page=page, query=query)
    return redirect(url_for('search_by_query'))


if __name__ == '__main__':
    app.run(debug=True, port=8000)
