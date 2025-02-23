import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
def load_and_preprocess():
    """Load and clean anime data with proper genre normalization"""
    df = pd.read_csv('anime.csv').head(1000)

    # Normalize genres and type
    df['genre'] = (
        df['genre']
        .fillna('unknown')
        .str.lower()
        .str.replace(r'\s+', '-', regex=True)
        .str.replace(',', ' ')
    )

    df['type'] = df['type'].fillna('unknown').str.lower()
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(1)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(df['rating'].median())


    all_genres = set()
    for genres in df['genre'].str.split():
        all_genres.update(genres)


    df['search_features'] = df['genre'] + ' ' + df['type']
    return df, sorted(all_genres)


def parse_query(query, genres):
    """Convert natural language query to structured parameters"""
    # Normalize query text
    clean_query = (
        query.lower()
        .replace(',', ' ')
        .replace(' and ', ' ')
        .replace(' with ', ' ')
    )


    found_genres = []
    for genre in genres:
        if re.search(rf'\b{genre}\b', clean_query):
            found_genres.append(genre)


    params = {
        'genres': found_genres,
        'type': next((t for t in ['movie', 'tv'] if re.search(rf'\b{t}\b', clean_query)), None),
        'max_episodes': (
            24 if re.search(r'\b(short|binge|brief)\b', clean_query)
            else int(match.group(1)) if (match := re.search(r'(\d+)\s*episodes?', clean_query))
            else None
        ),
        'min_rating': (
            8.0 if re.search(r'\b(top|high|best)\s*rated?\b', clean_query)
            else float(match.group(1)) if (match := re.search(r'rating\s*above\s*(\d\.?\d?)', clean_query))
            else None
        )
    }
    return params


def get_recommendations(df, genres, query):
    """Generate recommendations with proper similarity scoring"""
    params = parse_query(query, genres)
    filtered = df.copy()

    # Apply filters
    if params['type']:
        filtered = filtered[filtered['type'] == params['type']]
    if params['max_episodes']:
        filtered = filtered[filtered['episodes'] <= params['max_episodes']]
    if params['min_rating']:
        filtered = filtered[filtered['rating'] >= params['min_rating']]

    if params['genres']:
        # Configure TF-IDF with proper weighting
        tfidf = TfidfVectorizer(norm='l2', use_idf=True, smooth_idf=True)


        tfidf_matrix = tfidf.fit_transform(filtered['search_features'])
        query_vector = tfidf.transform([' '.join(params['genres'])])


        filtered['similarity'] = cosine_similarity(query_vector, tfidf_matrix).flatten().round(2)
        results = filtered.sort_values(['similarity', 'rating'], ascending=[False, False])
    else:

        results = filtered.sort_values(['rating', 'members'], ascending=[False, False])
        results['similarity'] = None

    return results[['name', 'similarity']].head(5)


def main():
    df, genres = load_and_preprocess()

    print("🎌 Anime Recommendation System")
    print("Enter queries like:")
    print("- 'historical sci-fi movies with high ratings'")
    print("- 'short comedy series under 20 episodes'")
    print("- 'top rated drama tv shows'")
    print("Type 'exit' to quit\n")

    while True:
        try:
            query = input("\n🔍 Your query: ").strip()
            if query.lower() in ('exit', 'quit'):
                break

            recommendations = get_recommendations(df, genres, query)

            if recommendations.empty:
                print("No matches found. Try a different query!")
                continue

            print("\nTop Recommendations:")
            print(recommendations.to_string(
                index=False,
                formatters={'similarity': lambda x: f"{x:.2f}" if not pd.isnull(x) else "N/A"}
            ))

        except Exception as e:
            print(f"⚠️ Error processing query: {str(e)}")


if __name__ == "__main__":
    main()