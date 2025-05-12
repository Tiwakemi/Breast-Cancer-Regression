import joblib
from flask import Flask, render_template, request
import sqlite3
import numpy as np
from contextlib import closing

app = Flask(__name__)

model = joblib.load("logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")

def init_db(db_name='breastcancer.db'):
    """Initialize the database and create the predictions table if it doesn't exist."""
    
    table_schema = """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        mean_radius REAL NOT NULL,
        mean_texture REAL NOT NULL,
        mean_perimeter REAL NOT NULL,
        mean_area REAL NOT NULL,
        mean_smoothness REAL NOT NULL,
        mean_compactness REAL NOT NULL,
        mean_concavity REAL NOT NULL,
        mean_concave_points REAL NOT NULL,
        mean_symmetry REAL NOT NULL,
        mean_fractal_dimension REAL NOT NULL,
        prediction TEXT NOT NULL
    )
    """

    with closing(sqlite3.connect(db_name)) as conn:
        with closing(conn.cursor()) as cursor:
            cursor.execute(table_schema)
        conn.commit()


def insert_prediction(data):
    """Insert a prediction record into the database."""
    insert_query = '''
        INSERT INTO predictions (
            user_name, mean_radius, mean_texture, mean_perimeter, mean_area,
            mean_smoothness, mean_compactness, mean_concavity,
            mean_concave_points, mean_symmetry, mean_fractal_dimension, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    with closing(sqlite3.connect('breastcancer.db')) as conn:
        with closing(conn.cursor()) as c:
            c.execute(insert_query, data)
        conn.commit()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        user_name = request.form['user_name']
        inputs = [float(request.form[key]) for key in [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area',
            'mean_smoothness', 'mean_compactness', 'mean_concavity',
            'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension'
        ]]

        # Scale and predict
        features = np.array([inputs])
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)
        result = "Malignant" if prediction[0] == 0 else "Benign"

        # Save to database
        insert_prediction((user_name, *inputs, result))

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")


@app.route('/output')
def output():
    select_query = '''
        SELECT user_name, mean_radius, mean_texture, mean_perimeter, mean_area,
               mean_smoothness, mean_compactness, mean_concavity,
               mean_concave_points, mean_symmetry, mean_fractal_dimension, prediction
        FROM predictions
    '''
    with closing(sqlite3.connect('breastcancer.db')) as conn:
        with closing(conn.cursor()) as c:
            c.execute(select_query)
            records = c.fetchall()

    return render_template('result.html', records=records)

if __name__ == "__main__":
    init_db()
    app.run(debug=True)