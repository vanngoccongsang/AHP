import pandas as pd
import os
from flask import Flask, render_template, request, redirect, url_for, session, flash

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'your_secret_key_here'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx', 'xls'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        criteria_names = request.form.get('criteria_names', '').split(',')
        criteria_names = [name.strip() for name in criteria_names if name.strip()]
        if not criteria_names:
            return render_template('index.html', error="Vui lòng nhập ít nhất một tiêu chí.")
        session['criteria_names'] = criteria_names
        session['criteria_size'] = len(criteria_names)
        session.modified = True
        return redirect(url_for('matrix_input'))
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        data = pd.read_excel(file_path)

        criteria_names = data.iloc[:, 0].tolist()
        session['criteria_names'] = criteria_names
        session['criteria_size'] = len(criteria_names)

        criteria_matrix = [[1 if i == j else 0 for j in range(len(criteria_names))] for i in range(len(criteria_names))]
        for i, row in enumerate(data.values[:, 1:]):
            for j, cell in enumerate(row):
                if i < j: 
                    try:
                        numeric_value = float(cell)
                        criteria_matrix[i][j] = numeric_value
                        criteria_matrix[j][i] = 1 / numeric_value if numeric_value != 0 else 0
                    except (ValueError, TypeError):
                        criteria_matrix[i][j] = 0
                        criteria_matrix[j][i] = 0

        session['criteria_matrix'] = criteria_matrix
        session.modified = True
        
        flash('File uploaded and data processed successfully')
        return redirect(url_for('matrix_input'))
    flash('Invalid file format')
    return redirect(request.url)

@app.route('/matrix_input', methods=['GET', 'POST'])
def matrix_input():
    criteria_size = session.get('criteria_size', 0)
    criteria_matrix = session.get('criteria_matrix', None)

    if request.method == 'POST':
        if criteria_matrix is None:
            criteria_matrix = create_matrix(criteria_size)
            for i in range(criteria_size):
                for j in range(i + 1, criteria_size):
                    value = float(request.form.get(f'value_{i}_{j}', 0))
                    criteria_matrix[i][j] = value
                    criteria_matrix[j][i] = 1 / value if value != 0 else 0

        criteria_pairwise_matrix = create_pairwise_matrix(criteria_matrix)
        criteria_cw = calculate_cw(criteria_pairwise_matrix)
        session['criteria_weights'] = criteria_cw
        _, consistency_index, lambda_max = calculate_consistency_ratio(criteria_matrix, criteria_cw)
        consistency_ratio = consistency_index / get_random_consistency(criteria_size)
        session['consistency_ratio'] = consistency_ratio
        session.modified = True
        
        if consistency_ratio < 0.1:
            return redirect(url_for('house_types'))
        else:
            return render_template('matrix_input.html', size=criteria_size, error="Ma trận không nhất quán, vui lòng nhập lại.")
    elif criteria_matrix is not None:
        return render_template('matrix_input.html', size=criteria_size, criteria_matrix=criteria_matrix)
    else:
        return render_template('matrix_input.html', size=criteria_size)

@app.route('/house_types', methods=['GET', 'POST'])
def house_types():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.makedirs(app.config['UPLOAD_FOLDER'])
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)
                process_excel_sheets(file_path)
                flash('File uploaded and data processed successfully')
                return redirect(url_for('enter_house_matrices'))
            else:
                flash('Invalid file format')
                return redirect(request.url)
        else:
            house_names_input = request.form.get('house_names', '')
            house_names = [name.strip() for name in house_names_input.split(',') if name.strip()]
            if not house_names:
                return render_template('house_types.html', error="Vui lòng nhập ít nhất một tên nhà.")
            session['house_names'] = house_names
            session['house_count'] = len(house_names)
            return redirect(url_for('enter_house_matrices'))
    return render_template('house_types.html')

def process_excel_sheets(file_path):
    data = pd.read_excel(file_path, sheet_name=None)
    house_matrices = []
    house_names = None
    for sheet_name, sheet_data in data.items():
        house_matrix = sheet_data.values.tolist()
        if house_names is None:
            house_names = sheet_data.columns.tolist()
        matrix_size = len(house_matrix)
        processed_matrix = [[1 if i == j else 0 for j in range(matrix_size)] for i in range(matrix_size)]
        for i in range(matrix_size):
            for j in range(matrix_size):
                if i < j:
                    processed_matrix[i][j] = house_matrix[i][j]
                    processed_matrix[j][i] = 1 / house_matrix[i][j] if house_matrix[i][j] != 0 else 0
        house_matrices.append(processed_matrix)
    session['house_matrices'] = house_matrices
    session['house_count'] = len(house_names)
    session['house_names'] = house_names
    session.modified = True

@app.route('/enter_house_matrices', methods=['GET', 'POST'])
def enter_house_matrices():
    if request.method == 'POST':
        house_count = session.get('house_count', 0)
        criteria_names = session.get('criteria_names', [])
        house_names = session.get('house_names', [])
        alternative_cws = []
        final_scores = [0] * house_count

        house_matrices = session.get('house_matrices', [])
        criteria_weights = session.get('criteria_weights', [])

        for crit_index, crit_name in enumerate(criteria_names):
            matrix = house_matrices[crit_index]
            alt_pairwise_matrix = create_pairwise_matrix(matrix)
            alt_cw = calculate_cw(alt_pairwise_matrix)
            alternative_cws.append(alt_cw)

            for i in range(house_count):
                final_scores[i] += alt_cw[i] * criteria_weights[crit_index]

        session['final_scores'] = final_scores
        return redirect(url_for('results'))
    else:
        house_count = session.get('house_count', 0)
        criteria_names = session.get('criteria_names', [])
        house_matrices = session.get('house_matrices', [])
        house_names = session.get('house_names', [])
        return render_template('enter_house_matrices.html', house_count=house_count, criteria_names=criteria_names, house_matrices=house_matrices, house_names=house_names)

@app.route('/results')
def results():
    final_scores = session.get('final_scores', [])
    sorted_scores = sorted(enumerate(final_scores), key=lambda x: x[1], reverse=True)
    house_names = session.get('house_names', [])
    consistency_ratio = session.get('consistency_ratio', 0)

    results_with_names = [(house_names[idx], score) for idx, score in sorted_scores]

    return render_template('results.html', sorted_scores=results_with_names, consistency_ratio=consistency_ratio)

def to_percent(value):
    return "{:.2%}".format(value)

app.jinja_env.filters['percent'] = to_percent

def get_random_consistency(size):
    cr_table = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59}
    return cr_table.get(size, 1.59) 

def create_matrix(size):
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

def create_pairwise_matrix(matrix):
    size = len(matrix)
    pairwise_matrix = [[0] * size for _ in range(size)]
    column_sums_result = column_sums(matrix)
    for i in range(size):
        for j in range(size):
            pairwise_matrix[i][j] = round(matrix[i][j] / column_sums_result[j], 4)
    return pairwise_matrix

def calculate_cw(pairwise_matrix):
    size = len(pairwise_matrix)
    cw = [sum(pairwise_matrix[i][j] for j in range(size)) / size for i in range(size)]
    return cw

def calculate_consistency_ratio(matrix, cw):
    size = len(matrix)
    weighted_sum = [sum(matrix[i][j] * cw[j] for j in range(size)) for i in range(size)]

    lambda_max = sum(weighted_sum[i] / cw[i] for i in range(size)) / size
    consistency_index = (lambda_max - size) / (size - 1)
    
    return consistency_index, consistency_index, lambda_max

def column_sums(matrix):
    size = len(matrix)
    col_sum = [sum(matrix[i][j] for i in range(size)) for j in range(size)]
    return col_sum

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
