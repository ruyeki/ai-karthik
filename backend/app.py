from flask import Flask, request, jsonify
from flask_cors import CORS
from multi_modal_llm import process_pdf, summarize_content, connect_db, store_to_db, query_llm,create_new_db
import os


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_DIR = "./backend/documents/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

retriever = connect_db("ava")

#upload pdf and then parse, summarize, and store in db
@app.route('/upload', methods=['POST'])
def upload_pdf(): 
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    project_name = request.form.get('project_name', 'default')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Only PDF files allowed"}), 400

    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    # Process PDF
    texts, tables, images = process_pdf(file_path)

    # Summarize content
    text_summaries, table_summaries, image_summaries = summarize_content(texts, tables, images)

    # Connect or create DB for project
    retriever = connect_db(project_name)

    # Store data in DB
    store_to_db(retriever, text_summaries, texts, image_summaries, images, tables, table_summaries)

    return jsonify({"message": f"Uploaded and processed {file.filename} for project {project_name}"}), 200



@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    question = data.get('question')
    print("User question: ", question)

    if not question:
        return jsonify({"error": "No question provided"}), 400


    if retriever: 
        print("DB Connection is ready.")
    else: 
        return jsonify({"error": "Retriever not available"}), 500

    text_response = query_llm(retriever, question)

    return jsonify(text_response = text_response)

@app.route('/create_project', methods = ['POST'])
def create_project(): 

    #user submits name of project
    data = request.get_json()
    project_name = data.get('project_name')

    if not project_name: 
        return jsonify({"error": "Missing project name."}), 400

    create_new_db(project_name)

    return jsonify({
        "message": f"Successfully created project: {project_name}"
    }), 200

if __name__ == '__main__':
    app.run(debug=True)