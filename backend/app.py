from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
from helper_functions import process_pdf, summarize_content, connect_db, store_to_db, query_llm,create_new_db, display_base64_image, process_html
import os
from flask_sqlalchemy import SQLAlchemy
from models import db, Projects
from llm_utils import classify_intent, casual_conversation_agent, run_generate_report

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_DIR = "./documents/"
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
    project_name = data.get('project_name')
    retriever = connect_db(project_name)
    print("User question: ", question)

    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    intent = classify_intent(question)

    if intent == "casual": 
        print("Casual")
        result = casual_conversation_agent(question)
        return jsonify(text_response = result)
    
    elif intent == "generate_report": 
        print("generating report")
        retriever = connect_db(project_name)
        doc_url = url_for('static', filename='documents/output.docx', _external = True)
        return jsonify(text_response = run_generate_report(retriever, project_name), doc_url = doc_url)

    elif intent == "get_data": 
        print("getting data")
        result = query_llm(retriever, question)
        image_response = result["image_response"]

        return jsonify(text_response = result["text_response"], images=[f"data:image/png;base64,{img}" for img in image_response])
    


@app.route('/create_project', methods = ['POST'])
def create_project(): 

    #user submits name of project
    data = request.get_json()
    project_name = data.get('project_name')

    if not project_name: 
        return jsonify({"error": "Missing project name."}), 400

    #creates databases for the project (vector and doc store)
    create_new_db(project_name)

    return jsonify({
        "message": f"Successfully created project: {project_name}"
    }), 200

@app.route('/connect_to_db', methods = ['POST'])
def connect_to_db(): 
    data = request.get_json()
    project_name = data.get('project_name')
    print("This is the project name: ", project_name)

    try: 
        connect_db(project_name) #return retriever for a project
        print("DB Connected Successfully")

    except Exception as e: 

        return jsonify({
            "message": f"Failed to connect to project: {project_name}"
        }), 400


    return jsonify({
        "message": f"Successfully connected to project: {project_name}"
    }), 200

basedir = os.path.abspath(os.path.dirname(__file__))
db_folder = os.path.join(basedir, 'db')
os.makedirs(db_folder, exist_ok=True)

db_path = os.path.join(db_folder, 'project_db.sqlite')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

#to get names of all the projects for the sidebar
@app.route('/get_project_names', methods = ['GET'])
def get_project_names(): 
    try:
        projects = Projects.query.all()
        data = [{
            "id": project.id,
            "name": project.name,
            "file_path": project.file_path
        } for project in projects]

        return jsonify({"projects": data}), 200
    
    except Exception as e: 
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)