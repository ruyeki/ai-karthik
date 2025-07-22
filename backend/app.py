from flask import Flask, request, jsonify, url_for, send_from_directory
from flask_cors import CORS
from helper_functions import process_pdf, summarize_content, connect_db, store_to_db, query_llm,create_new_db, display_base64_image, process_html
import os
from flask_sqlalchemy import SQLAlchemy
from models import db, Projects, Reports
from llm_utils import (
    classify_intent,
    casual_conversation_agent,
    run_generate_report,
    classify_edit_intent,
    edit_introduction_section,
    edit_objectives_section,
    edit_summary_section,
    edit_conclusion_section,
    edit_methodology_section,
    edit_results_section,
    generate_docx,
    formatting_agent,
    regenerate_report
)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

UPLOAD_DIR = "./documents/" #sets a folder path called documents in current directory
os.makedirs(UPLOAD_DIR, exist_ok=True) #creates documents folder if it doesnt alrdy exist


#this is the database that holds all the project names along with the generated reports for each project
basedir = os.path.abspath(os.path.dirname(__file__))
db_folder = os.path.join(basedir, 'db')
os.makedirs(db_folder, exist_ok=True)

db_path = os.path.join(db_folder, 'project_db.sqlite')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)


with app.app_context():

    #THIS CODE POPULATES THE PROJECTS TABLE (dont need it anymore)
    '''
    basedir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(basedir, "db")

    if not os.path.exists(base_path):
        print(f"Folder not found: {base_path}")
    else:
        for folder_name in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder_name)
            if os.path.isdir(folder_path):
                print(f"Adding project: {folder_name}, Path: {folder_path}")

                project_data = Projects(
                    name=folder_name,
                    file_path=folder_path
                )

                db.session.add(project_data)

        db.session.commit()
        print("All projects added to the database.")
        '''
    db.create_all()

    #db.drop_all()
    #db.session.commit()


@app.route('/query', methods=['POST'])
def query():

    question = request.form.get("question")
    project_name = request.form.get('project_name')
    files = request.files.getlist('pdf')
    retriever = connect_db(project_name)
    print("User question: ", question)
    print("Uploaded pdf: ", files)


    #handle pdf uploads first: 

    if files:
        for file in files:
            if file and file.filename.endswith('.pdf'):
                print(f"Processing file: {file.filename}")
                file_path = os.path.join(UPLOAD_DIR, file.filename)
                file.save(file_path)

                # Process PDF
                texts, tables, images = process_pdf(file_path)

                # Summarize content
                text_summaries, table_summaries, image_summaries = summarize_content(texts, tables, images)

                # Store data in DB
                store_to_db(retriever, text_summaries, texts, image_summaries, images, tables, table_summaries)

        print("All files successfully processed!")



    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    intent = classify_intent(question)


    if intent == "casual": 
        print("Casual")
        result = casual_conversation_agent(question)
        return jsonify(text_response = result)

    if intent == "generate_report": 
        report_data = Reports.query.filter_by(project_name = project_name).first()

        if not report_data: #if no report has been generated yet, add it to the database
            
            print("no report found, generating a new one")
            full_report, summary, introduction, objectives, methodology, results_section, conclusion = run_generate_report(retriever, project_name)

            report_data = Reports(
                project_name = project_name,
                summary = summary,
                introduction = introduction,
                objectives = objectives,
                methodology = methodology,
                results = results_section,
                conclusion = conclusion
            )

            db.session.add(report_data)
            db.session.commit()

        
        else: #if the report already exists in the database, return it
            print("report found, displaying to ui...")
            full_report = f"""
                    {report_data.summary}\n
                    \n
                    {report_data.introduction}\n
                    \n
                    {report_data.objectives}\n
                    \n
                    {report_data.methodology}\n
                    \n
                    {report_data.results}\n
                    \n
                    {report_data.conclusion}\n
                    \n
                    """
            generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology, report_data.results, report_data.conclusion)
            
            #check for intent to edit HERE 

            edit_intent = classify_edit_intent(question)
            print(f"editing {edit_intent}")

            if edit_intent == "introduction": 
                new_introduction = edit_introduction_section(retriever, question, report_data.introduction)
                formatted_introduction = formatting_agent("Introduction", new_introduction)
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {formatted_introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                
                report_data.introduction = formatted_introduction
                db.session.commit()

                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results,report_data.conclusion)
                print("introduction successfully edited")

            elif edit_intent == "summary": 
                new_summary = edit_summary_section(retriever, question, report_data.summary)
                formatted_summary = formatting_agent("Summary", new_summary)
                full_report = f"""
                        {formatted_summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                
                report_data.summary = formatted_summary
                db.session.commit()
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results,report_data.conclusion)
                print("summary successfully edited")
                
            elif edit_intent == "conclusion": 
                new_conclusion = edit_conclusion_section(retriever, question, report_data.conclusion)
                formatted_conclusion = formatting_agent("Conclusion", new_conclusion)
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {formatted_conclusion}\n
                        \n
                        """
                
                report_data.conclusion = formatted_conclusion
                db.session.commit()
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results,report_data.conclusion)
                print("conclusion successfully edited")

            elif edit_intent == "methodology": 
                new_methodology = edit_methodology_section(retriever, question, report_data.conclusion, project_name)
                formatted_methodology = formatting_agent("Methodology", new_methodology)
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {formatted_methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                
                report_data.methodology = formatted_methodology
                db.session.commit()
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results, report_data.conclusion)
                print("methodology successfully edited")

            elif edit_intent == "results": 
                new_results = edit_results_section(retriever, question, report_data.results, project_name)
                formatted_results = formatting_agent("Results", new_results)
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {formatted_results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                
                report_data.results = formatted_results
                db.session.commit()
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results, report_data.conclusion)
                print("results successfully edited")
            
            elif edit_intent == "objectives": 
                new_objectives = edit_objectives_section(retriever, question, report_data.objectives)
                formatted_objectives = formatting_agent("Objectives", new_objectives)
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {formatted_objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                
                report_data.objectives = formatted_objectives
                db.session.commit()
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results, report_data.conclusion)
                print("objectives successfully edited")      

            elif edit_intent == "regenerate": 
                full_report, summary, introduction, objectives, methodology, results_section, conclusion = run_generate_report(retriever, project_name)

                summary = formatting_agent("Summary", summary)
                introduction = formatting_agent("Introduction", introduction)
                methodology = formatting_agent("Methodology", methodology)
                results_section = formatting_agent("Results", results_section)
                conclusion = formatting_agent("Conclusion", conclusion)

                report_data.summary = summary
                report_data.introduction = introduction
                report_data.methodology = methodology
                report_data.results_section = results_section
                report_data.conclusion = conclusion
                
                db.session.commit()

                generate_docx(summary, introduction, objectives, methodology,  results_section, conclusion)
                print("report successfully regenerated!")     

            
            else: #if the edit intent was none, return whats already in the database
                full_report = f"""
                        {report_data.summary}\n
                        \n
                        {report_data.introduction}\n
                        \n
                        {report_data.objectives}\n
                        \n
                        {report_data.methodology}\n
                        \n
                        {report_data.results}\n
                        \n
                        {report_data.conclusion}\n
                        \n
                        """
                generate_docx(report_data.summary, report_data.introduction, report_data.objectives, report_data.methodology,  report_data.results, report_data.conclusion)  

        doc_url = url_for('static', filename='documents/output.docx', _external = True)
        return jsonify(text_response = full_report, doc_url = doc_url)

    if intent == "get_data": 
        print("getting data")
        result = query_llm(retriever, question)
        image_response = result["image_response"]
        #print(f"This is images: {image_response}")

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
    file_path = os.path.join(os.path.dirname(__file__), 'db', project_name)

    new_project = Projects(
        name = project_name,
        file_path = file_path
    )

    db.session.add(new_project)
    db.session.commit()

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