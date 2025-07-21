import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


#this is purely just for the sidebar to list out all the project names
class Projects(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)


class Reports(db.Model):
    __tablename__ = "reports"
    id = db.Column(db.Integer, primary_key=True)
    project_name = db.Column(db.String(255), unique=True, nullable=False)  # Tie report to project
    summary = db.Column(db.Text, nullable=True)
    introduction = db.Column(db.Text, nullable=True)
    objectives = db.Column(db.Text, nullable=True)
    methodology = db.Column(db.Text, nullable=True)
    results = db.Column(db.Text, nullable=True)
    conclusion = db.Column(db.Text, nullable=True)