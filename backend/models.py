import os
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


#this is purely just for the sidebar to list out all the project names
class Projects(db.Model):
    __tablename__ = "projects"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), unique=True, nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
