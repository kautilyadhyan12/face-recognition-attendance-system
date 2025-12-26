# models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class Department(db.Model, UserMixin):
    __tablename__ = "department"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

    users = db.relationship('User', backref='department', lazy=True)
    subjects = db.relationship('Subject', backref='department', lazy=True)
    students = db.relationship('Student', backref='department', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_id(self):
        return f"dept:{self.id}"


class User(db.Model, UserMixin):
    """Professor user"""
    __tablename__ = "user"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(128), nullable=False)
    prof_id = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(32), default='professor')
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), nullable=False)
    password_change_required = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    subjects = db.relationship('Subject', backref='professor', lazy=True)
    students = db.relationship('Student', backref='professor_user', lazy=True)

    def set_password(self, pw):
        self.password_hash = generate_password_hash(pw)

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    def get_id(self):
        return f"user:{self.id}"


class Subject(db.Model):
    __tablename__ = "subject"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256))
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), nullable=False)
    professor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    students = db.relationship('Student', backref='subject', lazy=True)
    sessions = db.relationship('ClassSession', backref='subject', lazy=True)


class Student(db.Model, UserMixin):
    __tablename__ = "student"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(256))
    roll = db.Column(db.String(64), nullable=False)
    branch = db.Column(db.String(128))
    course = db.Column(db.String(128))
    department_id = db.Column(db.Integer, db.ForeignKey('department.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'))
    professor_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    eligible_override = db.Column(db.Boolean, default=False)

    # No unique constraints - same student can be in multiple departments and subjects
    # Relationships
    attendances = db.relationship('Attendance', backref='student_ref', lazy=True)

    def get_id(self):
        return f"student:{self.id}"


class ClassSession(db.Model):
    __tablename__ = "class_session"
    id = db.Column(db.Integer, primary_key=True)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'))
    date = db.Column(db.Date)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    attendances = db.relationship('Attendance', backref='class_session', lazy=True)


class Attendance(db.Model):
    __tablename__ = "attendance"
    id = db.Column(db.Integer, primary_key=True)
    class_session_id = db.Column(db.Integer, db.ForeignKey('class_session.id'))
    student_id = db.Column(db.Integer, db.ForeignKey('student.id'))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    status = db.Column(db.String(32), default='present')
    confidence = db.Column(db.Float, nullable=True)
    reason = db.Column(db.Text, nullable=True)
    edited = db.Column(db.Boolean, default=False)

    student = db.relationship('Student', backref='attendance_records')