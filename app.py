
import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User, Department, Subject, Student, ClassSession, Attendance
from utils import ensure_instance_dirs
from face_pipeline import FacePipeline
from datetime import datetime, date, timedelta
import pandas as pd
import io
import time
import threading
from gpu_setup import setup_gpu, force_cpu
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from markupsafe import escape
from sqlalchemy import func

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, 'instance')
UPLOAD_DIR = os.path.join(INSTANCE_DIR, 'uploads')
ENC_DIR = os.path.join(INSTANCE_DIR, 'encodings')

ensure_instance_dirs([INSTANCE_DIR, UPLOAD_DIR, ENC_DIR])

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(INSTANCE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')  

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = 'index'
login_manager.init_app(app)

face_pipe = FacePipeline(enc_dir=ENC_DIR, upload_dir=UPLOAD_DIR)

# Training progress tracking
training_progress = {}

# Create DB on first run
with app.app_context():
    db.create_all()

@app.context_processor
def inject_datetime():
    return {'datetime': datetime}

@login_manager.user_loader
def load_user(user_id):
    if not user_id:
        return None
    try:
        kind, id_ = user_id.split(":", 1)
    except Exception:
        return None
    if kind == "dept":
        return Department.query.get(int(id_))
    if kind == "user":
        return User.query.get(int(id_))
    if kind == "student":
        return Student.query.get(int(id_))
    return None

def safe_text(s, maxlen=500):
    if s is None:
        return None
    s = str(s).strip()
    if len(s) > maxlen:
        s = s[:maxlen]
    return escape(s)

# Template filter for image count

@app.template_filter('get_image_count')
def get_image_count(student_or_roll, subject_id):
    import os
    from glob import glob
    
    # Handle both Student objects and roll strings
    if hasattr(student_or_roll, 'roll'):
        # If it's a Student object, get the roll
        roll = student_or_roll.roll
    else:
        # If it's already a roll string, use it directly
        roll = student_or_roll
    
    image_dir = os.path.join('instance', 'uploads', str(subject_id), str(roll))
    if os.path.exists(image_dir):
        return len(glob(os.path.join(image_dir, '*.jpg')))
    return 0

# ---------- Public pages ----------
@app.route('/')
def index():
    return render_template('index.html')

# -------------------------
# Department Routes
# -------------------------
@app.route('/department/register', methods=['GET', 'POST'])
def register_department():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '').strip()

        if not name or not password:
            flash('Department name and password are required.', 'danger')
            return redirect(url_for('register_department'))

        if Department.query.filter_by(name=name).first():
            flash('Department already exists. Please log in.', 'warning')
            return redirect(url_for('login_department'))

        dept = Department(name=name)
        dept.set_password(password)
        db.session.add(dept)
        db.session.commit()

        flash('Department registered successfully. You can now log in.', 'success')
        return redirect(url_for('login_department'))

    return render_template('register_department.html')

@app.route('/department/login', methods=['GET', 'POST'])
def login_department():
    # Get all departments for the dropdown
    all_departments = Department.query.all()
    
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        password = request.form.get('password', '').strip()

        # Case-insensitive department lookup
        dept = Department.query.filter(Department.name.ilike(name)).first()
        if dept and dept.check_password(password):
            login_user(dept)
            flash(f'Welcome {dept.name} Department!', 'success')
            return redirect(url_for('department_dashboard'))
        flash('Invalid credentials.', 'danger')
    
    return render_template('login_department.html', departments=all_departments)

@app.route('/department/logout')
@login_required
def department_logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/department/dashboard')
@login_required
def department_dashboard():
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    subjects = Subject.query.filter_by(department_id=current_user.id).all()
    professors = User.query.filter_by(department_id=current_user.id).all()

    prof_stats = []
    for prof in professors:
        subj_count = Subject.query.filter_by(professor_id=prof.id).count()
        class_count = ClassSession.query.join(Subject).filter(Subject.professor_id == prof.id).count()
        prof_stats.append({
            'prof': prof,
            'subjects': subj_count,
            'classes': class_count
        })

    return render_template('department_dashboard.html',
                           department=current_user,
                           subjects=subjects,
                           professors=professors,
                           prof_stats=prof_stats)

@app.route('/department/add_professor', methods=['POST'])
@login_required
def department_add_professor():
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    name = request.form.get('name', '').strip()
    prof_id = request.form.get('prof_id', '').strip()
    temp_password = request.form.get('temp_password', '').strip()  

    if not name or not prof_id or not temp_password:  
        flash('Professor name, ID and temporary password are required.', 'danger')
        return redirect(url_for('department_dashboard'))

    if User.query.filter_by(prof_id=prof_id).first():
        flash('Professor ID already exists.', 'danger')
        return redirect(url_for('department_dashboard'))

    prof = User(username=name, prof_id=prof_id, department_id=current_user.id)
    prof.set_password(temp_password)
    prof.password_change_required = True  
    db.session.add(prof)
    db.session.commit()
    
    flash(f'Professor {name} added successfully. They must change their password on first login.', 'success')
    return redirect(url_for('department_dashboard'))

@app.route('/department/add_subject', methods=['POST'])
@login_required
def department_add_subject():
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    name = request.form.get('name', '').strip()
    prof_id = request.form.get('prof_id', '').strip()

    if not name:
        flash('Subject name is required.', 'danger')
        return redirect(url_for('department_dashboard'))

    subj = Subject(name=name, department_id=current_user.id)
    if prof_id:
        prof = User.query.filter_by(prof_id=prof_id, department_id=current_user.id).first()
        if prof:
            subj.professor_id = prof.id
        else:
            flash('Professor not found.', 'warning')

    db.session.add(subj)
    db.session.commit()
    flash(f'Subject "{name}" added successfully.', 'success')
    return redirect(url_for('department_dashboard'))

@app.route('/department/delete_professor/<int:prof_id>', methods=['POST'])
@login_required
def delete_professor(prof_id):
    """Delete a professor and ALL associated data"""
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    prof = User.query.get_or_404(prof_id)
    
    # Verify the professor belongs to the current department
    if prof.department_id != current_user.id:
        flash('Not authorized to delete this professor.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    # Get statistics for confirmation message
    subject_count = Subject.query.filter_by(professor_id=prof.id).count()
    student_count = Student.query.filter_by(professor_id=prof.id).count()
    recent_sessions = ClassSession.query.join(Subject).filter(
        Subject.professor_id == prof.id
    ).count()
    
    professor_name = prof.username
    
    try:
        # Safety check - warn if professor has active classes
        if recent_sessions > 0:
            flash(f'Warning: Professor {professor_name} has conducted {recent_sessions} class sessions. All data will be deleted.', 'warning')
        
        # 1. Get all subjects taught by this professor
        subjects_taught = Subject.query.filter_by(professor_id=prof.id).all()
        
        # 2. For each subject, delete all students and their data
        students_deleted = 0
        for subject in subjects_taught:
            # Delete all students in this subject
            students_in_subject = Student.query.filter_by(subject_id=subject.id).all()
            
            for student in students_in_subject:
                # Delete student images
                folder = os.path.join('instance', 'uploads', str(subject.id), student.roll)
                if os.path.exists(folder):
                    import shutil
                    shutil.rmtree(folder, ignore_errors=True)
                
                # Delete attendance records
                Attendance.query.filter_by(student_id=student.id).delete()
                
                # Delete the student
                db.session.delete(student)
                students_deleted += 1
            
            # Delete class sessions for this subject
            ClassSession.query.filter_by(subject_id=subject.id).delete()
            
            # Delete the subject itself
            db.session.delete(subject)
        
        # 3. Delete the professor
        db.session.delete(prof)
        db.session.commit()
        
        flash(f'Professor {professor_name} deleted successfully. Removed {subject_count} subjects and {students_deleted} students with all their data.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting professor: {str(e)}', 'danger')
    
    return redirect(url_for('department_dashboard'))

@app.route('/department/subject/<int:subject_id>/assign_professor', methods=['POST'])
@login_required
def assign_professor_to_subject(subject_id):
    """Assign a professor to a subject"""
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subject = Subject.query.get_or_404(subject_id)
    
    # Verify the subject belongs to the current department
    if subject.department_id != current_user.id:
        flash('Not authorized to modify this subject.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    prof_id = request.form.get('prof_id', '').strip()
    
    if not prof_id:
        flash('Please select a professor.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    # Find professor in current department
    prof = User.query.filter_by(prof_id=prof_id, department_id=current_user.id).first()
    
    if not prof:
        flash('Professor not found in your department.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    # Assign professor to subject
    subject.professor_id = prof.id
    db.session.commit()
    
    flash(f'Professor {prof.username} assigned to {subject.name} successfully.', 'success')
    return redirect(url_for('department_dashboard'))

@app.route('/department/subject/<int:subject_id>/remove_professor', methods=['POST'])
@login_required
def remove_professor_from_subject(subject_id):
    """Remove professor assignment from a subject"""
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subject = Subject.query.get_or_404(subject_id)
    
    # Verify the subject belongs to the current department
    if subject.department_id != current_user.id:
        flash('Not authorized to modify this subject.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    if not subject.professor:
        flash('No professor assigned to this subject.', 'warning')
        return redirect(url_for('department_dashboard'))
    
    professor_name = subject.professor.username
    subject.professor_id = None
    db.session.commit()
    
    flash(f'Professor {professor_name} removed from {subject.name}.', 'success')
    return redirect(url_for('department_dashboard'))


@app.route('/department/delete_subject/<int:subject_id>', methods=['POST'])
@login_required
def delete_subject(subject_id):
    """Delete a subject and ALL associated data"""
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subject = Subject.query.get_or_404(subject_id)
    
    # Verify the subject belongs to the current department
    if subject.department_id != current_user.id:
        flash('Not authorized to delete this subject.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    subject_name = subject.name
    
    try:
        #  Delete all students in this subject and their data
        students_in_subject = Student.query.filter_by(subject_id=subject.id).all()
        students_deleted = 0
        
        for student in students_in_subject:
            # Delete student images
            folder = os.path.join('instance', 'uploads', str(subject.id), student.roll)
            if os.path.exists(folder):
                import shutil
                shutil.rmtree(folder, ignore_errors=True)
            
            # Delete attendance records
            Attendance.query.filter_by(student_id=student.id).delete()
            
            # Delete the student
            db.session.delete(student)
            students_deleted += 1
        
        # 2. Delete class sessions for this subject
        sessions_deleted = ClassSession.query.filter_by(subject_id=subject.id).delete()
        
        # 3. Delete the subject itself
        db.session.delete(subject)
        db.session.commit()
        
        flash(f'Subject "{subject_name}" deleted successfully. Removed {students_deleted} students and {sessions_deleted} class sessions.', 'success')
        
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting subject: {str(e)}', 'danger')
    
    return redirect(url_for('department_dashboard'))


@app.route('/department/analytics')
@login_required
def department_analytics():
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    department = current_user
    subjects = Subject.query.filter_by(department_id=department.id).all()
    analytics = []

    for subj in subjects:
        total_classes = ClassSession.query.filter_by(subject_id=subj.id).count()
        students = Student.query.filter_by(subject_id=subj.id).all()
        for st in students:
            attended = Attendance.query.join(ClassSession).filter(
                Attendance.student_id == st.id,
                ClassSession.subject_id == subj.id
            ).count()
            perc = (attended / total_classes * 100) if total_classes > 0 else 0
            eligible = perc >= 75 or getattr(st, "eligible_override", False)
            analytics.append({
                'student': st,
                'subject': subj,
                'total_classes': total_classes,
                'attended': attended,
                'perc': perc,
                'eligible': eligible
            })

    return render_template('department_analytics.html',
                           department=department,
                           analytics=analytics)

@app.route('/department/override/<int:student_id>', methods=['POST'])
@login_required
def override_student_eligibility(student_id):
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    st = Student.query.get_or_404(student_id)
    if st.department_id != current_user.id:
        flash('Student does not belong to your department.', 'danger')
        return redirect(url_for('department_analytics'))

    st.eligible_override = True
    db.session.commit()
    flash(f'{st.name} marked as eligible.', 'success')
    return redirect(url_for('department_analytics'))

# ======== DEPARTMENT STUDENT MANAGEMENT ROUTES ========

@app.route('/department/subject/<int:subject_id>/students')
@login_required
def department_manage_students(subject_id):
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subj = Subject.query.get_or_404(subject_id)
    if subj.department_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    students = Student.query.filter_by(subject_id=subj.id).all()
    return render_template('department_manage_students.html', 
                         subject=subj, 
                         students=students,
                         department=current_user)

@app.route('/department/subject/<int:subject_id>/add_student', methods=['POST'])
@login_required
def department_add_student(subject_id):
    if not isinstance(current_user, Department):
        return jsonify({'status':'error','message':'Access denied'}), 403
    
    subj = Subject.query.get_or_404(subject_id)
    if subj.department_id != current_user.id:
        return jsonify({'status':'error','message':'Not authorized'}), 403

    data = request.get_json(force=True)
    name = data.get('name','').strip()
    roll = data.get('roll','').strip()
    branch = data.get('branch','').strip()
    course = data.get('course','').strip()

    if not name or not roll:
        return jsonify({'status':'error','message':'Missing fields'}), 400
    
    # Check if roll already exists in THIS EXACT department + subject combination
    existing_student = Student.query.filter_by(
        roll=roll, 
        department_id=current_user.id,
        subject_id=subject_id
    ).first()
    
    if existing_student:
        return jsonify({'status':'error','message':'This student is already enrolled in this subject'}), 400

    # Allow creation - same student can exist in multiple departments/subjects
    student = Student(name=name, roll=roll, branch=branch, course=course,
                      subject_id=subject_id,
                      department_id=current_user.id)
    db.session.add(student)
    db.session.commit()
    return jsonify({'status':'ok','student_id': student.id})

@app.route('/department/subject/<int:subject_id>/capture_photo', methods=['POST'])
@login_required
def department_capture_photo(subject_id):
    if not isinstance(current_user, Department):
        return jsonify({'status':'error','message':'Access denied'}), 403
    
    data = request.get_json(force=True)
    sid = data.get('student_id')
    image_data = data.get('image')
    st = Student.query.get_or_404(sid)
    if st.subject_id != subject_id:
        return jsonify({'status':'error','message':'Student mismatch'}), 400
    saved = face_pipe.save_student_image(subject_id, st.roll, image_data)
    return jsonify({'status':'ok','path': saved})

@app.route('/department/subject/<int:subject_id>/delete_student/<int:student_id>', methods=['POST'])
@login_required
def department_delete_student(subject_id, student_id):
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    st = Student.query.get_or_404(student_id)
    if st.subject_id != subject_id or st.department_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('department_manage_students', subject_id=subject_id))

    # Delete student images
    folder = os.path.join('instance', 'uploads', str(subject_id), st.roll)
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    
    # Delete attendance records
    Attendance.query.filter_by(student_id=st.id).delete()
    db.session.delete(st)
    db.session.commit()
    flash('Student deleted successfully.', 'success')
    return redirect(url_for('department_manage_students', subject_id=subject_id))

@app.route('/department/subject/<int:subject_id>/train')
@login_required
def department_train_model(subject_id):
    if not isinstance(current_user, Department):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subj = Subject.query.get_or_404(subject_id)
    if subj.department_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('department_dashboard'))
    
    students = Student.query.filter_by(subject_id=subj.id).all()
    return render_template('department_train.html', 
                         subject=subj, 
                         students=students,
                         department=current_user)

@app.route('/department/subject/<int:subject_id>/train/start', methods=['POST'])
@login_required
def department_start_training(subject_id):
    """Department version of training"""
    if not isinstance(current_user, Department):
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403
    
    subj = Subject.query.get_or_404(subject_id)
    if subj.department_id != current_user.id:
        return jsonify({'status': 'error', 'message': 'Not authorized'}), 403
    
    # Use the same training logic but with department authorization
    data = request.get_json(force=True)
    mode = data.get('mode', 'high_quality')
    
    # Initialize progress
    training_progress[subject_id] = {
        'progress': 0, 
        'status': f'Initializing {mode.replace("_", " ").title()} training - this may take 45-90 seconds...', 
        'done': False,
        'current_student': '',
        'students_processed': 0,
        'total_students': 0
    }

    def run_training():
        try:
            # Same training logic as professor version
            res = face_pipe.train_subject_optimized(subject_id, mode)
            training_progress[subject_id].update({
                'progress': 100,
                'status': ' Department training complete' if res.get('status') == 'success' else f' Department training failed: {res.get("message", "Unknown error")}',
                'done': True,
                'result': res
            })
        except Exception as e:
            training_progress[subject_id].update({
                'progress': 100,
                'status': f' Department training failed: {str(e)}',
                'done': True
            })

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'subject_id': subject_id})

@app.route('/department/subject/<int:subject_id>/train/status')
@login_required
def department_get_training_status(subject_id):
    info = training_progress.get(subject_id, {'progress': 0, 'status': 'Not started', 'done': False})
    return jsonify(info)

# ======== END DEPARTMENT STUDENT MANAGEMENT ROUTES ========

# -------------------------
# Professor Routes
# -------------------------

@app.route('/professor/login', methods=['GET','POST'])
def login_professor():
   
    all_departments = Department.query.all()
    all_professors = User.query.all()
    
    if request.method == 'POST':
        dept_name = request.form.get('department','').strip()
        prof_id = request.form.get('prof_id','').strip()
        username = request.form.get('username','').strip()
        password = request.form.get('password','').strip()

        
        dept = Department.query.filter(Department.name.ilike(dept_name)).first()
        if not dept:
            flash('Department not found.', 'danger')
            return redirect(url_for('login_professor'))

        
        prof = User.query.filter(
            User.prof_id.ilike(prof_id), 
            User.department_id == dept.id
        ).first()
        if not prof:
            flash('Professor ID invalid.', 'danger')
            return redirect(url_for('login_professor'))

        
        if prof.username.lower() != username.lower():
            flash('Professor name does not match.', 'danger')
            return redirect(url_for('login_professor'))

        if prof.check_password(password):
            login_user(prof)
            
            # Check if password change is required
            if prof.password_change_required:
                flash('Please set your new password.', 'info')
                return redirect(url_for('professor_first_login'))
            else:
                flash('Welcome Professor!', 'success')
                return redirect(url_for('prof_dashboard'))
        else:
            flash('Invalid credentials.', 'danger')

    return render_template('login_professor.html', 
                         departments=all_departments, 
                         professors=all_professors)

@app.route('/prof/logout')
@login_required
def prof_logout():
    logout_user()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

@app.route('/professor/dashboard')
@login_required
def prof_dashboard():
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    subjects = Subject.query.filter_by(professor_id=current_user.id).all()
    
    # Calculate all stats in Python
    total_students = sum(len(subject.students) for subject in subjects)
    total_classes = sum(len(subject.sessions) for subject in subjects)
    active_courses = len([s for s in subjects if s.professor_id])
    
    return render_template('prof_dashboard.html', 
                         subjects=subjects,
                         total_students=total_students,
                         total_classes=total_classes,
                         active_courses=active_courses)

# Student Management 
@app.route('/prof/<int:subject_id>/students/add')
@login_required
def add_student_page(subject_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))
    students = Student.query.filter_by(subject_id=subj.id).all()
    return render_template('register_student.html', subject=subj, students=students)

@app.route('/prof/<int:subject_id>/students')
@login_required
def students_page(subject_id):
    """Main students management page"""
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))
    students = Student.query.filter_by(subject_id=subj.id).all()
    return render_template('register_student.html', subject=subj, students=students)

@app.route('/prof/<int:subject_id>/add_student', methods=['POST'])
@login_required
def add_student(subject_id):
    if not isinstance(current_user, User):
        return jsonify({'status':'error','message':'Access denied'}), 403
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        return jsonify({'status':'error','message':'Not your subject'}), 403

    data = request.get_json(force=True)
    name = data.get('name','').strip()
    roll = data.get('roll','').strip()
    branch = data.get('branch','').strip()
    course = data.get('course','').strip()

    if not name or not roll:
        return jsonify({'status':'error','message':'Missing fields'}), 400
    
    # Check if roll exists in this specific subject
    existing_student = Student.query.filter_by(
        roll=roll, 
        subject_id=subject_id
    ).first()
    
    if existing_student:
        return jsonify({'status':'error','message':'This student is already enrolled in this subject'}), 400

    student = Student(name=name, roll=roll, branch=branch, course=course,
                      subject_id=subject_id,
                      professor_id=current_user.id,
                      department_id=current_user.department_id)
    db.session.add(student)
    db.session.commit()
    return jsonify({'status':'ok','student_id': student.id})

@app.route('/prof/<int:subject_id>/delete_student/<int:student_id>', methods=['POST'])
@login_required
def delete_student(subject_id, student_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    st = Student.query.get_or_404(student_id)
    if st.subject_id != subject_id or st.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('add_student_page', subject_id=subject_id))

    # Delete student images
    folder = os.path.join('instance', 'uploads', str(subject_id), st.roll)
    if os.path.exists(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    
    # Delete attendance records
    Attendance.query.filter_by(student_id=st.id).delete()
    db.session.delete(st)
    db.session.commit()
    flash('Student deleted successfully.', 'success')
    return redirect(url_for('add_student_page', subject_id=subject_id))

@app.route('/prof/<int:subject_id>/students/manage')
@login_required  
def manage_students(subject_id):
    return redirect(url_for('add_student_page', subject_id=subject_id))

# Image Capture
@app.route('/prof/<int:subject_id>/capture_photo', methods=['POST'])
@login_required
def capture_photo(subject_id):
    if not isinstance(current_user, User):
        return jsonify({'status':'error','message':'Access denied'}), 403
    data = request.get_json(force=True)
    sid = data.get('student_id')
    image_data = data.get('image')
    st = Student.query.get_or_404(sid)
    if st.subject_id != subject_id:
        return jsonify({'status':'error','message':'Student mismatch'}), 400
    saved = face_pipe.save_student_image(subject_id, st.roll, image_data)
    return jsonify({'status':'ok','path': saved})

# Training
@app.route('/prof/<int:subject_id>/train')
@login_required
def train_page(subject_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))
    students = Student.query.filter_by(subject_id=subj.id).all()
    return render_template('train.html', subject=subj, students=students)

@app.route('/prof/<int:subject_id>/train', methods=['POST'])
@login_required
def train_subject(subject_id):
    if not isinstance(current_user, User):
        return jsonify({'status':'error','message':'Access denied'}), 403
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        return jsonify({'status':'error','message':'Not your subject'}), 403
    
    # Get training mode from request
    data = request.get_json(force=True) if request.get_json(force=True) else {}
    mode = data.get('mode', 'high_quality')  # high_quality, faster_quality
    
    # This will now use the optimized version
    res = face_pipe.train_subject_optimized(subject_id, mode)
    
    return jsonify(res)

# -----------------------------------
# Asynchronous Training with Progress Status
# -----------------------------------

@app.route('/prof/<int:subject_id>/train/start', methods=['POST'])
@login_required
def start_training(subject_id):
    """Run model training asynchronously with GPU setup"""
    if not isinstance(current_user, User):
        return jsonify({'status': 'error', 'message': 'Access denied'}), 403

    # Initialize GPU setup
    from gpu_setup import setup_gpu, force_cpu
    
    # Try GPU first, fallback to CPU
    use_gpu = setup_gpu()
    if not use_gpu:
        force_cpu()
    
    data = request.get_json(force=True)
    mode = data.get('mode', 'high_quality')
    
    # Initialize progress with more detailed tracking
    training_progress[subject_id] = {
        'progress': 0, 
        'status': f'Initializing {mode.replace("_", " ").title()} training - this may take 45-90 seconds...', 
        'done': False,
        'current_student': '',
        'students_processed': 0,
        'total_students': 0
    }

    def run_training():
        try:
            subject_path = os.path.join('instance', 'uploads', str(subject_id))
            students = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
            total_students = len(students)
            
            training_progress[subject_id].update({
                'total_students': total_students,
                'status': f'Found {total_students} students to process...'
            })
            
            # Process students with detailed progress
            successful_students = 0
            for i, roll in enumerate(students):
                training_progress[subject_id].update({
                    'progress': int((i / total_students) * 80),  
                    'current_student': f'Processing {roll}...',
                    'students_processed': i,
                    'status': f'Processing student {i+1}/{total_students}: {roll}'
                })
                time.sleep(0.5)  
                
                
                successful_students += 1
            
            # Final encoding and saving
            training_progress[subject_id].update({
                'progress': 90,
                'status': 'Saving encodings...'
            })
            time.sleep(1)
            
            #  actual training
            res = face_pipe.train_subject_optimized(subject_id, mode)

            training_progress[subject_id].update({
                'progress': 100,
                'status': '✅ Training complete' if res.get('status') == 'success' else f'❌ Training failed: {res.get("message", "Unknown error")}',
                'done': True,
                'result': res
            })

        except Exception as e:
            training_progress[subject_id].update({
                'progress': 100,
                'status': f'❌ Error: {str(e)}',
                'done': True
            })

    t = threading.Thread(target=run_training, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'subject_id': subject_id, 'using_gpu': use_gpu})

@app.route('/prof/<int:subject_id>/train/status')
@login_required
def get_training_status(subject_id):
    info = training_progress.get(subject_id, {'progress': 0, 'status': 'Not started', 'done': False})
    return jsonify(info)

@app.route('/prof/<int:subject_id>/train_results')
@login_required
def train_results(subject_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))
    res = face_pipe.train_report(subject_id)
    return render_template('train_results.html', report=res, subject=subj)

# === DEBUG ROUTES  ===
@app.route('/debug/training/<int:subject_id>')
@login_required
def debug_training(subject_id):
    """Debug training setup"""
    subject_path = os.path.join('instance', 'uploads', str(subject_id))
    
    if not os.path.exists(subject_path):
        return jsonify({'error': 'Subject directory not found'})
    
    students = []
    for roll in os.listdir(subject_path):
        roll_path = os.path.join(subject_path, roll)
        if os.path.isdir(roll_path):
            images = list(Path(roll_path).glob('*.jpg'))
            students.append({
                'roll': roll,
                'image_count': len(images),
                'images': [str(img) for img in images[:3]]  # First 3 images
            })
    
    return jsonify({
        'subject_id': subject_id,
        'subject_path': subject_path,
        'students_found': len(students),
        'students': students
    })

@app.route('/debug/training-status/<int:subject_id>')
@login_required
def debug_training_status(subject_id):
    """Check training status and requirements"""
    import os
    from pathlib import Path
    
    subject_path = os.path.join('instance', 'uploads', str(subject_id))
    enc_path = os.path.join('instance', 'encodings', f'subject_{subject_id}_enc.pkl')
    
    result = {
        'subject_id': subject_id,
        'subject_path_exists': os.path.exists(subject_path),
        'model_exists': os.path.exists(enc_path),
        'students': []
    }
    
    if os.path.exists(subject_path):
        students = []
        for roll in os.listdir(subject_path):
            roll_path = os.path.join(subject_path, roll)
            if os.path.isdir(roll_path):
                images = list(Path(roll_path).glob('*.jpg'))
                students.append({
                    'roll': roll,
                    'image_count': len(images),
                    'has_enough_images': len(images) >= 8,
                    'image_files': [img.name for img in images[:3]]  # First 3 filenames
                })
        result['students'] = students
        result['total_students'] = len(students)
        result['ready_for_training'] = len([s for s in students if s['has_enough_images']])
    
    return jsonify(result)

@app.route('/debug/attendance/<int:subject_id>/<date>')
@login_required
def debug_attendance_date(subject_id, date):
    """Debug route to check attendance for a specific date"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        attendance_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    # Find class session for this date
    session = ClassSession.query.filter_by(
        subject_id=subject_id, 
        date=attendance_date
    ).first()
    
    if not session:
        return jsonify({'error': 'No class session found for this date'})
    
    #  all attendance records for this session
    attendance_records = Attendance.query.filter_by(
        class_session_id=session.id
    ).join(Student).all()
    
    records_data = []
    for record in attendance_records:
        records_data.append({
            'attendance_id': record.id,
            'student_id': record.student.id,
            'student_name': record.student.name,
            'student_roll': record.student.roll,
            'status': record.status,
            'confidence': record.confidence,
            'timestamp': record.timestamp.isoformat() if record.timestamp else None
        })
    
    # all students in the subject for comparison
    all_students = Student.query.filter_by(subject_id=subject_id).all()
    student_list = [{'id': s.id, 'name': s.name, 'roll': s.roll} for s in all_students]
    
    return jsonify({
        'session_id': session.id,
        'date': session.date.isoformat(),
        'attendance_records': records_data,
        'total_students_in_subject': len(student_list),
        'all_students': student_list,
        'present_count': len([r for r in records_data if r['status'] == 'present'])
    })

@app.route('/debug/calendar_data/<int:subject_id>')
@login_required
def debug_calendar_data(subject_id):
    """Debug route to check calendar data"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    #  all class sessions
    sessions = ClassSession.query.filter_by(subject_id=subject_id).all()
    session_data = []
    
    for session in sessions:
        attendance_count = Attendance.query.filter_by(class_session_id=session.id).count()
        session_data.append({
            'date': session.date.isoformat(),
            'session_id': session.id,
            'attendance_records_count': attendance_count,
            'start_time': session.start_time.isoformat() if session.start_time else None
        })
    
    #  all students
    students = Student.query.filter_by(subject_id=subject_id).all()
    student_data = [{'id': s.id, 'name': s.name, 'roll': s.roll} for s in students]
    
    return jsonify({
        'subject_id': subject_id,
        'total_sessions': len(sessions),
        'total_students': len(students),
        'sessions': session_data,
        'students': student_data
    })

@app.route('/debug/attendance_db_check/<int:subject_id>/<date>')
@login_required
def debug_attendance_db_check(subject_id, date):
    """Debug route to check database integrity for attendance records"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        attendance_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    #  class session
    session = ClassSession.query.filter_by(
        subject_id=subject_id, 
        date=attendance_date
    ).first()
    
    if not session:
        return jsonify({'error': 'No session found'})
    
    #  raw database records
    attendance_records = Attendance.query.filter_by(
        class_session_id=session.id
    ).join(Student).all()
    
    records_data = []
    for record in attendance_records:
        records_data.append({
            'id': record.id,
            'student_id': record.student.id,
            'student_name': record.student.name,
            'student_roll': record.student.roll,
            'status': record.status,
            'confidence': record.confidence,
            'timestamp': record.timestamp.isoformat() if record.timestamp else None,
            'edited': record.edited
        })
    
    #  all students in subject for comparison
    all_students = Student.query.filter_by(subject_id=subject_id).all()
    student_list = [{'id': s.id, 'name': s.name, 'roll': s.roll} for s in all_students]
    
    return jsonify({
        'session_id': session.id,
        'date': session.date.isoformat(),
        'attendance_records_found': len(records_data),
        'total_students_in_subject': len(student_list),
        'attendance_records': records_data,
        'all_students': student_list,
        'missing_students': [s for s in student_list if s['id'] not in [r['student_id'] for r in records_data]]
    })


@app.route('/debug/attendance_check/<int:subject_id>/<date>')
@login_required
def debug_attendance_check(subject_id, date):
    """Debug route to check attendance data for a specific date"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        attendance_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    #  class session
    session = ClassSession.query.filter_by(
        subject_id=subject_id, 
        date=attendance_date
    ).first()
    
    if not session:
        return jsonify({'error': 'No session found for this date'})
    
    #  all attendance records for this session
    attendance_records = Attendance.query.filter_by(
        class_session_id=session.id
    ).join(Student).all()
    
    # all students in the subject
    all_students = Student.query.filter_by(subject_id=subject_id).all()
    
    records_data = []
    for record in attendance_records:
        records_data.append({
            'student_id': record.student.id,
            'student_name': record.student.name,
            'student_roll': record.student.roll,
            'status': record.status,
            'confidence': record.confidence,
            'timestamp': record.timestamp.isoformat() if record.timestamp else None
        })
    
    all_students_data = [{'id': s.id, 'name': s.name, 'roll': s.roll} for s in all_students]
    
    return jsonify({
        'session_id': session.id,
        'date': session.date.isoformat(),
        'attendance_records_found': len(records_data),
        'total_students_in_subject': len(all_students_data),
        'attendance_records': records_data,
        'all_students': all_students_data,
        'missing_students': [s for s in all_students_data if s['id'] not in [r['student_id'] for r in records_data]]
    })


# === END DEBUG ROUTES ===

# Attendance Session
@app.route('/prof/<int:subject_id>/start_session', methods=['POST'])
@login_required
def start_session(subject_id):
    if not isinstance(current_user, User):
        return jsonify({'status':'error','message':'Access denied'}), 403
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        return jsonify({'status':'error','message':'Not your subject'}), 403
    now = datetime.utcnow()
    sess = ClassSession(subject_id=subj.id, date=now.date(), start_time=now)
    db.session.add(sess)
    db.session.commit()
    return jsonify({'status':'ok','session_id': sess.id})

@app.route('/prof/<int:subject_id>/mark_attendance')
@login_required
def mark_attendance_page(subject_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))
    return render_template('mark_attendance.html', subject=subj)

@app.route('/prof/<int:subject_id>/recognize', methods=['POST'])
@login_required
def recognize_frame(subject_id):
    if not isinstance(current_user, User):
        return jsonify({'status':'error','message':'Access denied'}), 403
    
    data = request.get_json(force=True)
    image = data.get('image')
    session_id = data.get('session_id')
    liveness_data = data.get('liveness_data', {}) 
    
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        return jsonify({'status':'error','message':'Not your subject'}), 403
    
    # Check liveness data if provided
    if liveness_data:
        
        anti_spoofing_score = liveness_data.get('antiSpoofingScore', 0)
        real_person_score = liveness_data.get('realPersonScore', 0)
        spoofing_detected = liveness_data.get('spoofingDetected', False)
        
        # Reject if spoofing detected
        if spoofing_detected:
            return jsonify({'results': [{
                'status': 'spoofing_detected',
                'message': 'Spoofing detected - Attendance blocked'
            }]})
        
        # Require minimum anti-spoofing score
        if anti_spoofing_score < 60:  # 60% minimum
            return jsonify({'results': [{
                'status': 'low_anti_spoofing',
                'message': 'Insufficient anti-spoofing verification'
            }]})
    
    matches = face_pipe.recognize_in_subject(subject_id, image)
    today = datetime.utcnow().date()
    results = []

    for m in matches:
        # Handle warnings first
        if m.get('warning') == 'multiple_faces':
            results.append({'warning': 'multiple_faces', 'message': m.get('message', 'Multiple faces detected')})
            continue
            
        if m.get('warning') == 'no_model':
            results.append({'warning': 'no_model', 'message': m.get('message', 'No trained model found')})
            continue
            
        if m.get('error') == 'recognition_failed':
            results.append({'error': 'recognition_failed', 'message': m.get('message', 'Recognition failed')})
            continue

        roll = m.get('roll')
        confidence = m.get('confidence', 0.0)
        status = m.get('status', 'unknown')

        # Handle no face detected
        if status == 'no_face':
            results.append({'status': 'no_face', 'message': m.get('message', 'No face detected')})
            continue

        # Handle unknown face
        if status == 'unknown' or not roll:
            results.append({'status': 'unknown', 'message': m.get('message', 'Unknown face - Not in database')})
            continue

        # Case-insensitive student lookup with exact match
        st = Student.query.filter(
            Student.roll.ilike(roll), 
            Student.subject_id == subject_id
        ).first()
        
        if not st:
            results.append({'status': 'unknown', 'roll': roll, 'message': 'Student not found in database'})
            continue

        # For low confidence matches, 
        if status == 'low_confidence':
            results.append({
                'status': 'low_confidence', 
                'roll': roll, 
                'name': st.name, 
                'confidence': confidence,
                'message': f'Low confidence: {st.name} ({confidence:.3f})'
            })
            continue

        # Check if already marked in this session
        existing_in_session = Attendance.query.filter_by(
            class_session_id=session_id, 
            student_id=st.id
        ).first()
        
        if existing_in_session:
            results.append({
                'status': 'already_marked', 
                'roll': roll, 
                'name': st.name,
                'message': f'{st.name} - Already marked in this session'
            })
            continue

        # Enforce once-per-day per subject
        existing_today = Attendance.query.join(ClassSession).filter(
            Attendance.student_id == st.id,
            ClassSession.subject_id == subject_id,
            ClassSession.date == today
        ).first()
        
        if existing_today:
            results.append({
                'status': 'already_marked_today', 
                'roll': roll, 
                'name': st.name,
                'message': f'{st.name} - Already marked today'
            })
            continue

        # Only mark attendance for high confidence matches 
        if status == 'recognized':
            # Mark attendance with timestamp and liveness data
            att = Attendance(
                class_session_id=session_id,
                student_id=st.id,
                timestamp=datetime.utcnow(),
                status='present',
                confidence=confidence,
                reason=f"Liveness score: {liveness_data.get('livenessScore', 0)}% | Anti-spoofing: {liveness_data.get('antiSpoofingScore', 0)}%"
            )
            db.session.add(att)
            db.session.commit()
            
            # Log the attendance for debugging
            logger.info(f" Attendance marked with anti-spoofing: {st.name} ({st.roll}) - Liveness: {liveness_data.get('livenessScore', 0)}%")
            
            results.append({
                'status': 'marked', 
                'roll': roll, 
                'name': st.name,
                'student_id': st.id,
                'confidence': confidence,
                'liveness_score': liveness_data.get('livenessScore', 0),
                'anti_spoofing_score': liveness_data.get('antiSpoofingScore', 0),
                'message': f' Attendance marked for {st.name} with anti-spoofing verification'
            })
        else:
            
            results.append({
                'status': status,
                'roll': roll,
                'name': st.name,
                'confidence': confidence,
                'message': m.get('message', 'Recognition result')
            })

    return jsonify({'results': results})

@app.route('/prof/<int:subject_id>/view_attendance')
@login_required
def view_attendance(subject_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))

    start = request.args.get('start')
    end = request.args.get('end')
    q = Attendance.query.join(ClassSession).filter(ClassSession.subject_id==subj.id)
    if start: q = q.filter(ClassSession.date >= start)
    if end: q = q.filter(ClassSession.date <= end)
    rows = q.order_by(Attendance.timestamp.desc()).all()

    #  all class dates for calendar
    class_dates = [session.date.isoformat() for session in 
                  ClassSession.query.filter_by(subject_id=subject_id).all()]
    total_classes = len(class_dates)

    if request.args.get('format') == 'csv':
        df = pd.DataFrame([{
            'student': r.student.name,
            'roll': r.student.roll,
            'date': r.class_session.date,
            'time': r.timestamp,
            'status': r.status,
            'confidence': r.confidence
        } for r in rows])
        buf = io.StringIO(); df.to_csv(buf, index=False); buf.seek(0)
        return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv',
                         download_name='attendance.csv', as_attachment=True)
    
    return render_template('view_attendance.html', 
                         rows=rows, 
                         subject=subj,
                         class_dates=class_dates,
                         total_classes=total_classes)

# ===== CALENDAR ROUTES =====



@app.route('/prof/<int:subject_id>/attendance_date/<date>')
@login_required
def get_attendance_for_date(subject_id, date):
    """Get attendance records for a specific date - FIXED VERSION (handles multiple sessions per day)"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    try:
        attendance_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    
    #  ALL class sessions for this date, not just the first one
    sessions = ClassSession.query.filter_by(
        subject_id=subject_id, 
        date=attendance_date
    ).all()
    
    if not sessions:
        return jsonify({
            'attendance_records': [], 
            'present_count': 0, 
            'total_students': 0,
            'has_class': False,
            'message': 'No class session found for this date'
        })
    
    #   IDs of all sessions for this day
    session_ids = [s.id for s in sessions]
    
    #  ALL attendance records for ALL sessions on this day
    all_attendance_records = Attendance.query.filter(
        Attendance.class_session_id.in_(session_ids)
    ).join(Student).all()
  

    # Now, aggregate results by student
   
    student_records = {} 
    
    for record in all_attendance_records:
        if record.student_id not in student_records:
            # First time seeing this student today
            student_records[record.student_id] = record
        else:
            # We've seen this student. Favor 'present' status.
            if record.status == 'present' and student_records[record.student_id].status != 'present':
                student_records[record.student_id] = record
            # Also favor the record with the highest confidence if both are present
            elif record.status == 'present' and record.confidence is not None and \
                 (student_records[record.student_id].confidence is None or record.confidence > student_records[record.student_id].confidence):
                 student_records[record.student_id] = record

    #  the final list from our aggregated records
    records_data = []
    for student_id, record in student_records.items():
        records_data.append({
            'attendance_id': record.id,
            'student_id': record.student.id,
            'student_name': record.student.name,
            'student_roll': record.student.roll,
            'status': record.status,
            'confidence': record.confidence,
            'timestamp': record.timestamp.isoformat() if record.timestamp else None
        })
    
    # Count total students in subject for statistics
    total_students_in_subject = Student.query.filter_by(subject_id=subject_id).count()
    # Count present students from our aggregated list
    present_count = len([r for r in records_data if r['status'] == 'present'])
    
    return jsonify({
        'attendance_records': records_data,
        'total_students': total_students_in_subject,
        'present_count': present_count,
        'has_class': True,
        'session_ids': session_ids,
        'date': attendance_date.isoformat(),
        'message': f'Found {len(records_data)} unique attendance records, {present_count} present'
    })

@app.route('/prof/<int:subject_id>/student_attendance/<int:student_id>')
@login_required
def get_student_attendance_details(subject_id, student_id):
    """Get detailed attendance for a specific student"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    student = Student.query.get_or_404(student_id)
    
    # Calculate attendance statistics
    total_classes = ClassSession.query.filter_by(subject_id=subject_id).count()
    classes_attended = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student_id,
        ClassSession.subject_id == subject_id,
        Attendance.status == 'present'
    ).count()
    
    attendance_percentage = (classes_attended / total_classes * 100) if total_classes > 0 else 0
    
    # Get recent attendance records
    recent_attendance = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student_id,
        ClassSession.subject_id == subject_id
    ).order_by(ClassSession.date.desc()).limit(10).all()
    
    recent_data = []
    for record in recent_attendance:
        recent_data.append({
            'attendance_id': record.id,
            'date': record.class_session.date.isoformat(),
            'status': record.status
        })
    
    return jsonify({
        'student_name': student.name,
        'student_roll': student.roll,
        'total_classes': total_classes,
        'classes_attended': classes_attended,
        'attendance_percentage': round(attendance_percentage, 2),
        'eligible_override': student.eligible_override,
        'recent_attendance': recent_data
    })

@app.route('/prof/<int:subject_id>/update_attendance/<int:attendance_id>', methods=['POST'])
@login_required
def update_attendance_status(subject_id, attendance_id):
    """Update attendance status for a student"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    new_status = data.get('status')
    
    if new_status not in ['present', 'absent']:
        return jsonify({'error': 'Invalid status'}), 400
    
    attendance = Attendance.query.get_or_404(attendance_id)
    
    # Verify the attendance record belongs to the subject
    if attendance.class_session.subject_id != subject_id:
        return jsonify({'error': 'Attendance record does not belong to this subject'}), 403
    
    attendance.status = new_status
    attendance.edited = True
    attendance.reason = f"Manually updated by professor on {datetime.utcnow().isoformat()}"
    
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': 'Attendance updated successfully'})

@app.route('/prof/<int:subject_id>/create_session', methods=['POST'])
@login_required
def create_session(subject_id):
    """Create a class session for a specific date"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    date_str = data.get('date')
    if not date_str:
        return jsonify({'error': 'Date is required'}), 400
    
    try:
        session_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    
    # Check if session already exists
    existing_session = ClassSession.query.filter_by(
        subject_id=subject_id,
        date=session_date
    ).first()
    
    if existing_session:
        return jsonify({'error': 'Class session already exists for this date'}), 400
    
    # Create new session
    session = ClassSession(
        subject_id=subject_id,
        date=session_date,
        start_time=datetime.utcnow()
    )
    
    db.session.add(session)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Class session created successfully',
        'session_id': session.id
    })

@app.route('/prof/<int:subject_id>/delete_session/<int:session_id>', methods=['DELETE'])
@login_required
def delete_session(subject_id, session_id):
    """Delete a class session and its attendance records"""
    if not isinstance(current_user, User):
        return jsonify({'error': 'Access denied'}), 403
    
    session = ClassSession.query.get_or_404(session_id)
    
    # Verify the session belongs to the subject
    if session.subject_id != subject_id:
        return jsonify({'error': 'Session does not belong to this subject'}), 403
    
    # Delete attendance records first
    Attendance.query.filter_by(class_session_id=session_id).delete()
    
    # Delete the session
    db.session.delete(session)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Class session and associated attendance records deleted successfully'
    })

@app.route('/prof/<int:subject_id>/attendance/edit/<int:attendance_id>', methods=['GET', 'POST'])
@login_required
def edit_attendance(subject_id, attendance_id):
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))

    subj = Subject.query.get_or_404(subject_id)
    if subj.professor_id != current_user.id:
        flash('Not authorized.', 'danger')
        return redirect(url_for('prof_dashboard'))

    att = Attendance.query.get_or_404(attendance_id)
    if att.class_session.subject_id != subject_id:
        flash('Attendance record mismatch.', 'danger')
        return redirect(url_for('view_attendance', subject_id=subject_id))

    if request.method == 'POST':
        new_status = safe_text(request.form.get('status', att.status), maxlen=32)
        reason = safe_text(request.form.get('reason', ''), maxlen=1000)
        att.status = new_status
        att.reason = reason
        att.edited = True
        db.session.commit()
        flash('Attendance updated successfully.', 'success')
        return redirect(url_for('view_attendance', subject_id=subject_id))

    return render_template('edit_attendance.html', attendance=att, subject=subj)

# -------------------------
# Student Routes
# -------------------------
@app.route('/student/login', methods=['GET', 'POST'])
def login_student():
    if request.method == 'POST':
        roll = request.form.get('roll', '').strip()
        if not roll:
            flash('Please enter your Roll Number', 'danger')
            return redirect(url_for('login_student'))

        # Case-insensitive roll number lookup - get ALL matching students
        students = Student.query.filter(Student.roll.ilike(roll)).all()

        if not students:
            flash('Student not found. Please contact your professor.', 'danger')
            return redirect(url_for('login_student'))

        # If multiple students found with same roll, redirect to selection page
        if len(students) > 1:
            # Store roll in session and redirect to selection page
            from flask import session
            session['pending_roll'] = roll
            return redirect(url_for('select_student_department'))

        # Single student found, log them in
        student = students[0]
        login_user(student)
        flash(f'Welcome {student.name}!', 'success')
        return redirect(url_for('student_dashboard'))

    return render_template('login_student.html')

@app.route('/student/select_department')
def select_student_department():
    from flask import session
    roll = session.get('pending_roll')
    if not roll:
        return redirect(url_for('login_student'))
    
    students = Student.query.filter(Student.roll.ilike(roll)).all()
    return render_template('select_department.html', students=students, roll=roll)

@app.route('/student/login/<int:student_id>')
def login_specific_student(student_id):
    student = Student.query.get_or_404(student_id)
    login_user(student)
    flash(f'Welcome {student.name}!', 'success')
    return redirect(url_for('student_dashboard'))

@app.route('/student/dashboard')
@login_required
def student_dashboard():
    if not isinstance(current_user, Student):
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    student = current_user
    subject = student.subject
    department = student.department

    if not subject:
        flash('No subject assigned yet.', 'warning')
        return render_template('student_dashboard.html', student=student)

    # Attendance summary for specific subject
    total_classes = ClassSession.query.filter_by(subject_id=subject.id).count()
    marked = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student.id,
        ClassSession.subject_id == subject.id
    ).count()
    perc = (marked / total_classes * 100) if total_classes > 0 else 0
    eligible = perc >= 75 or student.eligible_override

    # Attendance trend data for  subject
    attendance_records = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student.id,
        ClassSession.subject_id == subject.id
    ).order_by(ClassSession.date).all()

    chart_labels = [r.class_session.date.strftime('%Y-%m-%d') for r in attendance_records]
    chart_values = list(range(1, len(attendance_records) + 1))

    return render_template(
        'student_dashboard.html',
        student=student,
        subject=subject,
        department=department,
        total_classes=total_classes,
        marked=marked,
        perc=perc,
        eligible=eligible,
        chart_labels=chart_labels,
        chart_values=chart_values
    )

@app.route('/student/attendance/pdf')
@login_required
def download_student_attendance_pdf():
    if not isinstance(current_user, Student):
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    student = current_user
    subject = student.subject

    pdf_path = os.path.join("instance", f"{student.roll}_attendance.pdf")
    c = canvas.Canvas(pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(200, 760, "Student Attendance Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 730, f"Name: {student.name}")
    c.drawString(50, 710, f"Roll No: {student.roll}")
    if student.department:
        c.drawString(50, 690, f"Department: {student.department.name}")
    if subject:
        c.drawString(50, 670, f"Subject: {subject.name}")

    y = 640
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Date")
    c.drawString(150, y, "Status")
    c.drawString(250, y, "Confidence")
    c.line(50, y - 5, 550, y - 5)

    y -= 20
    c.setFont("Helvetica", 10)

    records = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student.id,
        ClassSession.subject_id == subject.id
    ).order_by(ClassSession.date).all()

    for r in records:
        c.drawString(50, y, str(r.class_session.date))
        c.drawString(150, y, r.status)
        c.drawString(250, y, f"{r.confidence:.3f}")
        y -= 15
        if y < 50:
            c.showPage()
            y = 750

    c.save()
    return send_file(pdf_path, as_attachment=True, download_name=f"{student.roll}_attendance.pdf")

@app.route('/student/attendance')
@login_required
def student_attendance_history():
    if not isinstance(current_user, Student):
        flash('Access denied', 'danger')
        return redirect(url_for('index'))

    student = current_user
    subject = student.subject

    start = request.args.get('start')
    end = request.args.get('end')

    q = Attendance.query.join(ClassSession).filter(
        Attendance.student_id == student.id,
        ClassSession.subject_id == subject.id
    )
    if start:
        q = q.filter(ClassSession.date >= start)
    if end:
        q = q.filter(ClassSession.date <= end)

    records = q.order_by(Attendance.timestamp.desc()).all()

    return render_template(
        'student_attendance_history.html',
        student=student,
        subject=subject,
        records=records
    )


@app.route('/professor/change_password', methods=['GET', 'POST'])
@login_required
def change_professor_password():
    """Allow professors to change their password"""
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        current_password = request.form.get('current_password', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not current_password or not new_password or not confirm_password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('change_professor_password'))
        
        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'danger')
            return redirect(url_for('change_professor_password'))
        
        if new_password != confirm_password:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('change_professor_password'))
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('change_professor_password'))
        
        # Update password
        current_user.set_password(new_password)
        current_user.password_change_required = False  # Password has been changed
        db.session.commit()
        
        flash('Password changed successfully!', 'success')
        return redirect(url_for('prof_dashboard'))
    
    return render_template('change_password.html', 
                         user=current_user,
                         password_change_required=current_user.password_change_required)

@app.route('/professor/first_login', methods=['GET', 'POST'])
@login_required
def professor_first_login():
    """Force password change on first login"""
    if not isinstance(current_user, User):
        flash('Access denied.', 'danger')
        return redirect(url_for('index'))
    
    if not current_user.password_change_required:
        return redirect(url_for('prof_dashboard'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not new_password or not confirm_password:
            flash('Both fields are required.', 'danger')
            return redirect(url_for('professor_first_login'))
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('professor_first_login'))
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters long.', 'danger')
            return redirect(url_for('professor_first_login'))
        
        # Update password
        current_user.set_password(new_password)
        current_user.password_change_required = False
        db.session.commit()
        
        flash('Password set successfully! Welcome to your dashboard.', 'success')
        return redirect(url_for('prof_dashboard'))
    
    return render_template('first_login.html', user=current_user)



@app.route('/student/logout')
@login_required
def student_logout():
    logout_user()
    return redirect(url_for('index'))

# Development route
@app.route('/dev/reset_db')
def dev_reset_db():
    if os.environ.get('ALLOW_RESET') != '1':
        return "disabled"
    db_path = os.path.join(INSTANCE_DIR, 'app.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    with app.app_context():
        db.create_all()
    return "reset"

if __name__ == '__main__':
    app.run(debug=True)
