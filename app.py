import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import sqlite3
import os
import json
from passlib.hash import bcrypt_sha256 as bcrypt
from passlib.hash import sha256_crypt
import base64
import ast
import google.generativeai as genai
from streamlit_calendar import calendar
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import io

# Create assets directory and default avatar
def create_default_avatar():
    # Create assets directory if it doesn't exist
    if not os.path.exists('assets'):
        os.makedirs('assets')
    
    # Create a default avatar if it doesn't exist
    if not os.path.exists('assets/default_avatar.png'):
        # Create a 200x200 image with a light blue background
        img = Image.new('RGB', (200, 200), color='#3498db')
        draw = ImageDraw.Draw(img)
        
        # Draw a simple user icon
        # Draw a circle for the head
        draw.ellipse((50, 30, 150, 130), fill='white')
        # Draw a body
        draw.rectangle((75, 130, 125, 180), fill='white')
        
        # Save the image
        img.save('assets/default_avatar.png')

# Create default avatar at startup
create_default_avatar()

# Configure Google Generative AI (replace with your API key)
genai.configure(api_key="AIzaSyCNzvUNK_wMMplbUMy0HnDBCkHtxgmpJDY")
model = genai.GenerativeModel("gemini-2.0-flash")

# Set page config
st.set_page_config(
    page_title="Student Study Tracker",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Create assets directory
if not os.path.exists('assets'):
    os.makedirs('assets')

# Load custom CSS and Font Awesome
def load_css():
    css = """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f6f8fc 0%, #e9f0f7 100%);
        font-family: 'Inter', sans-serif;
        color: #2c3e50;
    }
    .nav-bar {
        background: linear-gradient(90deg, #2c3e50 0%, #3498db 100%);
        padding: 15px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .nav-bar a, .nav-bar button {
        color: white;
        text-decoration: none;
        font-size: 1.1rem;
        font-weight: 500;
        padding: 10px 20px;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
    }
    .nav-bar a:hover, .nav-bar button:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: translateY(-2px);
    }
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        text-align: center;
        margin: 15px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        color: #2c3e50;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #34495e;
        font-weight: 500;
    }
    .hero-section {
        background: linear-gradient(135deg, #3498db 0%, #2c3e50 100%);
        padding: 40px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .hero-section h2 {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 15px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .hero-section p {
        font-size: 1.2rem;
        opacity: 0.9;
        color: white;
    }
    .avatar {
        border-radius: 50%;
        width: 120px;
        height: 120px;
        object-fit: cover;
        border: 4px solid white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #3498db 0%, #2980b9 100%);
        color: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2980b9 0%, #2c3e50 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(52, 152, 219, 0.3);
    }
    .stTextInput>div>input, .stTextArea>div>textarea, .stSelectbox>div>select, .stDateInput>div>input {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 12px;
        background: white;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        color: #2c3e50;
    }
    .task-form, .study-plan-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 20px 0;
        border: 1px solid rgba(0,0,0,0.05);
        color: #2c3e50;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #2c3e50;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #3498db;
        color: white;
        transform: translateY(-2px);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #3498db;
        color: white;
        box-shadow: 0 4px 15px rgba(52, 152, 219, 0.2);
    }
    .study-plan-card {
        background: white;
        padding: 25px;
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin: 20px 0;
        color: #2c3e50;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .study-plan-card p {
        color: #34495e;
        font-size: 1.1rem;
        line-height: 1.7;
    }
    /* Add styles for Streamlit elements */
    .stMarkdown {
        color: #2c3e50;
    }
    .stSubheader {
        color: #2c3e50;
    }
    .stTitle {
        color: #2c3e50;
    }
    .stDataFrame {
        color: #2c3e50;
    }
    .stAlert {
        color: #2c3e50 !important;
    }
    .stExpander {
        color: #2c3e50;
    }
    .stSelectbox label, .stTextInput label, .stTextArea label, .stDateInput label {
        color: #2c3e50;
        font-weight: 500;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: #2c3e50;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: #2c3e50;
    }
    .stSelectbox div[data-baseweb="select"] div {
        color: #2c3e50;
    }
    .stSelectbox div[data-baseweb="select"] div[aria-selected="true"] {
        color: white;
    }
    /* Error message styling */
    .stAlert div[data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] strong {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] em {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] code {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] pre {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] blockquote {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] ul {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] ol {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] li {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] a {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] img {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] table {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] th {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] td {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] tr {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] thead {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] tbody {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] tfoot {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] caption {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] hr {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] br {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] div {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] span {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h1 {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h2 {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h3 {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h4 {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h5 {
        color: #2c3e50 !important;
    }
    .stAlert div[data-testid="stMarkdownContainer"] h6 {
        color: #2c3e50 !important;
    }
    </style>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """
    st.markdown(css, unsafe_allow_html=True)

load_css()

# Database migration and setup
def migrate_db():
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    try:
        # Check if onboarded column exists
        c.execute("PRAGMA table_info(students)")
        columns = [col[1] for col in c.fetchall()]
        if 'onboarded' not in columns:
            c.execute('ALTER TABLE students ADD COLUMN onboarded BOOLEAN DEFAULT 0')
        
        # Check if email column exists in students table
        if 'email' not in columns:
            # First add the column without UNIQUE constraint
            c.execute('ALTER TABLE students ADD COLUMN email TEXT')
            # Update existing rows with unique email addresses
            c.execute('SELECT student_id FROM students')
            student_ids = c.fetchall()
            for student_id in student_ids:
                c.execute('UPDATE students SET email = ? WHERE student_id = ?',
                         (f"student_{student_id[0]}@example.com", student_id[0]))
            
            # Now create new table with UNIQUE constraint
            c.execute('''
                CREATE TABLE students_new (
                    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    bio TEXT,
                    linkedin TEXT,
                    instagram TEXT,
                    avatar TEXT,
                    onboarded BOOLEAN DEFAULT 0
                )
            ''')
            # Copy data from old table to new table
            c.execute('INSERT INTO students_new SELECT * FROM students')
            # Drop old table
            c.execute('DROP TABLE students')
            # Rename new table to old name
            c.execute('ALTER TABLE students_new RENAME TO students')
            
        # Check if email column exists in credentials table
        c.execute("PRAGMA table_info(credentials)")
        cred_columns = [col[1] for col in c.fetchall()]
        if 'email' not in cred_columns:
            # First add the column without UNIQUE constraint
            c.execute('ALTER TABLE credentials ADD COLUMN email TEXT')
            # Update existing rows with unique email addresses
            c.execute('SELECT student_id FROM credentials')
            cred_ids = c.fetchall()
            for cred_id in cred_ids:
                c.execute('UPDATE credentials SET email = ? WHERE student_id = ?',
                         (f"student_{cred_id[0]}@example.com", cred_id[0]))
            
            # Now create new table with UNIQUE constraint
            c.execute('''
                CREATE TABLE credentials_new (
                    student_id INTEGER,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password TEXT,
                    is_admin BOOLEAN,
                    FOREIGN KEY(student_id) REFERENCES students(student_id)
                )
            ''')
            # Copy data from old table to new table
            c.execute('INSERT INTO credentials_new SELECT * FROM credentials')
            # Drop old table
            c.execute('DROP TABLE credentials')
            # Rename new table to old name
            c.execute('ALTER TABLE credentials_new RENAME TO credentials')
            
        # Migrate existing passwords to new format
        c.execute('SELECT student_id, password FROM credentials')
        credentials = c.fetchall()
        for student_id, old_hash in credentials:
            try:
                # Try to verify with new format
                bcrypt.verify("test", old_hash)
            except:
                # If verification fails, update to new format
                new_hash = bcrypt.hash("default_password")  # Set a default password
                c.execute('UPDATE credentials SET password = ? WHERE student_id = ?',
                         (new_hash, student_id))
        
        conn.commit()
    except sqlite3.OperationalError as e:
        st.error(f"Database migration error: {e}")
    finally:
        conn.close()

def init_db():
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS students (
        student_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        bio TEXT,
        linkedin TEXT,
        instagram TEXT,
        avatar TEXT,
        onboarded BOOLEAN DEFAULT 0
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS credentials (
        student_id INTEGER,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT,
        is_admin BOOLEAN,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS password_resets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        token TEXT UNIQUE,
        expiry TEXT,
        FOREIGN KEY(email) REFERENCES credentials(email)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS courses (
        course_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_name TEXT,
        marking_scheme TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS tasks (
        task_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER,
        task_name TEXT,
        due_date TEXT,
        status TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id),
        FOREIGN KEY(course_id) REFERENCES courses(course_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS grades (
        grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_id INTEGER,
        quiz REAL,
        midterm REAL,
        final REAL,
        FOREIGN KEY(student_id) REFERENCES students(student_id),
        FOREIGN KEY(course_id) REFERENCES courses(course_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        date TEXT,
        organizer TEXT,
        verified BOOLEAN
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS clubs (
        club_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        club_name TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')
    c.execute('''CREATE TABLE IF NOT EXISTS study_plans (
        plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id INTEGER,
        course_name TEXT,
        exam_date TEXT,
        plan_text TEXT,
        created_at TEXT,
        FOREIGN KEY(student_id) REFERENCES students(student_id)
    )''')
    conn.commit()
    conn.close()

# Run migration and initialization
migrate_db()
init_db()

# Database helper functions
def migrate_password(old_hash):
    """Migrate old password hash to new format"""
    try:
        # Try to verify with old hash format
        if sha256_crypt.verify(old_hash, old_hash):
            return old_hash
        return None
    except:
        return None

def add_reset_token(email):
    """Add a password reset token for a user"""
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    # Check if email exists
    c.execute('SELECT email FROM credentials WHERE email = ?', (email,))
    if not c.fetchone():
        conn.close()
        return None
    
    # Generate a random token
    token = base64.b64encode(os.urandom(32)).decode('utf-8')
    # Store token with expiration (24 hours from now)
    expiry = (datetime.now() + timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO password_resets (email, token, expiry) VALUES (?, ?, ?)',
              (email, token, expiry))
    conn.commit()
    conn.close()
    return token

def verify_reset_token(token):
    """Verify if a reset token is valid and not expired"""
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    c.execute('SELECT email FROM password_resets WHERE token = ? AND expiry > ?',
              (token, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    result = c.fetchone()
    conn.close()
    return result[0] if result else None

def reset_password(token, new_password):
    """Reset password using a valid token"""
    email = verify_reset_token(token)
    if not email:
        return False
    
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    hashed_password = bcrypt.hash(new_password)
    c.execute('UPDATE credentials SET password = ? WHERE email = ?',
              (hashed_password, email))
    # Delete used token
    c.execute('DELETE FROM password_resets WHERE token = ?', (token,))
    conn.commit()
    conn.close()
    return True

def add_student(name, email, bio, linkedin, instagram, avatar, username, password, is_admin=False):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    try:
        # Handle avatar file
        avatar_path = "assets/default_avatar.png"
        if avatar is not None:
            try:
                # Create a unique filename
                avatar_filename = f"assets/{username}_avatar.png"
                # Save the uploaded file
                with open(avatar_filename, "wb") as f:
                    f.write(avatar.getvalue())
                avatar_path = avatar_filename
            except Exception as e:
                st.error(f"Failed to save avatar: {e}")
                avatar_path = "assets/default_avatar.png"
        
        c.execute('INSERT INTO students (name, email, bio, linkedin, instagram, avatar, onboarded) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  (name, email, bio, linkedin, instagram, avatar_path, False))
        student_id = c.lastrowid
        hashed_password = bcrypt.hash(password)
        c.execute('INSERT INTO credentials (student_id, username, email, password, is_admin) VALUES (?, ?, ?, ?, ?)',
                  (student_id, username, email, hashed_password, is_admin))
        conn.commit()
        return student_id
    except sqlite3.IntegrityError as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def verify_login(identifier, password):
    """Verify login using either username or email"""
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    c.execute('SELECT student_id, password, is_admin FROM credentials WHERE username = ? OR email = ?', 
              (identifier, identifier))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return None, None
        
    stored_hash = result[1]
    
    try:
        if bcrypt.verify(password, stored_hash):
            return result[0], result[2]
    except:
        migrated_hash = migrate_password(stored_hash)
        if migrated_hash:
            conn = sqlite3.connect('study_tracker.db')
            c = conn.cursor()
            new_hash = bcrypt.hash(password)
            c.execute('UPDATE credentials SET password = ? WHERE student_id = ?',
                     (new_hash, result[0]))
            conn.commit()
            conn.close()
            return result[0], result[2]
    
    return None, None

def check_onboarding(student_id):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    try:
        c.execute('SELECT onboarded FROM students WHERE student_id = ?', (student_id,))
        result = c.fetchone()
        conn.close()
        return result[0] if result else False
    except sqlite3.OperationalError:
        conn.close()
        return False  # Fallback if column is missing

def set_onboarded(student_id):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    c.execute('UPDATE students SET onboarded = ? WHERE student_id = ?', (True, student_id))
    conn.commit()
    conn.close()

def load_data(student_id):
    conn = sqlite3.connect('study_tracker.db')
    try:
        students = pd.read_sql_query('SELECT * FROM students WHERE student_id = ?', conn, params=(student_id,))
        courses = pd.read_sql_query('SELECT * FROM courses WHERE student_id = ?', conn, params=(student_id,))
        tasks = pd.read_sql_query('''
            SELECT t.task_id, t.student_id, t.course_id, t.task_name, t.due_date, t.status, c.course_name 
            FROM tasks t 
            LEFT JOIN courses c ON t.course_id = c.course_id 
            WHERE t.student_id = ?
            ORDER BY t.due_date
        ''', conn, params=(student_id,))
        grades = pd.read_sql_query('SELECT g.*, c.course_name, c.marking_scheme FROM grades g JOIN courses c ON g.course_id = c.course_id WHERE g.student_id = ?', conn, params=(student_id,))
        events = pd.read_sql_query('SELECT * FROM events', conn)
        clubs = pd.read_sql_query('SELECT * FROM clubs WHERE student_id = ?', conn, params=(student_id,))
        study_plans = pd.read_sql_query('SELECT * FROM study_plans WHERE student_id = ?', conn, params=(student_id,))
        return students, courses, tasks, grades, events, clubs, study_plans
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    finally:
        conn.close()

def add_task(student_id, course_id, task_name, due_date, status):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    try:
        # First check if the task already exists
        c.execute('SELECT task_id FROM tasks WHERE student_id = ? AND course_id = ? AND task_name = ?',
                  (student_id, course_id, task_name))
        if c.fetchone():
            st.error("This task already exists")
            return False
            
        # Add the new task
        c.execute('INSERT INTO tasks (student_id, course_id, task_name, due_date, status) VALUES (?, ?, ?, ?, ?)',
                  (student_id, course_id, task_name, due_date, status))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Error adding task: {e}")
        return False
    finally:
        conn.close()

def add_course(student_id, course_name, marking_scheme):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    marking_scheme_json = json.dumps(marking_scheme)  # Ensure proper JSON encoding
    c.execute('INSERT INTO courses (student_id, course_name, marking_scheme) VALUES (?, ?, ?)',
              (student_id, course_name, marking_scheme_json))
    course_id = c.lastrowid
    conn.commit()
    conn.close()
    return course_id

def add_grade(student_id, course_id, marks):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    c.execute('INSERT INTO grades (student_id, course_id, quiz, midterm, final) VALUES (?, ?, ?, ?, ?)',
              (student_id, course_id, marks.get('quiz', None), marks.get('midterm', None), marks.get('final', None)))
    conn.commit()
    conn.close()

def add_study_plan(student_id, course_name, exam_date, plan_text):
    conn = sqlite3.connect('study_tracker.db')
    c = conn.cursor()
    created_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute('INSERT INTO study_plans (student_id, course_name, exam_date, plan_text, created_at) VALUES (?, ?, ?, ?, ?)',
              (student_id, course_name, exam_date, plan_text, created_at))
    conn.commit()
    conn.close()

# Helper functions
def calculate_grade(row):
    try:
        marking_scheme = json.loads(row['marking_scheme'])
        marks = []
        weights = []
        for key, weight in marking_scheme.items():
            if key in row and not np.isnan(row[key]):
                marks.append(row[key])
                weights.append(weight / 100)
        return np.sum(np.array(marks) * np.array(weights)) if marks else np.nan
    except (ValueError, json.JSONDecodeError):
        return np.nan

def predict_grade(historical_grades, course):
    if len(historical_grades) < 2:
        return np.nan
    X = np.array(range(len(historical_grades))).reshape(-1, 1)
    y = historical_grades
    model = LinearRegression()
    model.fit(X, y)
    return model.predict([[len(historical_grades)]])[0]

def generate_study_plan(exam_date, course, days):
    prompt = f"Generate a {days}-day study plan for a {course} exam scheduled on {exam_date}. Include daily tasks and study hours."
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating study plan: {e}"

# Session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
    st.session_state.is_admin = False
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'current_plan' not in st.session_state:
    st.session_state.current_plan = None

# Navigation bar
def render_nav_bar():
    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1])
    with cols[0]:
        if st.button("Dashboard"):
            st.session_state.page = "Dashboard"
            st.rerun()
    with cols[1]:
        if st.button("Calendar"):
            st.session_state.page = "Calendar"
            st.rerun()
    with cols[2]:
        if st.button("Tasks"):
            st.session_state.page = "Tasks"
            st.rerun()
    with cols[3]:
        if st.button("Grades"):
            st.session_state.page = "Grades"
            st.rerun()
    with cols[4]:
        if st.button("Study Plan"):
            st.session_state.page = "Study Plan"
            st.rerun()
    with cols[5]:
        if st.button("Profile"):
            st.session_state.page = "Profile"
            st.rerun()
    with cols[6]:
        if st.button("Events"):
            st.session_state.page = "Events"
            st.rerun()
    with cols[7]:
        if st.session_state.is_admin:
            if st.button("Admin"):
                st.session_state.page = "Admin"
                st.rerun()
        if st.button("Logout"):
            st.session_state.page = "Logout"
            st.rerun()

# Login/Signup
if st.session_state.page in ["Login", "Signup", "ForgotPassword", "ResetPassword"]:
    st.title("Student Study Tracker")
    st.markdown('<div class="hero-section"><h2>Your Path to Academic Success!</h2><p><i class="fas fa-rocket"></i> Organize, study, and shine!</p></div>', unsafe_allow_html=True)
    
    if st.session_state.page == "ForgotPassword":
        st.subheader("Forgot Password")
        email = st.text_input("Enter your email address")
        if st.button("Send Reset Link"):
            if email:
                try:
                    token = add_reset_token(email)
                    if token:
                        # In a real application, you would send this token via email
                        # For demo purposes, we'll show it in the UI
                        st.success(f"Password reset link generated! (Demo token: {token})")
                        st.info("In a production environment, this would be sent via email.")
                        st.session_state.page = "Login"
                        st.rerun()
                    else:
                        st.error("Email address not found")
                except Exception as e:
                    st.error(f"Error generating reset link: {e}")
            else:
                st.error("Please enter your email address")
        if st.button("Back to Login"):
            st.session_state.page = "Login"
            st.rerun()
    
    elif st.session_state.page == "ResetPassword":
        st.subheader("Reset Password")
        token = st.text_input("Enter reset token")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Reset Password"):
            if new_password and confirm_password:
                if new_password == confirm_password:
                    if reset_password(token, new_password):
                        st.success("Password reset successful! Please login with your new password.")
                        st.session_state.page = "Login"
                        st.rerun()
                    else:
                        st.error("Invalid or expired token")
                else:
                    st.error("Passwords do not match")
            else:
                st.error("Please fill in all fields")
        if st.button("Back to Login"):
            st.session_state.page = "Login"
            st.rerun()
    
    else:
        tab1, tab2 = st.tabs(["Login", "Signup"])
        
        with tab1:
            st.subheader("Login")
            identifier = st.text_input("Email or Username")
            password = st.text_input("Password", type="password", key="login_password")
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("Login"):
                    user_id, is_admin = verify_login(identifier, password)
                    if user_id:
                        st.session_state.user_id = user_id
                        st.session_state.is_admin = is_admin
                        st.session_state.page = "Onboarding" if not check_onboarding(user_id) else "Dashboard"
                        st.rerun()
                    else:
                        st.error("Invalid credentials")
            with col2:
                if st.button("Forgot Password?"):
                    st.session_state.page = "ForgotPassword"
                    st.rerun()
        
        with tab2:
            st.subheader("Signup")
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
            username = st.text_input("Username", key="signup_username")
            password = st.text_input("Password", type="password", key="signup_password")
            bio = st.text_area("Bio")
            linkedin = st.text_input("LinkedIn URL")
            instagram = st.text_input("Instagram URL")
            avatar = st.file_uploader("Upload Avatar", type=["png", "jpg"])
            
            if st.button("Signup"):
                if username and password and name and email:
                    avatar_path = "assets/default_avatar.png"
                    if avatar:
                        avatar_path = f"assets/{username}_avatar.png"
                        try:
                            with open(avatar_path, "wb") as f:
                                f.write(avatar.read())
                        except Exception as e:
                            st.error(f"Failed to save avatar: {e}")
                            avatar_path = "assets/default_avatar.png"
                    try:
                        user_id = add_student(name, email, bio, linkedin, instagram, avatar_path, username, password)
                        st.session_state.user_id = user_id
                        st.session_state.is_admin = False
                        st.session_state.page = "Onboarding"
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("Username or email already exists")
                else:
                    st.error("Please fill all required fields")
    st.stop()

# Load user data
if st.session_state.user_id:
    students, courses, tasks, grades, events, clubs, study_plans = load_data(st.session_state.user_id)

# Onboarding
if st.session_state.page == "Onboarding":
    st.title("Welcome to Your Study Journey!")
    st.markdown("<div class='hero-section'><h2>Let's Set Up Your Courses</h2><p><i class='fas fa-book'></i> Add your courses to get started!</p></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    .course-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .course-section h3 {
        color: #2c3e50;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("Add Courses")
    num_courses = st.number_input("Number of Courses", min_value=1, max_value=10, value=1, step=1)
    course_data = []
    
    for i in range(num_courses):
        st.markdown(f"""
        <div class='course-section'>
            <h3>Course {i+1}</h3>
        </div>
        """, unsafe_allow_html=True)
        course_name = st.text_input(f"Course Name", key=f"course_name_{i}")
        st.markdown("**Marking Scheme**")
        components = st.number_input(f"Number of Components", min_value=1, max_value=5, value=1, step=1, key=f"components_{i}")
        marking_scheme = {}
        total_weight = 0
        for j in range(components):
            col1, col2 = st.columns(2)
            with col1:
                component_name = st.text_input(f"Component Name", key=f"component_name_{i}_{j}")
            with col2:
                weight = st.number_input(f"Weight (%)", min_value=0, max_value=100, step=1, key=f"weight_{i}_{j}")
            if component_name:
                marking_scheme[component_name] = weight
                total_weight += weight
        if total_weight != 100 and total_weight > 0:
            st.error(f"Total weight for {course_name or f'Course {i+1}'} must be 100% (currently {total_weight}%)")
        else:
            course_data.append({"course_name": course_name, "marking_scheme": marking_scheme})
    
    if st.button("Complete Onboarding"):
        if all(data["course_name"] and data["marking_scheme"] for data in course_data) and all(sum(data["marking_scheme"].values()) == 100 for data in course_data):
            for data in course_data:
                add_course(st.session_state.user_id, data["course_name"], data["marking_scheme"])
            set_onboarded(st.session_state.user_id)
            st.session_state.page = "Dashboard"
            st.rerun()
        else:
            st.error("Please fill all course names and ensure marking schemes total 100%.")
    st.stop()

# Logout
if st.session_state.page == "Logout":
    st.session_state.user_id = None
    st.session_state.is_admin = False
    st.session_state.page = "Login"
    st.rerun()

# Render navigation bar
if st.session_state.user_id and st.session_state.page != "Onboarding":
    render_nav_bar()

# Dashboard
if st.session_state.page == "Dashboard":
    st.title("Your Study Dashboard")
    st.markdown('<div class="hero-section"><h2>Keep the Momentum Going!</h2><p><i class="fas fa-star"></i> "Success is the sum of small efforts, repeated day in and day out."</p></div>', unsafe_allow_html=True)
    
    st.subheader("Upcoming Deadlines")
    if not tasks.empty:
        upcoming = tasks[tasks['due_date'] >= datetime.now().strftime('%Y-%m-%d')]
        for _, task in upcoming.head(3).iterrows():
            st.markdown(f'<div class="metric-card"><div class="metric-value"><i class="fas fa-tasks"></i> {task["task_name"]} ({task["course_name"]})</div><div class="metric-label">Due: {task["due_date"]}</div></div>', unsafe_allow_html=True)
    else:
        st.info("No upcoming tasks.")

    st.subheader("Recent Grades")
    if not grades.empty:
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
        
        # Bar plot for quiz grades
        ax1 = fig.add_subplot(gs[0, 0])
        sns.barplot(data=grades, x='course_name', y='quiz', ax=ax1, palette='viridis')
        ax1.set_title("Quiz Grades by Course", fontsize=14, pad=20)
        ax1.set_xlabel("Course", fontsize=12)
        ax1.set_ylabel("Quiz Score", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Box plot for grade distribution
        ax2 = fig.add_subplot(gs[0, 1])
        sns.boxplot(data=grades[['quiz', 'midterm', 'final']], ax=ax2, palette='Set3')
        ax2.set_title("Grade Distribution", fontsize=14, pad=20)
        ax2.set_ylabel("Score", fontsize=12)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add a line plot using plotly for trend analysis
        fig = px.line(grades, x='course_name', y=['quiz', 'midterm', 'final'],
                     title="Grade Trends Across Components",
                     template="plotly_white",
                     markers=True)
        fig.update_layout(
            xaxis_title="Course",
            yaxis_title="Score",
            legend_title="Component",
            hovermode="x unified",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No grades available.")

# Calendar
elif st.session_state.page == "Calendar":
    st.title("Academic Calendar")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Your Calendar")
        calendar_events = []
        
        # Load fresh data
        students, courses, tasks, grades, events, clubs, study_plans = load_data(st.session_state.user_id)
        
        if not tasks.empty:
            for _, task in tasks.iterrows():
                calendar_events.append({
                    "title": f"{task['task_name']} ({task['course_name']})",
                    "start": task['due_date'],
                    "end": task['due_date'],
                    "color": "#3b82f6" if task['status'] == "Pending" else "#10b981"
                })
        if not events.empty:
            for _, event in events[events['verified']].iterrows():
                calendar_events.append({
                    "title": event['name'],
                    "start": event['date'],
                    "end": event['date'],
                    "color": "#7c3aed"
                })
        
        if calendar_events:
            calendar(
                events=calendar_events,
                options={
                    "headerToolbar": {
                        "left": "prev,next today",
                        "center": "title",
                        "right": "dayGridMonth,timeGridWeek,timeGridDay"
                    },
                    "initialView": "dayGridMonth"
                }
            )
        else:
            st.info("No events or tasks to display in calendar.")
    
    with col2:
        st.subheader("Add Event")
        event_date = st.date_input("Event Date")
        event_name = st.text_input("Event Name")
        event_type = st.selectbox("Event Type", ["Assignment", "Exam", "Event", "Class"])
        course = st.selectbox("Course", courses['course_name'] if not courses.empty else ["None"])
        if st.button("Add to Calendar"):
            if course != "None" and not courses.empty and event_name:
                course_id = courses[courses['course_name'] == course]['course_id'].iloc[0]
                if add_task(st.session_state.user_id, course_id, event_name, event_date.strftime('%Y-%m-%d'), 'Pending'):
                    st.success("Event added!")
                    st.rerun()
            else:
                st.error("Please provide an event name and select a valid course.")

# Tasks
elif st.session_state.page == "Tasks":
    st.title("Task Management")
    col1, col2 = st.columns(2)
    
    # Load fresh data
    students, courses, tasks, grades, events, clubs, study_plans = load_data(st.session_state.user_id)
    
    with col1:
        st.subheader("Add Task")
        with st.container():
            st.markdown('<div class="task-form"><h4><i class="fas fa-plus-circle"></i> Create New Task</h4>', unsafe_allow_html=True)
            task_name = st.text_input("Task Name", placeholder="e.g., Assignment 1")
            course = st.selectbox("Course", courses['course_name'] if not courses.empty else ["None"])
            due_date = st.date_input("Due Date")
            status = st.selectbox("Status", ["Pending", "Completed"])
            if st.button("Add Task"):
                if course != "None" and not courses.empty and task_name:
                    course_id = courses[courses['course_name'] == course]['course_id'].iloc[0]
                    if add_task(st.session_state.user_id, course_id, task_name, due_date.strftime('%Y-%m-%d'), status):
                        st.success("Task added!")
                        st.rerun()
                else:
                    st.error("Please provide a task name and select a valid course.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Your Tasks")
        if not tasks.empty:
            # Separate current and past tasks
            current_tasks = tasks[tasks['status'] == 'Pending']
            past_tasks = tasks[tasks['status'] == 'Completed']
            
            # Display current tasks
            if not current_tasks.empty:
                st.markdown("### Current Tasks")
                for _, task in current_tasks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value"><i class="fas fa-tasks"></i> {task['task_name']}</div>
                        <div class="metric-label">
                            Course: {task['course_name']}<br>
                            Due: {task['due_date']}<br>
                            Status: {task['status']}
                        </div>
                        <div style="margin-top: 15px; display: flex; gap: 10px; justify-content: center;">
                            <button class="stButton" onclick="document.getElementById('complete_{task['task_id']}').click()">
                                <i class="fas fa-check"></i> Complete
                            </button>
                            <button class="stButton" onclick="document.getElementById('update_{task['task_id']}').click()">
                                <i class="fas fa-calendar"></i> Update Deadline
                            </button>
                            <button class="stButton" onclick="document.getElementById('delete_{task['task_id']}').click()">
                                <i class="fas fa-trash"></i> Delete
                            </button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Hidden buttons for functionality
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Complete", key=f"complete_{task['task_id']}", type="primary"):
                            conn = sqlite3.connect('study_tracker.db')
                            c = conn.cursor()
                            try:
                                c.execute('UPDATE tasks SET status = ? WHERE task_id = ?',
                                         ('Completed', task['task_id']))
                                conn.commit()
                                st.success("Task marked as completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating task: {e}")
                            finally:
                                conn.close()
                    
                    with col2:
                        new_date = st.date_input("New Deadline", 
                                               value=datetime.strptime(task['due_date'], '%Y-%m-%d').date(),
                                               key=f"date_{task['task_id']}")
                        if st.button("Update Deadline", key=f"update_{task['task_id']}", type="primary"):
                            conn = sqlite3.connect('study_tracker.db')
                            c = conn.cursor()
                            try:
                                c.execute('UPDATE tasks SET due_date = ? WHERE task_id = ?',
                                         (new_date.strftime('%Y-%m-%d'), task['task_id']))
                                conn.commit()
                                st.success("Deadline updated!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error updating deadline: {e}")
                            finally:
                                conn.close()
                    
                    with col3:
                        if st.button("Delete", key=f"delete_{task['task_id']}", type="primary"):
                            conn = sqlite3.connect('study_tracker.db')
                            c = conn.cursor()
                            try:
                                c.execute('DELETE FROM tasks WHERE task_id = ?', (task['task_id'],))
                                conn.commit()
                                st.success("Task deleted!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting task: {e}")
                            finally:
                                conn.close()
            
            # Display past tasks
            if not past_tasks.empty:
                st.markdown("### Completed Tasks")
                for _, task in past_tasks.iterrows():
                    st.markdown(f"""
                    <div class="metric-card" style="opacity: 0.7;">
                        <div class="metric-value"><i class="fas fa-check-circle"></i> {task['task_name']}</div>
                        <div class="metric-label">
                            Course: {task['course_name']}<br>
                            Completed on: {task['due_date']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No tasks available.")

# Grades
elif st.session_state.page == "Grades":
    st.title("Grade Tracker")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Add Course")
        course_name = st.text_input("Course Name")
        st.markdown("**Marking Scheme**")
        components = st.number_input("Number of Components", min_value=1, max_value=5, value=1, step=1)
        marking_scheme = {}
        total_weight = 0
        for i in range(components):
            col_a, col_b = st.columns(2)
            with col_a:
                component_name = st.text_input(f"Component {i+1} Name", key=f"component_{i}")
            with col_b:
                weight = st.number_input(f"Weight (%)", min_value=0, max_value=100, step=1, key=f"weight_{i}")
            if component_name:
                marking_scheme[component_name] = weight
                total_weight += weight
        if total_weight != 100 and total_weight > 0:
            st.error(f"Total weight must be 100% (currently {total_weight}%)")
        if st.button("Add Course"):
            if course_name and total_weight == 100 and marking_scheme:
                add_course(st.session_state.user_id, course_name, marking_scheme)
                st.success("Course added!")
                st.rerun()
            else:
                st.error("Please provide a course name, valid components, and ensure marking scheme totals 100%.")
    
        st.subheader("Add Grades")
        course = st.selectbox("Course", courses['course_name'] if not courses.empty else ["None"])
        if course != "None" and not courses.empty:
            try:
                marking_scheme = json.loads(courses[courses['course_name'] == course]['marking_scheme'].iloc[0])
            except (json.JSONDecodeError, KeyError):
                st.error("Error loading marking scheme. Please try adding the course again.")
                marking_scheme = {}
            marks = {}
            for component in marking_scheme.keys():
                marks[component] = st.number_input(f"{component} Score", 0, 100, step=1, key=f"mark_{component}")
            if st.button("Add Grade"):
                course_id = courses[courses['course_name'] == course]['course_id'].iloc[0]
                add_grade(st.session_state.user_id, course_id, marks)
                st.success("Grade added!")
                st.rerun()
        else:
            st.info("Please add a course first.")
    
    with col2:
        st.subheader("Grade Visualization")
        if not grades.empty:
            grades['overall'] = grades.apply(calculate_grade, axis=1)
            
            # Create a modern dashboard-style visualization
            fig = px.line(grades, x='course_name', y='overall',
                         title="Overall Grade Progression",
                         template="plotly_white",
                         markers=True)
            
            # Customize the layout
            fig.update_layout(
                xaxis_title="Course",
                yaxis_title="Overall Grade (%)",
                hovermode="x unified",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                showlegend=False
            )
            
            # Add a gradient fill
            fig.update_traces(
                line=dict(width=3),
                marker=dict(size=8),
                fill='tonexty'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a radar chart for component comparison
            if len(grades) > 0:
                latest_course = grades.iloc[-1]
                components = ['quiz', 'midterm', 'final']
                values = [latest_course[comp] for comp in components if not pd.isna(latest_course[comp])]
                
                if len(values) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=components[:len(values)],
                        fill='toself',
                        name=latest_course['course_name']
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        showlegend=False,
                        title="Component Distribution",
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Grade Prediction")
            historical_grades = grades['overall'].dropna()
            if not historical_grades.empty:
                predicted = predict_grade(historical_grades, course if course != "None" else "Unknown")
                
                # Create a gauge chart for the prediction
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=predicted,
                    title={'text': f"Predicted Grade ({course if course != 'None' else 'Unknown'})"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, 60], 'color': "#e74c3c"},
                            {'range': [60, 75], 'color': "#f1c40f"},
                            {'range': [75, 100], 'color': "#2ecc71"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 75
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No grades available for prediction.")
        else:
            st.info("No grades available.")

# Study Plan
elif st.session_state.page == "Study Plan":
    st.title("Study Plan Generator")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Generate Study Plan")
        course = st.selectbox("Course", courses['course_name'] if not courses.empty else ["None"])
        exam_date = st.date_input("Exam Date")
        days = (exam_date - datetime.now().date()).days
        if st.button("Generate Plan"):
            if days > 0 and course != "None":
                plan = generate_study_plan(exam_date.strftime('%Y-%m-%d'), course, days)
                add_study_plan(st.session_state.user_id, course, exam_date.strftime('%Y-%m-%d'), plan)
                st.session_state.current_plan = {
                    'course': course,
                    'exam_date': exam_date.strftime('%Y-%m-%d'),
                    'plan': plan
                }
                st.rerun()
            else:
                st.error("Select a future exam date and a valid course.")
        
        # Display current plan if exists
        if st.session_state.current_plan:
            st.markdown(f"""
            <div class='study-plan-card'>
                <h4><i class='fas fa-book-open'></i> Study Plan for {st.session_state.current_plan['course']}</h4>
                <p>{st.session_state.current_plan['plan']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Study Plan History")
        if not study_plans.empty:
            for _, plan in study_plans.iterrows():
                with st.expander(f"{plan['course_name']} - {plan['exam_date']}"):
                    st.markdown(f"""
                    <div class='study-plan-card'>
                        <p>{plan['plan_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No study plans generated yet.")

# Profile
elif st.session_state.page == "Profile":
    st.title("Your Profile")
    user_data = students.iloc[0] if not students.empty else None
    if user_data is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            try:
                # Display current avatar
                if os.path.exists(user_data['avatar']):
                    st.image(user_data['avatar'], width=200, caption="Current Avatar")
                else:
                    st.image("assets/default_avatar.png", width=200, caption="Default Avatar")
                
                # Add avatar upload
                new_avatar = st.file_uploader("Upload New Avatar", type=["png", "jpg", "jpeg"], key="profile_avatar")
                if new_avatar is not None:
                    try:
                        # Create a unique filename based on username
                        avatar_filename = f"assets/{st.session_state.user_id}_avatar.png"
                        # Save the uploaded file
                        with open(avatar_filename, "wb") as f:
                            f.write(new_avatar.getvalue())
                        
                        # Update the database
                        conn = sqlite3.connect('study_tracker.db')
                        c = conn.cursor()
                        c.execute('UPDATE students SET avatar = ? WHERE student_id = ?',
                                 (avatar_filename, st.session_state.user_id))
                        conn.commit()
                        
                        st.success("Avatar updated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error updating avatar: {e}")
            except Exception as e:
                st.error(f"Error loading avatar: {e}")
                st.image("assets/default_avatar.png", width=200, caption="Default Avatar")
        
        with col2:
            bio = st.text_area("Bio", user_data['bio'])
            linkedin = st.text_input("LinkedIn", user_data['linkedin'])
            instagram = st.text_input("Instagram", user_data['instagram'])
            st.write("Clubs:", ", ".join(clubs['club_name'].tolist()) if not clubs.empty else "None")
            if st.button("Update Profile"):
                conn = sqlite3.connect('study_tracker.db')
                c = conn.cursor()
                c.execute('UPDATE students SET bio = ?, linkedin = ?, instagram = ? WHERE student_id = ?',
                          (bio, linkedin, instagram, st.session_state.user_id))
                conn.commit()
                conn.close()
                st.success("Profile updated!")
    else:
        st.error("Profile data not found.")

# Events
elif st.session_state.page == "Events":
    st.title("Campus Events")
    
    # Load fresh data
    students, courses, tasks, grades, events, clubs, study_plans = load_data(st.session_state.user_id)
    
    if not events.empty and all(col in events.columns for col in ['name', 'date', 'organizer']):
        # Format the display
        events_display = events[events['verified']][['name', 'date', 'organizer']].copy()
        events_display.columns = ['Event Name', 'Date', 'Organizer']
        st.dataframe(events_display, use_container_width=True)
        
        # Add event joining functionality
        st.subheader("Join Event")
        event_to_join = st.selectbox("Select Event", events[events['verified']]['name'].tolist())
        if st.button("Join Event"):
            st.success(f"Successfully joined {event_to_join}!")
    else:
        st.info("No events available.")

# Admin Panel
elif st.session_state.page == "Admin":
    st.title("Admin Panel")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Verify Clubs")
        club_name = st.text_input("Club Name")
        if st.button("Verify Club"):
            conn = sqlite3.connect('study_tracker.db')
            c = conn.cursor()
            c.execute('INSERT INTO events (name, date, organizer, verified) VALUES (?, ?, ?, ?)',
                      (f"{club_name} Event", datetime.now().strftime('%Y-%m-%d'), club_name, True))
            conn.commit()
            conn.close()
            st.success(f"{club_name} verified!")
    
    with col2:
        st.subheader("Manage Events")
        if not events.empty and all(col in events.columns for col in ['name', 'date', 'organizer']):
            st.write(events[['name', 'date', 'organizer']].rename(columns={'name': 'Name', 'date': 'Date'}))
        else:
            st.info("No events available.")

if __name__ == "__main__":
    pass