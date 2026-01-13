"""Simple Flask app for exploring the research dataset."""

import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from flask import Flask, render_template, request, g, send_file

app = Flask(__name__)
DATABASE = '../data/research.db'


def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


@app.route('/')
def index():
    """Show summary stats and filters."""
    db = get_db()

    # Get counts
    stats = {}
    stats['total_posts'] = db.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
    stats['total_comments'] = db.execute('SELECT COUNT(*) FROM comments').fetchone()[0]
    stats['classified_posts'] = db.execute('SELECT COUNT(*) FROM post_classifications').fetchone()[0]
    stats['relationship_posts'] = db.execute(
        'SELECT COUNT(*) FROM post_classifications WHERE is_relationship_advice = 1'
    ).fetchone()[0]

    # Gender breakdown
    gender_counts = db.execute('''
        SELECT poster_gender, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1
        GROUP BY poster_gender
        ORDER BY count DESC
    ''').fetchall()

    # Relationship type breakdown
    type_counts = db.execute('''
        SELECT relationship_type, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1
        GROUP BY relationship_type
        ORDER BY count DESC
    ''').fetchall()

    # Severity breakdown
    severity_counts = db.execute('''
        SELECT situation_severity, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1 AND situation_severity IS NOT NULL
        GROUP BY situation_severity
        ORDER BY count DESC
    ''').fetchall()

    # Fault breakdown
    fault_counts = db.execute('''
        SELECT op_fault, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1 AND op_fault IS NOT NULL
        GROUP BY op_fault
        ORDER BY count DESC
    ''').fetchall()

    # Problem category breakdown
    category_counts = db.execute('''
        SELECT problem_category, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1 AND problem_category IS NOT NULL
        GROUP BY problem_category
        ORDER BY count DESC
    ''').fetchall()

    return render_template('index.html',
                          stats=stats,
                          gender_counts=gender_counts,
                          type_counts=type_counts,
                          severity_counts=severity_counts,
                          fault_counts=fault_counts,
                          category_counts=category_counts)


@app.route('/posts')
def posts():
    """List posts with filters."""
    db = get_db()

    # Get filter params
    gender = request.args.get('gender', '')
    rel_type = request.args.get('type', '')
    severity = request.args.get('severity', '')
    fault = request.args.get('fault', '')
    category = request.args.get('category', '')
    page = int(request.args.get('page', 1))
    per_page = 20

    # Build query
    where_clauses = ['pc.is_relationship_advice = 1']
    params = []

    if gender:
        where_clauses.append('pc.poster_gender = ?')
        params.append(gender)
    if rel_type:
        where_clauses.append('pc.relationship_type = ?')
        params.append(rel_type)
    if severity:
        where_clauses.append('pc.situation_severity = ?')
        params.append(severity)
    if fault:
        where_clauses.append('pc.op_fault = ?')
        params.append(fault)
    if category:
        where_clauses.append('pc.problem_category = ?')
        params.append(category)

    where_sql = ' AND '.join(where_clauses)

    # Get total count
    count_sql = f'''
        SELECT COUNT(*) FROM posts p
        JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE {where_sql}
    '''
    total = db.execute(count_sql, params).fetchone()[0]

    # Get posts
    offset = (page - 1) * per_page
    posts_sql = f'''
        SELECT p.post_id, p.title, p.author, p.timestamp,
               pc.poster_gender, pc.gender_confidence, pc.relationship_type,
               pc.brief_situation_summary, pc.situation_severity, pc.op_fault,
               pc.problem_category,
               (SELECT COUNT(*) FROM comments c WHERE c.post_id = p.post_id) as comment_count
        FROM posts p
        JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE {where_sql}
        ORDER BY p.timestamp DESC
        LIMIT ? OFFSET ?
    '''
    posts_list = db.execute(posts_sql, params + [per_page, offset]).fetchall()

    # Get filter options
    genders = db.execute('''
        SELECT DISTINCT poster_gender FROM post_classifications
        WHERE is_relationship_advice = 1 ORDER BY poster_gender
    ''').fetchall()
    types = db.execute('''
        SELECT DISTINCT relationship_type FROM post_classifications
        WHERE is_relationship_advice = 1 ORDER BY relationship_type
    ''').fetchall()
    severities = db.execute('''
        SELECT DISTINCT situation_severity FROM post_classifications
        WHERE is_relationship_advice = 1 AND situation_severity IS NOT NULL
        ORDER BY situation_severity
    ''').fetchall()
    faults = db.execute('''
        SELECT DISTINCT op_fault FROM post_classifications
        WHERE is_relationship_advice = 1 AND op_fault IS NOT NULL
        ORDER BY op_fault
    ''').fetchall()
    categories = db.execute('''
        SELECT DISTINCT problem_category FROM post_classifications
        WHERE is_relationship_advice = 1 AND problem_category IS NOT NULL
        ORDER BY problem_category
    ''').fetchall()

    total_pages = (total + per_page - 1) // per_page

    return render_template('posts.html',
                          posts=posts_list,
                          genders=genders,
                          types=types,
                          severities=severities,
                          faults=faults,
                          categories=categories,
                          current_gender=gender,
                          current_type=rel_type,
                          current_severity=severity,
                          current_fault=fault,
                          current_category=category,
                          page=page,
                          total_pages=total_pages,
                          total=total)


@app.route('/post/<post_id>')
def post_detail(post_id):
    """Show single post with all details and comments."""
    db = get_db()

    # Get post
    post = db.execute('''
        SELECT p.*, pc.poster_gender, pc.gender_confidence, pc.relationship_type,
               pc.brief_situation_summary, pc.situation_severity, pc.op_fault,
               pc.problem_category, pc.classification_model
        FROM posts p
        LEFT JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE p.post_id = ?
    ''', [post_id]).fetchone()

    if not post:
        return "Post not found", 404

    # Get comments
    comments_raw = db.execute('''
        SELECT c.*, cc.tone_labels, cc.advice_direction
        FROM comments c
        LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
        WHERE c.post_id = ?
        ORDER BY c.score DESC, c.timestamp ASC
    ''', [post_id]).fetchall()

    # Parse tone_labels JSON
    comments = []
    for c in comments_raw:
        comment = dict(c)
        if comment.get('tone_labels'):
            try:
                comment['tone_labels'] = json.loads(comment['tone_labels'])
            except json.JSONDecodeError:
                comment['tone_labels'] = []
        else:
            comment['tone_labels'] = []
        comments.append(comment)

    return render_template('post_detail.html', post=post, comments=comments)


@app.route('/analysis')
def analysis():
    """Show analysis dashboard with tables and charts."""
    db = get_db()

    # Dataset summary
    stats = {}
    stats['total_posts'] = db.execute('SELECT COUNT(*) FROM posts').fetchone()[0]
    stats['total_comments'] = db.execute('SELECT COUNT(*) FROM comments').fetchone()[0]
    stats['classified_posts'] = db.execute('SELECT COUNT(*) FROM post_classifications').fetchone()[0]
    stats['relationship_posts'] = db.execute(
        'SELECT COUNT(*) FROM post_classifications WHERE is_relationship_advice = 1'
    ).fetchone()[0]
    stats['classified_comments'] = db.execute('SELECT COUNT(*) FROM comment_classifications').fetchone()[0]

    # Gender distribution with confidence
    gender_data = db.execute('''
        SELECT poster_gender, COUNT(*) as count, AVG(gender_confidence) as avg_conf
        FROM post_classifications
        WHERE is_relationship_advice = 1
        GROUP BY poster_gender
        ORDER BY count DESC
    ''').fetchall()

    # Severity by gender crosstab
    severity_by_gender = db.execute('''
        SELECT poster_gender, situation_severity, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1
          AND poster_gender IN ('male', 'female')
          AND situation_severity IS NOT NULL
        GROUP BY poster_gender, situation_severity
    ''').fetchall()

    # Fault by gender crosstab
    fault_by_gender = db.execute('''
        SELECT poster_gender, op_fault, COUNT(*) as count
        FROM post_classifications
        WHERE is_relationship_advice = 1
          AND poster_gender IN ('male', 'female')
          AND op_fault IS NOT NULL
        GROUP BY poster_gender, op_fault
    ''').fetchall()

    # Tone frequencies by gender
    tone_data = db.execute('''
        SELECT pc.poster_gender, cc.tone_labels
        FROM comment_classifications cc
        JOIN comments c ON cc.comment_id = c.comment_id
        JOIN post_classifications pc ON c.post_id = pc.post_id
        WHERE pc.is_relationship_advice = 1
          AND pc.poster_gender IN ('male', 'female')
    ''').fetchall()

    # Count tones
    male_tones = defaultdict(int)
    female_tones = defaultdict(int)
    male_count = 0
    female_count = 0

    for row in tone_data:
        tones = json.loads(row['tone_labels']) if row['tone_labels'] else []
        if row['poster_gender'] == 'male':
            male_count += 1
            for t in tones:
                male_tones[t] += 1
        else:
            female_count += 1
            for t in tones:
                female_tones[t] += 1

    all_tones = sorted(set(male_tones.keys()) | set(female_tones.keys()))
    tone_table = []
    for tone in all_tones:
        tone_table.append({
            'tone': tone,
            'male_count': male_tones[tone],
            'female_count': female_tones[tone],
            'male_pct': f"{male_tones[tone]/male_count*100:.1f}%" if male_count > 0 else "0%",
            'female_pct': f"{female_tones[tone]/female_count*100:.1f}%" if female_count > 0 else "0%"
        })

    # Advice direction by gender
    direction_data = db.execute('''
        SELECT pc.poster_gender, cc.advice_direction, COUNT(*) as count
        FROM comment_classifications cc
        JOIN comments c ON cc.comment_id = c.comment_id
        JOIN post_classifications pc ON c.post_id = pc.post_id
        WHERE pc.is_relationship_advice = 1
          AND pc.poster_gender IN ('male', 'female')
          AND cc.advice_direction IS NOT NULL
        GROUP BY pc.poster_gender, cc.advice_direction
    ''').fetchall()

    # Check if HTML report exists
    report_path = Path(__file__).parent.parent / 'outputs' / 'report.html'
    report_exists = report_path.exists()

    return render_template('analysis.html',
                          stats=stats,
                          gender_data=gender_data,
                          severity_by_gender=severity_by_gender,
                          fault_by_gender=fault_by_gender,
                          tone_table=tone_table,
                          direction_data=direction_data,
                          male_count=male_count,
                          female_count=female_count,
                          report_exists=report_exists)


@app.route('/analysis/report')
def download_report():
    """Download the HTML analysis report."""
    report_path = Path(__file__).parent.parent / 'outputs' / 'report.html'
    if report_path.exists():
        return send_file(report_path, as_attachment=True, download_name='gender_bias_report.html')
    return "Report not generated yet. Run: python main.py --generate-report", 404


if __name__ == '__main__':
    app.run(debug=True, port=5001)
