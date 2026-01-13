"""Simple Flask app for exploring the research dataset."""

import sqlite3
from flask import Flask, render_template, request, g

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

    return render_template('index.html', stats=stats, gender_counts=gender_counts, type_counts=type_counts)


@app.route('/posts')
def posts():
    """List posts with filters."""
    db = get_db()

    # Get filter params
    gender = request.args.get('gender', '')
    rel_type = request.args.get('type', '')
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
               pc.brief_situation_summary,
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

    total_pages = (total + per_page - 1) // per_page

    return render_template('posts.html',
                          posts=posts_list,
                          genders=genders,
                          types=types,
                          current_gender=gender,
                          current_type=rel_type,
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
               pc.brief_situation_summary, pc.classification_model
        FROM posts p
        LEFT JOIN post_classifications pc ON p.post_id = pc.post_id
        WHERE p.post_id = ?
    ''', [post_id]).fetchone()

    if not post:
        return "Post not found", 404

    # Get comments
    comments = db.execute('''
        SELECT c.*, cc.harshness_score, cc.tone_labels, cc.advice_direction,
               cc.blame_assignment
        FROM comments c
        LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
        WHERE c.post_id = ?
        ORDER BY c.favorites DESC, c.timestamp ASC
    ''', [post_id]).fetchall()

    return render_template('post_detail.html', post=post, comments=comments)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
