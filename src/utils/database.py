"""Database utilities and schema management."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager

from .logging import get_logger

logger = get_logger("database")

# Default database path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "research.db"


SCHEMA = """
-- Source forums
CREATE TABLE IF NOT EXISTS sources (
    source_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    url TEXT
);

-- Original posts/questions
CREATE TABLE IF NOT EXISTS posts (
    post_id TEXT PRIMARY KEY,
    source_id INTEGER REFERENCES sources(source_id),
    title TEXT,
    body TEXT,
    author TEXT,
    timestamp DATETIME,
    flair TEXT,
    raw_json TEXT
);

-- Comments/answers
CREATE TABLE IF NOT EXISTS comments (
    comment_id TEXT PRIMARY KEY,
    post_id TEXT REFERENCES posts(post_id),
    body TEXT,
    author TEXT,
    score INTEGER,
    timestamp DATETIME,
    is_top_level BOOLEAN
);

-- LLM classification of posts
CREATE TABLE IF NOT EXISTS post_classifications (
    post_id TEXT PRIMARY KEY REFERENCES posts(post_id),
    is_relationship_advice BOOLEAN,
    poster_gender TEXT,
    gender_confidence FLOAT,
    relationship_type TEXT,
    brief_situation_summary TEXT,
    situation_severity TEXT,
    op_fault TEXT,
    problem_category TEXT,
    classification_model TEXT,
    classification_timestamp DATETIME
);

-- LLM classification of comments
CREATE TABLE IF NOT EXISTS comment_classifications (
    comment_id TEXT PRIMARY KEY REFERENCES comments(comment_id),
    is_advice BOOLEAN,
    advice_direction TEXT,
    tone_labels TEXT,
    classification_model TEXT,
    classification_timestamp DATETIME
);

-- Pairwise comparison results for Bradley-Terry scoring
CREATE TABLE IF NOT EXISTS pairwise_comparisons (
    comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id_a TEXT REFERENCES comments(comment_id),
    comment_id_b TEXT REFERENCES comments(comment_id),
    dimension TEXT,
    winner TEXT,
    reasoning TEXT,
    comparison_model TEXT,
    comparison_timestamp DATETIME
);

-- Computed latent scores
CREATE TABLE IF NOT EXISTS comment_scores (
    comment_id TEXT PRIMARY KEY REFERENCES comments(comment_id),
    harshness_score FLOAT,
    supportiveness_score FLOAT,
    constructiveness_score FLOAT,
    score_confidence FLOAT,
    last_updated DATETIME
);

-- Human validation of LLM classifications
CREATE TABLE IF NOT EXISTS classification_validations (
    validation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    comment_id TEXT NOT NULL REFERENCES comments(comment_id),
    field_name TEXT NOT NULL,
    llm_value TEXT NOT NULL,
    human_judgment TEXT NOT NULL,
    human_correction TEXT,
    notes TEXT,
    validated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_posts_source ON posts(source_id);
CREATE INDEX IF NOT EXISTS idx_comments_post ON comments(post_id);
CREATE INDEX IF NOT EXISTS idx_post_class_gender ON post_classifications(poster_gender);
CREATE INDEX IF NOT EXISTS idx_post_class_relationship ON post_classifications(is_relationship_advice);
CREATE INDEX IF NOT EXISTS idx_post_class_severity ON post_classifications(situation_severity);
CREATE INDEX IF NOT EXISTS idx_post_class_fault ON post_classifications(op_fault);
CREATE INDEX IF NOT EXISTS idx_post_class_category ON post_classifications(problem_category);
CREATE INDEX IF NOT EXISTS idx_comment_class_direction ON comment_classifications(advice_direction);
CREATE INDEX IF NOT EXISTS idx_pairwise_dimension ON pairwise_comparisons(dimension);
CREATE INDEX IF NOT EXISTS idx_validations_comment ON classification_validations(comment_id);
"""

# Initial source data
SOURCES = [
    (1, "amioverreacting", "https://reddit.com/r/AmIOverreacting"),
    (2, "amitheasshole", "https://reddit.com/r/AmItheAsshole"),
    (3, "askmetafilter", "https://ask.metafilter.com"),
]


class Database:
    """Database manager for the research pipeline."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def initialize(self):
        """Initialize the database schema and seed data."""
        logger.info(f"Initializing database at {self.db_path}")

        with self.get_connection() as conn:
            conn.executescript(SCHEMA)

            # Insert sources if not exist
            for source in SOURCES:
                conn.execute(
                    "INSERT OR IGNORE INTO sources (source_id, name, url) VALUES (?, ?, ?)",
                    source
                )

        logger.info("Database initialized successfully")

    def migrate_add_confound_columns(self):
        """Add situation_severity, op_fault, problem_category columns if they don't exist."""
        logger.info("Running migration to add confound control columns")
        with self.get_connection() as conn:
            # Check existing columns
            cursor = conn.execute("PRAGMA table_info(post_classifications)")
            existing_columns = {row['name'] for row in cursor.fetchall()}

            if 'situation_severity' not in existing_columns:
                conn.execute("ALTER TABLE post_classifications ADD COLUMN situation_severity TEXT")
                logger.info("Added situation_severity column")
            if 'op_fault' not in existing_columns:
                conn.execute("ALTER TABLE post_classifications ADD COLUMN op_fault TEXT")
                logger.info("Added op_fault column")
            if 'problem_category' not in existing_columns:
                conn.execute("ALTER TABLE post_classifications ADD COLUMN problem_category TEXT")
                logger.info("Added problem_category column")

            # Create indexes if not exist
            conn.execute("CREATE INDEX IF NOT EXISTS idx_post_class_severity ON post_classifications(situation_severity)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_post_class_fault ON post_classifications(op_fault)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_post_class_category ON post_classifications(problem_category)")

        logger.info("Migration complete")

    def get_source_id(self, name: str) -> int:
        """Get source ID by name."""
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT source_id FROM sources WHERE name = ?", (name,)
            ).fetchone()
            if result:
                return result[0]
            raise ValueError(f"Unknown source: {name}")

    # Post operations
    def insert_post(
        self,
        post_id: str,
        source_id: int,
        title: str,
        body: str,
        author: str,
        timestamp: datetime,
        flair: Optional[str] = None,
        raw_json: Optional[dict] = None
    ) -> bool:
        """Insert a post, returning True if inserted, False if already exists."""
        with self.get_connection() as conn:
            try:
                conn.execute(
                    """INSERT INTO posts
                    (post_id, source_id, title, body, author, timestamp, flair, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (post_id, source_id, title, body, author, timestamp, flair,
                     json.dumps(raw_json) if raw_json else None)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_post(self, post_id: str) -> Optional[Dict]:
        """Get a post by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM posts WHERE post_id = ?", (post_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_posts_by_source(self, source_id: int, limit: Optional[int] = None) -> List[Dict]:
        """Get all posts for a source."""
        with self.get_connection() as conn:
            query = "SELECT * FROM posts WHERE source_id = ?"
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query, (source_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_unclassified_posts(self, limit: Optional[int] = None) -> List[Dict]:
        """Get posts that haven't been classified yet."""
        with self.get_connection() as conn:
            query = """
                SELECT p.* FROM posts p
                LEFT JOIN post_classifications pc ON p.post_id = pc.post_id
                WHERE pc.post_id IS NULL
            """
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def get_relationship_posts(self, gender: Optional[str] = None) -> List[Dict]:
        """Get posts classified as relationship advice, optionally filtered by gender."""
        with self.get_connection() as conn:
            query = """
                SELECT p.*, pc.poster_gender, pc.gender_confidence,
                       pc.relationship_type, pc.brief_situation_summary,
                       pc.situation_severity, pc.op_fault, pc.problem_category
                FROM posts p
                JOIN post_classifications pc ON p.post_id = pc.post_id
                WHERE pc.is_relationship_advice = 1
            """
            params = []
            if gender:
                query += " AND pc.poster_gender = ?"
                params.append(gender)

            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def count_posts(self, source_id: Optional[int] = None) -> int:
        """Count posts, optionally by source."""
        with self.get_connection() as conn:
            if source_id:
                result = conn.execute(
                    "SELECT COUNT(*) FROM posts WHERE source_id = ?", (source_id,)
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) FROM posts").fetchone()
            return result[0]

    # Comment operations
    def insert_comment(
        self,
        comment_id: str,
        post_id: str,
        body: str,
        author: str,
        score: int,
        timestamp: datetime,
        is_top_level: bool
    ) -> bool:
        """Insert a comment, returning True if inserted, False if already exists."""
        with self.get_connection() as conn:
            try:
                conn.execute(
                    """INSERT INTO comments
                    (comment_id, post_id, body, author, score, timestamp, is_top_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (comment_id, post_id, body, author, score, timestamp, is_top_level)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_comments_for_post(self, post_id: str, top_level_only: bool = False) -> List[Dict]:
        """Get all comments for a post."""
        with self.get_connection() as conn:
            query = "SELECT * FROM comments WHERE post_id = ?"
            if top_level_only:
                query += " AND is_top_level = 1"
            query += " ORDER BY score DESC"
            rows = conn.execute(query, (post_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_unclassified_comments(
        self,
        relationship_posts_only: bool = True,
        filter_for_analysis: bool = True,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Get comments that haven't been classified yet.

        Args:
            relationship_posts_only: Only include comments on relationship posts
            filter_for_analysis: Apply strict filter criteria for analysis:
                - problem_category IS NOT NULL (advice-seeking posts)
                - poster_gender IN ('male', 'female') (known binary gender)
                - gender_confidence > 0.7 (high confidence)
            limit: Maximum number of comments to return
        """
        with self.get_connection() as conn:
            if relationship_posts_only:
                query = """
                    SELECT c.*, p.title as post_title, p.body as post_body,
                           pc.brief_situation_summary, pc.poster_gender
                    FROM comments c
                    JOIN posts p ON c.post_id = p.post_id
                    JOIN post_classifications pc ON p.post_id = pc.post_id
                    LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
                    WHERE cc.comment_id IS NULL
                    AND pc.is_relationship_advice = 1
                """
                if filter_for_analysis:
                    query += """
                    AND pc.problem_category IS NOT NULL
                    AND pc.poster_gender IN ('male', 'female')
                    AND pc.gender_confidence > 0.7
                """
            else:
                query = """
                    SELECT c.* FROM comments c
                    LEFT JOIN comment_classifications cc ON c.comment_id = cc.comment_id
                    WHERE cc.comment_id IS NULL
                """
            if limit:
                query += f" LIMIT {limit}"
            rows = conn.execute(query).fetchall()
            return [dict(row) for row in rows]

    def count_comments(self, post_id: Optional[str] = None) -> int:
        """Count comments, optionally for a specific post."""
        with self.get_connection() as conn:
            if post_id:
                result = conn.execute(
                    "SELECT COUNT(*) FROM comments WHERE post_id = ?", (post_id,)
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) FROM comments").fetchone()
            return result[0]

    # Classification operations
    def insert_post_classification(
        self,
        post_id: str,
        is_relationship_advice: bool,
        poster_gender: str,
        gender_confidence: float,
        relationship_type: str,
        brief_situation_summary: str,
        classification_model: str,
        situation_severity: Optional[str] = None,
        op_fault: Optional[str] = None,
        problem_category: Optional[str] = None
    ):
        """Insert or update a post classification."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO post_classifications
                (post_id, is_relationship_advice, poster_gender, gender_confidence,
                 relationship_type, brief_situation_summary, situation_severity,
                 op_fault, problem_category, classification_model,
                 classification_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (post_id, is_relationship_advice, poster_gender, gender_confidence,
                 relationship_type, brief_situation_summary, situation_severity,
                 op_fault, problem_category, classification_model,
                 datetime.now())
            )

    def insert_comment_classification(
        self,
        comment_id: str,
        is_advice: bool,
        advice_direction: Optional[str],
        tone_labels: List[str],
        classification_model: str
    ):
        """Insert or update a comment classification."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO comment_classifications
                (comment_id, is_advice, advice_direction, tone_labels,
                 classification_model, classification_timestamp)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (comment_id, is_advice, advice_direction,
                 json.dumps(tone_labels), classification_model, datetime.now())
            )

    # Pairwise comparison operations
    def insert_pairwise_comparison(
        self,
        comment_id_a: str,
        comment_id_b: str,
        dimension: str,
        winner: str,
        reasoning: str,
        comparison_model: str
    ):
        """Insert a pairwise comparison result."""
        with self.get_connection() as conn:
            conn.execute(
                """INSERT INTO pairwise_comparisons
                (comment_id_a, comment_id_b, dimension, winner, reasoning,
                 comparison_model, comparison_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (comment_id_a, comment_id_b, dimension, winner, reasoning,
                 comparison_model, datetime.now())
            )

    def get_pairwise_comparisons(self, dimension: str) -> List[Dict]:
        """Get all pairwise comparisons for a dimension."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM pairwise_comparisons WHERE dimension = ?",
                (dimension,)
            ).fetchall()
            return [dict(row) for row in rows]

    def update_comment_scores(
        self,
        comment_id: str,
        harshness_score: Optional[float] = None,
        supportiveness_score: Optional[float] = None,
        constructiveness_score: Optional[float] = None,
        score_confidence: Optional[float] = None
    ):
        """Update computed scores for a comment."""
        with self.get_connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT * FROM comment_scores WHERE comment_id = ?",
                (comment_id,)
            ).fetchone()

            if existing:
                updates = []
                params = []
                if harshness_score is not None:
                    updates.append("harshness_score = ?")
                    params.append(harshness_score)
                if supportiveness_score is not None:
                    updates.append("supportiveness_score = ?")
                    params.append(supportiveness_score)
                if constructiveness_score is not None:
                    updates.append("constructiveness_score = ?")
                    params.append(constructiveness_score)
                if score_confidence is not None:
                    updates.append("score_confidence = ?")
                    params.append(score_confidence)
                updates.append("last_updated = ?")
                params.append(datetime.now())
                params.append(comment_id)

                conn.execute(
                    f"UPDATE comment_scores SET {', '.join(updates)} WHERE comment_id = ?",
                    params
                )
            else:
                conn.execute(
                    """INSERT INTO comment_scores
                    (comment_id, harshness_score, supportiveness_score,
                     constructiveness_score, score_confidence, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    (comment_id, harshness_score, supportiveness_score,
                     constructiveness_score, score_confidence, datetime.now())
                )

    # Analysis queries
    def get_analysis_data(self) -> List[Dict]:
        """Get combined data for statistical analysis."""
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    c.comment_id,
                    c.post_id,
                    c.body as comment_body,
                    c.score as comment_score,
                    p.title as post_title,
                    s.name as source_name,
                    pc.poster_gender,
                    pc.gender_confidence,
                    pc.relationship_type,
                    pc.situation_severity,
                    pc.op_fault,
                    pc.problem_category,
                    cc.is_advice,
                    cc.advice_direction,
                    cc.tone_labels,
                    cs.harshness_score,
                    cs.supportiveness_score,
                    cs.constructiveness_score
                FROM comments c
                JOIN posts p ON c.post_id = p.post_id
                JOIN sources s ON p.source_id = s.source_id
                JOIN post_classifications pc ON p.post_id = pc.post_id
                JOIN comment_classifications cc ON c.comment_id = cc.comment_id
                LEFT JOIN comment_scores cs ON c.comment_id = cs.comment_id
                WHERE pc.is_relationship_advice = 1
                AND cc.is_advice = 1
                AND pc.poster_gender IN ('male', 'female')
            """).fetchall()
            return [dict(row) for row in rows]

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for the database."""
        with self.get_connection() as conn:
            stats = {}

            # Total counts
            stats["total_posts"] = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            stats["total_comments"] = conn.execute("SELECT COUNT(*) FROM comments").fetchone()[0]

            # Posts by source
            rows = conn.execute("""
                SELECT s.name, COUNT(*) as count
                FROM posts p
                JOIN sources s ON p.source_id = s.source_id
                GROUP BY s.name
            """).fetchall()
            stats["posts_by_source"] = {row["name"]: row["count"] for row in rows}

            # Classification stats
            stats["classified_posts"] = conn.execute(
                "SELECT COUNT(*) FROM post_classifications"
            ).fetchone()[0]
            stats["classified_comments"] = conn.execute(
                "SELECT COUNT(*) FROM comment_classifications"
            ).fetchone()[0]

            # Relationship posts by gender
            rows = conn.execute("""
                SELECT poster_gender, COUNT(*) as count
                FROM post_classifications
                WHERE is_relationship_advice = 1
                GROUP BY poster_gender
            """).fetchall()
            stats["relationship_posts_by_gender"] = {row["poster_gender"]: row["count"] for row in rows}

            # Pairwise comparisons
            rows = conn.execute("""
                SELECT dimension, COUNT(*) as count
                FROM pairwise_comparisons
                GROUP BY dimension
            """).fetchall()
            stats["pairwise_comparisons_by_dimension"] = {row["dimension"]: row["count"] for row in rows}

            return stats

    # Validation operations
    def get_random_unvalidated_comment(
        self,
        gender_filter: Optional[str] = None,
        direction_filter: Optional[str] = None
    ) -> Optional[Dict]:
        """Get a random classified comment that hasn't been validated yet."""
        with self.get_connection() as conn:
            query = """
                SELECT c.comment_id, c.body as comment_body, c.author, c.score,
                       p.post_id, p.title as post_title, p.body as post_body,
                       pc.poster_gender, pc.gender_confidence, pc.brief_situation_summary,
                       pc.situation_severity, pc.op_fault, pc.problem_category,
                       cc.advice_direction, cc.tone_labels, cc.is_advice
                FROM comments c
                JOIN posts p ON c.post_id = p.post_id
                JOIN post_classifications pc ON p.post_id = pc.post_id
                JOIN comment_classifications cc ON c.comment_id = cc.comment_id
                WHERE cc.is_advice = 1
                AND c.comment_id NOT IN (
                    SELECT DISTINCT comment_id FROM classification_validations
                )
            """
            params = []

            if gender_filter and gender_filter != 'all':
                query += " AND pc.poster_gender = ?"
                params.append(gender_filter)

            if direction_filter and direction_filter != 'all':
                query += " AND cc.advice_direction = ?"
                params.append(direction_filter)

            query += " ORDER BY RANDOM() LIMIT 1"

            row = conn.execute(query, params).fetchone()
            if row:
                result = dict(row)
                # Parse tone_labels JSON
                if result.get('tone_labels'):
                    result['tone_labels'] = json.loads(result['tone_labels'])
                return result
            return None

    def save_validation(
        self,
        comment_id: str,
        field_name: str,
        llm_value: str,
        human_judgment: str,
        human_correction: Optional[str] = None,
        notes: Optional[str] = None
    ) -> int:
        """Save a validation judgment."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO classification_validations
                (comment_id, field_name, llm_value, human_judgment, human_correction, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (comment_id, field_name, llm_value, human_judgment, human_correction, notes))
            return cursor.lastrowid

    def get_validation_stats(self) -> Dict:
        """Get validation statistics."""
        with self.get_connection() as conn:
            stats = {}

            # Total validations
            total = conn.execute(
                "SELECT COUNT(*) FROM classification_validations"
            ).fetchone()[0]
            stats['total_validations'] = total

            # Unique comments validated
            unique_comments = conn.execute(
                "SELECT COUNT(DISTINCT comment_id) FROM classification_validations"
            ).fetchone()[0]
            stats['unique_comments_validated'] = unique_comments

            # Total classified comments
            total_classified = conn.execute(
                "SELECT COUNT(*) FROM comment_classifications WHERE is_advice = 1"
            ).fetchone()[0]
            stats['total_classified'] = total_classified

            # Agreement rates by field
            for field in ['advice_direction', 'tone_labels']:
                row = conn.execute("""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN human_judgment = 'correct' THEN 1 ELSE 0 END) as correct,
                        SUM(CASE WHEN human_judgment = 'incorrect' THEN 1 ELSE 0 END) as incorrect,
                        SUM(CASE WHEN human_judgment = 'uncertain' THEN 1 ELSE 0 END) as uncertain
                    FROM classification_validations
                    WHERE field_name = ?
                """, (field,)).fetchone()

                if row and row['total'] > 0:
                    stats[f'{field}_total'] = row['total']
                    stats[f'{field}_correct'] = row['correct']
                    stats[f'{field}_incorrect'] = row['incorrect']
                    stats[f'{field}_uncertain'] = row['uncertain']
                    stats[f'{field}_agreement'] = row['correct'] / row['total'] if row['total'] > 0 else 0

            return stats

    def get_comment_for_validation(self, comment_id: str) -> Optional[Dict]:
        """Get a specific comment with all context for validation."""
        with self.get_connection() as conn:
            row = conn.execute("""
                SELECT c.comment_id, c.body as comment_body, c.author, c.score,
                       p.post_id, p.title as post_title, p.body as post_body,
                       pc.poster_gender, pc.gender_confidence, pc.brief_situation_summary,
                       pc.situation_severity, pc.op_fault, pc.problem_category,
                       cc.advice_direction, cc.tone_labels, cc.is_advice
                FROM comments c
                JOIN posts p ON c.post_id = p.post_id
                JOIN post_classifications pc ON p.post_id = pc.post_id
                JOIN comment_classifications cc ON c.comment_id = cc.comment_id
                WHERE c.comment_id = ?
            """, (comment_id,)).fetchone()

            if row:
                result = dict(row)
                if result.get('tone_labels'):
                    result['tone_labels'] = json.loads(result['tone_labels'])
                return result
            return None


# Global database instance
db = Database()


def initialize_database():
    """Initialize the database."""
    db.initialize()


if __name__ == "__main__":
    initialize_database()
    print("Database initialized successfully")
    print(f"Database path: {db.db_path}")
