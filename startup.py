import sys
from flask import Flask
from flask import render_template, request, redirect, render_template_string, Markup
from datetime import date
from flask_flatpages import FlatPages, pygments_style_defs
from flask_frozen import Freezer
from urllib.parse import urlparse, urlunparse
import markdown

BASE_URL = 'https://www.eastonpots.com'
DEBUG = True
FLATPAGES_AUTO_RELOAD = DEBUG
FLATPAGES_EXTENSION = '.md'
FLATPAGES_ROOT = 'content'
POST_DIR = 'posts'
PROJECT_DIR = 'projects'

FLATPAGES_MARKDOWN_EXTENSIONS = ['codehilite', 'fenced_code',]

app = Flask(__name__)
flatpages = FlatPages(app)
freezer = Freezer(app)

app.config.from_object(__name__)



@app.context_processor
def inject_ga():
    return dict(BASE_URL=BASE_URL)

@app.route('/pygments.css')
def pygments_css():
    return pygments_style_defs('native'), 200, {'Content-Type': 'text/css'}

@app.route("/")
def home():
    posts = [p for p in flatpages if p.path.startswith(POST_DIR)]
    posts.sort(key=lambda item: item['date'], reverse=True)
    return render_template('home.html', posts=posts, bigheader=True)

@app.route("/aboutme/")
def aboutme():
    return render_template('aboutme.html')

@app.route("/projects/")
def projects():
    return render_template('project_index.html')

@app.route("/posts/")
def post_index():
    return render_template('post_index.html')

@app.route('/posts/<name>/')
def post(name):
    path = '{}/{}'.format(POST_DIR, name)
    post = flatpages.get_or_404(path)
    return render_template('post.html', post=post)

@app.route('/robots.txt')
def robots():
    return render_template('robots.txt')

# @app.route('/sitemap.xml')
# def sitemap():
#     today = date.today()
#     recently = date(year=today.year, month=today.month, day=1)
#     posts = [p for p in flatpages if p.path.startswith(POST_DIR)]
#     posts.sort(key=lambda item: item['date'], reverse=True)
#     return render_template('sitemap.xml', posts=posts, today=today, recently=recently)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        freezer.freeze()
    else:
        app.run(host='0.0.0.0', port=8888, debug=True)
