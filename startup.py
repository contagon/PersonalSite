import sys, os, re
from flask import Flask, render_template
from datetime import date
from flask_flatpages import FlatPages, pygments_style_defs
from flask_frozen import Freezer

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

def convert_jupyter(type, file):
    #do initial conversions
    os.system(f'jupyter nbconvert --to markdown content/{type}/{file}')

    #remove previous and move data to static
    filename = file.split(".")[0]
    files = fr'{filename}_files'
    os.system(f'rm -rf static/{type}/{files}')
    os.system(f'mv content/{type}/{files} static/{type}')

    #open and replace various things so it works
    with open(f'content/{type}/{filename}.md', 'r') as reader:
        md = reader.read()
    md = re.sub(r'\\\\', r'\\\\\\\\', md)
    md = re.sub(r'_{', '\_{', md)
    md = re.sub(files, f'/static/{type}/{files}', md)
    with open(f'content/{type}/{filename}.md', 'w') as writer:
        writer.write(md)

def convert_all():
    #convert all posts
    for filename in os.listdir("content/posts/"):
        if filename.endswith(".ipynb"):
            convert_jupyter('posts', filename)

    #convert all projects
    for filename in os.listdir("content/projects/"):
        if filename.endswith(".ipynb"):
            convert_jupyter('projects', filename)

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
    projects = [p for p in flatpages if p.path.startswith(PROJECT_DIR)]
    projects.sort(key=lambda item: item['date'], reverse=True)
    return render_template('home.html', posts=posts, projects=projects)

@app.route("/aboutme/")
def aboutme():
    return render_template('aboutme.html')


@app.route("/posts/")
def post_index():
    return render_template('post_index.html')

@app.route('/posts/<name>/')
def post(name):
    path = '{}/{}'.format(POST_DIR, name)
    post = flatpages.get_or_404(path)
    return render_template('post.html', post=post)

@app.route("/projects/")
def project_index():
    return render_template('project_index.html')

@app.route('/projects/<name>/')
def project(name):
    path = '{}/{}'.format(PROJECT_DIR, name)
    project = flatpages.get_or_404(path)
    return render_template('project.html', project=project)

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
    if "convert" in sys.argv:
        convert_all()
    if "build" in sys.argv:
        freezer.freeze()
    if "test" in sys.argv:
        app.run(debug=True)#host='0.0.0.0', port=8888, debug=True)
