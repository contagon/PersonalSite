from flask import Flask
from flask import render_template
app = Flask(__name__)

@app.route("/")
def home():
	return render_template('home.html')

@app.route("/pilot")
def pilot():
	return render_template('pilot.html')

@app.route("/hobbies")
def hobbies():
	return render_template('hobbies.html')

@app.errorhandler(404)
def page_not_found(e):
    return render_template('index.html'), 404

if __name__ == "__main__":
    app.run(debug=True)