from flask import Flask, render_template, redirect

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start():
    return render_template('start.html', ans=None)

@app.route('/live')
def live():
    return redirect('https://www.google.com')

@app.route('/how-to-use')
def how_to_use():
    return 'This is the How to Use page'

if __name__ == '__main__':
    app.run(debug=True)