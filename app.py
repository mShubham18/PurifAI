from flask import Flask,render_template

app= Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/service")
def service():
    return render_template("service.html")
@app.route("/plans")
def plans():
    return render_template("plans.html")


if __name__=="__main__":
    app.run(debug=True)
