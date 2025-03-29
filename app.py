from flask import Flask,render_template,request
import pandas as pd
from pipelines.data_cleaning import clean_data
from pipelines.data_standardization import standardize_data
from pipelines.column_selection import column_selection
app= Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
@app.route("/service",method = ["GET","POST"])
def service():
    file = request.files.get("file")  # Get the uploaded file
    
    if not file or file.filename == "":
        return "No file uploaded", 400

    # Check if the file is a CSV
    if not file.filename.lower().endswith(".csv"):
        return "Invalid file type! Please upload a CSV file.", 400

    df = pd.dataframe(file)
    print("Initiating Data Cleaning..")
    cleaned_df = clean_data(df)
    print("Completed Data Cleaning !")
    print("Initiating Data Standardization..")
    standardized_df = standardize_data(cleaned_df)
    print("Completed Data Standardization !")
    print("Initiating Column Selection & Finalization..")
    final_df = column_selection(standardized_df)
    print("Completed Column Selection & Finalization !")

    return render_template("service.html")
@app.route("/plans")
def plans():
    return render_template("plans.html")


if __name__=="__main__":
    app.run(debug=True)
