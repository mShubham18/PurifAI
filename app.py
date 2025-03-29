from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
from pipelines.data_cleaning import clean_data
from pipelines.data_standardization import standardize_data
from pipelines.column_selection import column_selection
app = Flask(__name__)

UPLOAD_FOLDER = "processed_files"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/service", methods=["GET", "POST"])
def service():
    summary = None  # Ensure summary is always defined

    if request.method == "POST":
        file = request.files.get("file")  # Get uploaded file
        
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        # Check if the file is a CSV
        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Invalid file type! Please upload a CSV file."}), 400

        try:
            df = pd.read_csv(file)  # Read CSV into DataFrame

            print("Initiating Data Cleaning..")
            cleaned_df = clean_data(df)
            print("Completed Data Cleaning!")
            
            print("Initiating Data Standardization..")
            standardized_df = standardize_data(cleaned_df)
            print("Completed Data Standardization!")
            
            print("Initiating Column Selection & Finalization..")
            final_df = column_selection(standardized_df)
            print("Completed Column Selection & Finalization!")

            # Save final DataFrame as CSV
            output_filename = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
            final_df.to_csv(output_filename, index=False)

            return jsonify({"message": "File processed successfully", "download_link": "/download"})

        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return render_template("service.html")


@app.route("/download")
def download_file():
    output_file = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404

@app.route("/plans")
def plans():
    return render_template("plans.html")

if __name__ == "__main__":
    app.run(debug=True)
