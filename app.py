from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import os
from pipelines.data_cleaning import clean_data
from pipelines.data_standardization import standardize_data
from pipelines.column_selection import column_selection
from components.report_generation import generate_report
from pipelines.data_generation import generate_synthetic_data
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

            return jsonify({
                "message": "File processed successfully", 
                "download_link": "/download",
                "report_link": "/generate_report"
            })

        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return render_template("service.html")

@app.route("/generate_report")
def generate_report_route():
    try:
        # Read the processed data
        processed_file = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
        if not os.path.exists(processed_file):
            return jsonify({"error": "Processed file not found"}), 404
            
        df = pd.read_csv(processed_file)
        
        # Generate report using Gemini
        report = generate_report(df)
        
        # Save report to file with UTF-8 encoding
        report_file = os.path.join(UPLOAD_FOLDER, "report.md")
        with open(report_file, "w", encoding='utf-8', errors='ignore') as f:
            f.write(report)
            
        return jsonify({
            "message": "Report generated successfully",
            "report_link": "/download_report"
        })
    except Exception as e:
        return jsonify({"error": f"Report generation failed: {str(e)}"}), 500

@app.route("/download")
def download_file():
    output_file = os.path.join(UPLOAD_FOLDER, "processed_data.csv")
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404

@app.route("/download_report")
def download_report():
    report_file = os.path.join(UPLOAD_FOLDER, "report.md")
    if os.path.exists(report_file):
        return send_file(
            report_file,
            as_attachment=True,
            mimetype='text/markdown',
            download_name='report.md'
        )
    return "Report not found", 404

@app.route("/plans")
def plans():
    return render_template("plans.html")

@app.route("/generate", methods=["GET", "POST"])
def generate():
    if request.method == "POST":
        file = request.files.get("file")
        num_points = int(request.form.get("num_points", 1000))
        
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        if not file.filename.lower().endswith(".csv"):
            return jsonify({"error": "Invalid file type! Please upload a CSV file."}), 400

        try:
            # Read the sample dataset
            df = pd.read_csv(file)
            
            # Generate synthetic data
            synthetic_df = generate_synthetic_data(df, num_points)
            
            # Save generated data
            output_filename = os.path.join(UPLOAD_FOLDER, "generated_data.csv")
            synthetic_df.to_csv(output_filename, index=False)
            
            return jsonify({
                "message": "Data generated successfully",
                "download_link": "/download_generated"
            })
            
        except Exception as e:
            return jsonify({"error": f"Generation failed: {str(e)}"}), 500
            
    return render_template("generate.html")

@app.route("/download_generated")
def download_generated():
    output_file = os.path.join(UPLOAD_FOLDER, "generated_data.csv")
    if os.path.exists(output_file):
        return send_file(output_file, as_attachment=True)
    return "File not found", 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True)
    #app.run(host="0.0.0.0",debug=True,port=port)
