from components.model_configuration import model_config
def generate_report(df):
    model = model_config()
    prompt = f"""I have a dataset, analyze it and create a basic pointed summary under 200 words for it covering entire info about the dataset: {df}"""
    response = model.generate_content(prompt)
    summary = response.text
    return summary