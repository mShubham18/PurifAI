from pipelines.data_cleaning import clean_data
from pipelines.data_standardization import standardize_data
import pandas as pd
def column_selection(df):
    sample=df.head(2)
    sample_df_dict = sample.to_dict()
    prompt=f"""I'm providing you with a list of column and some snippet of its data, analyze its context and assume its cleaned . The main purpose of this is only to generate a cleaned and standardized dataset to feed directly into a ml model.
    , i need you to provide me with a list  of columns that i should keep after dropping which are unneccessary. The output must be a list containing the columns that i must keep, comma seperated under quotes, do not enclose it under square or any braces do not use any comments, any salutaion. just give me the list that's it

    example output: col1,col2

    columns_list = {list(df.columns)}

    sample_data: {sample_df_dict}"""

    from components.model_configuration import model_config
    model = model_config()

    response = model.generate_content(prompt)
    keep_column_list = response.text
    keep_column_list = keep_column_list.split(",")[:-1]

    keep_column_list = [col.replace("'","") for col in keep_column_list]
    final_dict = {key:value for key,value in df.items() if key in keep_column_list}
    final_df=pd.DataFrame(final_dict)
    return final_df
