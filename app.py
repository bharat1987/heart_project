import gradio as gr
import joblib
import pandas as pd
import os
from fastapi import FastAPI, Request, Response

app = FastAPI()
absolute_path = os.path.dirname(os.path.abspath(__file__))
save_file_name=os.path.join(absolute_path,"trained_model","xgboost-model.pkl")
heart_model= joblib.load(filename=save_file_name)


def predict_death_event(age_gr,anaemia_gr,creatinine_phosphokinase_gr,diabetes_gr,ejection_fraction_gr,
                                   high_blood_pressure_gr,platelets_gr,serum_creatinine_gr,serum_sodium_gr,
                                   sex_gr,smoking_gr,time_gr):

    # YOUR CODE HERE...
    print(diabetes_gr)
    input_df = pd.DataFrame({"age": [int(age_gr)], 
                            "anaemia": [int(anaemia_gr)], 
                            "creatinine_phosphokinase": [int(creatinine_phosphokinase_gr)],
                            "diabetes": [int(diabetes_gr)], 
                            "ejection_fraction": [int(ejection_fraction_gr)], 
                            "high_blood_pressure": [int(high_blood_pressure_gr)],
                            "platelets": [int(platelets_gr)], 
                            "serum_creatinine": [int(serum_creatinine_gr)], 
                            "serum_sodium": [int(serum_sodium_gr)],
                            "sex": [int(sex_gr)], 
                            "smoking":[int(smoking_gr)],
                            "time":[int(time_gr)]})

    predictions = heart_model.predict(input_df)
    print(predictions)
    label = "High Risk" if predictions[0]==1 else "Low Risk"
    return label

age_gr=gr.Slider(2, 100, value=30, label="Age", info="Choose between 2 and 100")
creatinine_phosphokinase_gr=gr.Slider(20, 8000, value=500, label="Creatinine Phosphokinase", info="Choose between 20 and 8000")
ejection_fraction_gr=gr.Slider(10, 100, value=40, label="Ejection", info="Choose between 10 and 100")
platelets_gr=gr.Slider(2500, 850000, value=9000, label="Platelets", info="Choose between 25000 and 850000")
serum_creatinine_gr=gr.Slider(0, 10, value=1, label="Serum Creatinine", info="Choose between 0 and 10")
serum_sodium_gr=gr.Slider(100, 200, value=4, label="Serum Sodium", info="Choose between 100 and 200")
time_gr=gr.Slider(50, 300, value=4, label="Follow Up Period", info="Choose between 50 and 300")

anaemia_gr=gr.Radio([("yes",1), ("no",0)],label="Anaemia", info="Does patient suffer with Anaemia?")
diabetes_gr=gr.Radio([("yes",1), ("no",0)],label="Diabetes", info="Does patient have Diabetes?")
high_blood_pressure_gr=gr.Radio([("yes",1), ("no",0)],label="High BP", info="Does patient have High BP?")
sex_gr=gr.Radio([("Male",1), ("Female",0)],label="Gender", info="Gender?")
smoking_gr=gr.Radio([("yes",1), ("no",0)],label="Smoking", info="Does patient smoke?")
# Output response
# YOUR CODE HERE
out_label = gr.Textbox(type="text", label='Prediction', elem_id="out_textbox")

title = "Patient Survival Prediction"
description = "Predict survival of patient with heart failure, given their clinical record"

iface = gr.Interface(fn = predict_death_event,
                         inputs = [age_gr,anaemia_gr,creatinine_phosphokinase_gr,diabetes_gr,ejection_fraction_gr,
                                   high_blood_pressure_gr,platelets_gr,serum_creatinine_gr,serum_sodium_gr,
                                   sex_gr,smoking_gr,time_gr],
                         outputs = [out_label],
                         title = title,
                         description = description,
                         allow_flagging='never')


app = gr.mount_gradio_app(app, iface, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 
