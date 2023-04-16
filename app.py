
from flask import Flask,request,render_template
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline


app = Flask(__name__)


@app.route("/")
def home():
    return render_template('home.html') 


@app.route("/predict_data",methods=['GET','POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
         data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score')))
         pred_df = data.get_data_as_dataframe()
         print(pred_df)
         predict_pipeline = PredictPipeline()
         print('Mid')
         results=predict_pipeline.predict(pred_df)
         print("after Prediction")
         return render_template('home.html',results=results[0])


if __name__ == "__main__":
    app.run(debug=True)