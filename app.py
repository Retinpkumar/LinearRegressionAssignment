from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import pandas as pd
from Data_Preprocessing.data_processing import DataProcessor
from application_logger.app_logger import Logger

app = Flask(__name__, template_folder='templates')

logfile_path = 'LogFiles/prediction_log.txt'
logger_object = Logger()


@app.route('/')
@cross_origin()
def home_page():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
@cross_origin()
def result_page():
    logfile = open(logfile_path, mode='a')
    logger_object.log(logfile, "Preparing for obtaining user input.'")
    if request.method == 'POST':
        try:
            crim = float(request.form['crim'])
            zn = float(request.form['zn'])
            indus = float(request.form['indus'])
            chas = float(request.form['chas'])
            nox = float(request.form['nox'])
            rm = float(request.form['rm'])
            age = float(request.form['age'])
            dis = float(request.form['dis'])
            rad = float(request.form['rad'])
            tax = float(request.form['tax'])
            ptratio = float(request.form['ptratio'])
            b = float(request.form['b'])
            lstat = float(request.form['lstat'])
            
            df = pd.DataFrame({'CRIM': crim,
            'ZN': zn,
            'INDUS': indus,
            'CHAS': chas,
            'NOX': nox,
            'RM': rm,
            'AGE': age,
            'DIS': dis,
            'RAD': rad,
            'TAX': tax,
            'PTRATIO': ptratio,
            'B': b,
            'LSTAT': lstat}, index=[1])

            logger_object.log(logfile, "User input obtained successfully.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while obtaining user input. Exception :" + str(e))
            logger_object.log(logfile, "Failed to obtain user input.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            # Perform cleaning
            processor = DataProcessor(df)
            df_test = processor.cleanData()
            logger_object.log(logfile, "Successfully processed the user input.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while processing user input. Exception :" + str(e))
            logger_object.log(logfile, "Failed to process user input.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            # Standardizing the data
            scaler_file = 'Model/new_standard_scaler.pickle'
            scaled_model = pickle.load(open(scaler_file, 'rb'))
            logger_object.log(logfile, "Successfully loaded the scaler file.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while loading the scaler file. Exception :" + str(e))
            logger_object.log(logfile, "Failed to load the scaler file.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            df_test_scaled = scaled_model.transform(df_test)
            logger_object.log(logfile, "Successfully standardized input data.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while standardizing. Exception :" + str(e))
            logger_object.log(logfile, "Failed to standardize the input data.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            model_file = 'Model/final_model.pickle'
            loaded_model = pickle.load(open(model_file, 'rb'))
            logger_object.log(logfile, "Successfully loaded the model for prediction.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while loading the model. Exception :" + str(e))
            logger_object.log(logfile, "Failed to load the model for prediction.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            prediction = loaded_model.predict(df_test_scaled)
            print("Prediction is :", prediction)
            return render_template("after.html", prediction=prediction[0].round(2))
            logger_object.log(logfile, "Successfully predicted the output.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while predicting the output. Exception :" + str(e))
            logger_object.log(logfile, "Failed to predict the output.")
            logfile.close()
            raise Exception()
    else:
        return render_template("home.html")


if __name__ == "__main__":
    app.run()
