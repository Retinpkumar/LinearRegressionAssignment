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
            age = float(request.form['age'])
            dis = float(request.form['dis'])
            rad = float(request.form['rad'])
            tax = float(request.form['tax'])
            b = float(request.form['b'])
            lstat = float(request.form['lstat'])
            indus = float(request.form['indus'])
            rm = float(request.form['rm'])
            logger_object.log(logfile, "User input obtained successfully.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while obtaining user input. Exception :" + str(e))
            logger_object.log(logfile, "Failed to obtain user input.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            df_pred = pd.DataFrame({"CRIM": crim,
                                    "ZN": zn,
                                    "AGE": age,
                                    "DIS": dis,
                                    "RAD": rad,
                                    "TAX": tax,
                                    "B": b,
                                    "LSTAT": lstat}, index=[1])

            # Input for RAD
            if rad < 9:
                df_pred['RAD'] = 1
            else:
                df_pred['RAD'] = 0

            # Input for B
            if b < 380:
                df_pred['B'] = 0
            else:
                df_pred['B'] = 1

            # Input for INDUS
            if indus == 1:
                df_pred['INDUS_1'] = 1
            elif indus == 2:
                df_pred['INDUS_2'] = 1
            elif indus == 3:
                df_pred['INDUS_3'] = 1
            elif indus == 4:
                df_pred['INDUS_4'] = 1
            elif indus == 5:
                df_pred['INDUS_5'] = 1
            elif indus == 6:
                df_pred['INDUS_6'] = 1
            elif indus == 7:
                df_pred['INDUS_7'] = 1
            elif indus == 8:
                df_pred['INDUS_8'] = 1
            elif indus == 9:
                df_pred['INDUS_9'] = 1
            elif indus == 10:
                df_pred['INDUS_10'] = 1
            elif indus == 11:
                df_pred['INDUS_11'] = 1
            elif indus == 12:
                df_pred['INDUS_12'] = 1
            elif indus == 13:
                df_pred['INDUS_13'] = 1
            elif indus == 15:
                df_pred['INDUS_15'] = 1
            elif indus == 18:
                df_pred['INDUS_18'] = 1
            elif indus == 19:
                df_pred['INDUS_19'] = 1
            elif indus == 21:
                df_pred['INDUS_21'] = 1
            elif indus == 25:
                df_pred['INDUS_25'] = 1
            elif indus == 27:
                df_pred['INDUS_27'] = 1
            else:
                df_pred['INDUS_1'] = 0
                df_pred['INDUS_2'] = 0
                df_pred['INDUS_3'] = 0
                df_pred['INDUS_4'] = 0
                df_pred['INDUS_5'] = 0
                df_pred['INDUS_6'] = 0
                df_pred['INDUS_7'] = 0
                df_pred['INDUS_8'] = 0
                df_pred['INDUS_9'] = 0
                df_pred['INDUS_10'] = 0
                df_pred['INDUS_11'] = 0
                df_pred['INDUS_12'] = 0
                df_pred['INDUS_13'] = 0
                df_pred['INDUS_15'] = 0
                df_pred['INDUS_18'] = 0
                df_pred['INDUS_19'] = 0
                df_pred['INDUS_21'] = 0
                df_pred['INDUS_25'] = 0
                df_pred['INDUS_27'] = 0

            # Input for RM
            if rm == 4:
                df_pred['RM_4'] = 1
            elif rm == 5:
                df_pred['RM_5'] = 1
            elif rm == 6:
                df_pred['RM_6'] = 1
            elif rm == 7:
                df_pred['RM_7'] = 1
            elif rm == 8:
                df_pred['RM_8'] = 1
            else:
                df_pred['RM_4'] = 0
                df_pred['RM_5'] = 0
                df_pred['RM_6'] = 0
                df_pred['RM_7'] = 0
                df_pred['RM_8'] = 0

            df_pred.to_csv('UserInput/test.csv')
            logger_object.log(logfile, "Successfully converted User input to 'test.csv'.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured during creation of 'test.csv'. Exception :" + str(e))
            logger_object.log(logfile, "Failed to create 'test.csv'")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            # Perform cleaning
            df_test = DataProcessor().cleanData()
            logger_object.log(logfile, "Successfully cleaned user input.")
            logfile.close()
        except Exception as e:
            logger_object.log(logfile, "Exception occured while cleaning user input. Exception :" + str(e))
            logger_object.log(logfile, "Failed to clean user input.")
            logfile.close()
            raise Exception()

        try:
            logfile = open(logfile_path, mode='a')
            # Standardizing the data
            scaler_file = 'Model/standard_scaler.pickle'
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
            model_file = 'Model/linear_regression_model.pickle'
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
