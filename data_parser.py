import os, sys, base64, json, log_config, shutil
import pandas as pd
from traceback import format_exc



class Parser():
    def __init__(self):
        self.ch_en = {
            'time': 'time',
            '頓紙用氣': 'Gas_for_paper',
            '每小時生產量': 'Hourly_Production',
            '清水總量': 'Total_water',
            '卷取掃描架定量測量值': 'Quantitative_measurement_of_coiling_scan_frame',
            '乾強使用量': 'dry_strength_use',
            '#1透平機電機側壓力': '#1_Turbine_Motor_Side_Pressure',
            '#1透平機自由側壓力': '#1_Turbine_Free_Side_Pressure',
            '#1透平機負壓': '#1_Turbine_Negative_Pressure',
            '#1透平機負壓.1': '#1_Turbine_Negative_Pressure.1',
            '#1透平機主馬達電流': '#1_Turbine_Main_Motor_Current',
            '#1真空泵電機側振動': '#1_Vacuum_Pump_Motor_Side_Vibration',
            '#1真空泵自由側振動': '#1_Vacuum_Pump_Free_Side_Vibration',
            '#2透平機電機側壓力': '#2_Turbine_Motor_Side_Pressure',
            '#2透平機自由側壓力': '#2_Turbine_Free_Side_Pressure',
            '#2透平機負壓': '#2_Turbine_Negative_Pressure',
            '#2透平機負壓.1': '#2_Turbine_Negative_Pressure.1',
            '#2透平機主馬達電流': '#2_Turbine_Main_Motor_Current',
            '#2真空泵電機側振動': '#2_Vacuum_Pump_Motor_Side_Vibration',
            '#2真空泵自由側振動': '#2_Vacuum_Pump_Free_Side_Vibration',
            '#3透平機電機側壓力': '#3_Turbine_Motor_Side_Pressure',
            '#3透平機自由側壓力': '#3_Turbine_Free_Side_Pressure',
            '#3透平機負壓': '#3_Turbine_Negative_Pressure',
            '#3透平機負壓.1': '#3_Turbine_Negative_Pressure.1',
            '#3透平機主馬達電流': '#3_Turbine_Main_Motor_Current',
            '#3真空泵電機側振動': '#3_Vacuum_Pump_Motor_Side_Vibration',
            '#3真空泵自由側振動': '#3_Vacuum_Pump_Free_Side_Vibration',
            '#4透平機自由側壓力': '#4_Turbine_Free_Side_Pressure',
            '#4透平機負壓': '#4_Turbine_Negative_Pressure',
            '#4透平機負壓.1': '#4_Turbine_Negative_Pressure.1',
            '#4透平機主馬達電流': '#4_Turbine_Main_Motor_Current',
            '#4真空泵電機側振動': '#4_Vacuum_Pump_Motor_Side_Vibration',
            '#4真空泵自由側振動': '#4_Vacuum_Pump_Free_Side_Vibration',
            'flow nukp': 'flow_nukp',
            'headbox feed fan pump': 'headbox_feed_fan_pump',
            }


        self.form_check = {
            "TG0802.csv": ['date', 'time', '頓紙用氣', '每小時生產量', '清水總量', '卷取掃描架定量測量值', '乾強使用量'],
            "TG0806.csv": ['date', 'time', '#1透平機電機側壓力', '#1透平機自由側壓力', '#1透平機負壓', '#1透平機負壓.1', '#1透平機主馬達電流', '#1真空泵電機側振動', '#1真空泵自由側振動'],
            "TG0807.csv": ['date', 'time', '#2透平機電機側壓力', '#2透平機自由側壓力', '#2透平機負壓', '#2透平機負壓.1', '#2透平機主馬達電流', '#2真空泵電機側振動', '#2真空泵自由側振動'],
            "TG0808.csv": ['date', 'time', '#3透平機電機側壓力', '#3透平機自由側壓力', '#3透平機負壓', '#3透平機負壓.1', '#3透平機主馬達電流', '#3真空泵電機側振動', '#3真空泵自由側振動'],
            "TG0809.csv": ['date', 'time', '#4透平機自由側壓力', '#4透平機負壓', '#4透平機負壓.1', '#4透平機主馬達電流', '#4真空泵電機側振動', '#4真空泵自由側振動'],
            "TG0811.csv": ['date', 'time', 'flow nukp', 'headbox feed fan pump'],
        }


    def data_clean(self, zip_name, temp_path, clean_path, init_path, data_info, log_path):
        logging = log_config.set_log(filepath = log_path, level = 2, freq = "D", interval = 30)
        try:
            logging.info('Start parsing.')
            count = 0
            for dirs1 in os.listdir(temp_path): # dir - year
                for dirs2 in os.listdir(os.path.join(temp_path, dirs1)): # dir - month
                    for dirs3 in os.listdir(os.path.join(temp_path, dirs1, dirs2)): # dir - day
                        
                        df2 = pd.DataFrame()
                        for file in os.listdir(os.path.join(temp_path, dirs1, dirs2, dirs3)): #file - csv
                            path = os.path.join(temp_path, dirs1, dirs2, dirs3, file)
                            df1 = pd.read_csv(path, encoding = "big5", header = 5)


                            # clean file
                            try:
                                if file.startswith('TG0802'):
                                    df1 = df1.iloc[7:, [2, 3, 5, 6, 7, 8, 9]]
                                    df1 = df1.rename(columns = {"Unnamed: 9": "乾強使用量"})
                                elif file.startswith('TG0806'):
                                    df1 = df1.iloc[7:, [2, 3, 5, 6, 7, 8, 9, 10, 11]]
                                elif file.startswith('TG0807') or file.startswith('TG0808'):
                                    df1 = df1.iloc[7:, [2, 3, 5, 6, 7, 8, 9, 11, 12]]
                                elif file.startswith('TG0809'):
                                    df1 = df1.iloc[7:, [2, 3, 5, 7, 8, 9, 11, 12]]
                                elif file.startswith('TG0811'):
                                    df1 = df1.iloc[7:, [2, 3, -2, -1]]
                                else:
                                    continue
                            except:
                                logging.error(f"Parser stopped. The format of {os.path.join(dirs1, dirs2, dirs3, file)} is wrong.")

                                result = {
                                    "status": "fail",
                                    "reason": f"The format of {os.path.join(dirs1, dirs2, dirs3, file)} is wrong.",
                                    "file": zip_name,
                                    }

                                with open(os.path.join(clean_path, "parser_result.json"), 'w', newline='') as file:
                                    json.dump(result, file)

                                return
                            

                            # check if file is empty
                            if df1.empty:
                                logging.error(f"Parser stopped. {os.path.join(dirs1, dirs2, dirs3, file)} is an empty file.")

                                result = {
                                    "status": "fail",
                                    "reason": f"{os.path.join(dirs1, dirs2, dirs3, file)} is an empty file.",
                                    "file": zip_name,
                                    }

                                with open(os.path.join(clean_path, "parser_result.json"), 'w', newline='') as file:
                                    json.dump(result, file)
                                
                                return


                            # check columns name
                            df1 = df1.rename(columns = {"Unnamed: 2": "date", "Unnamed: 3": "time"})
                            try:
                                df1 = df1[self.form_check[file]]
                            except:
                                logging.error(f"Parser stopped. The format of {os.path.join(dirs1, dirs2, dirs3, file)} is wrong.")

                                result = {
                                    "status": "fail",
                                    "reason": f"The format of {os.path.join(dirs1, dirs2, dirs3, file)} is wrong.",
                                    "file": zip_name,
                                    }

                                with open(os.path.join(clean_path, "parser_result.json"), 'w', newline='') as file:
                                    json.dump(result, file)

                                return


                            # save each csv file in clean dir
                            clean_dir = os.path.join(clean_path, dirs1, dirs2, dirs3)
                            if not os.path.isdir(clean_dir): # create dir if dir doesn't exist
                                os.makedirs(clean_dir)
                            df1.to_csv(os.path.join(clean_dir, file), encoding = "big5", index = False)


                            # combine data by day
                            if df2.empty:
                                df2 = df1
                            else:
                                df2 = pd.merge(df2, df1, on = ["date", "time"])  


                        # save daily data in init dir
                        init_dir = os.path.join(init_path, dirs1, dirs2)
                        if not os.path.isdir(init_dir): # create dir if dir doesn't exist
                                os.makedirs(init_dir)
                                
                        df2["time"] = pd.to_datetime(df2["date"] + " " + df2["time"])
                        df2 = df2.drop("date", axis = 1)
                        df2 = df2.drop_duplicates()
                        df2 = df2.sort_values("time").reset_index(drop = True)
                        
                        feat = [self.ch_en[col] for i, col in enumerate(df2.columns)]
                        df2.columns = feat
                        df2[df2.columns[1:]] = df2[df2.columns[1:]].astype(float)
                        df2.to_csv(os.path.join(init_dir, f"{dirs1}{dirs2}{dirs3}.csv"), index = False)                        


                        # save daily info in data.json
                        with open(data_info, newline='') as file:
                            data = json.load(file)
                            
                        if f"{dirs1}-{dirs2}" not in data:
                            data[f"{dirs1}-{dirs2}"] = {}

                        data[f"{dirs1}-{dirs2}"].update({
                            str(dirs3): {
                                "count": len(df2),
                                "Hourly_Production_Avg": df2["Hourly_Production"].mean(),
                                }
                            })

                        with open(data_info, 'w', newline='') as file:
                            json.dump(data, file)


                        count += len(df2)


            logging.info('Clear temp dir.')
            shutil.rmtree(temp_path)
            os.mkdir(temp_path)


            logging.info('Finish parsing.')

            result = {
                "status":"success",
                "reason":"",
                "file": zip_name,
                "count": count,
                }

            with open(os.path.join(clean_path, "parser_result.json"), 'w', newline='') as file:
                json.dump(result, file)
        
        
        except:
            logging.error(format_exc())
            result = {
                "status": "fail",
                "reason": format_exc(),
                "file": zip_name,
                }

            with open(os.path.join(clean_path, "parser_result.json"), 'w', newline='') as file:
                json.dump(result, file)
    


if __name__ == '__main__':
    if len(sys.argv) > 1: 
        input_ = sys.argv[1]
        input_ = base64.b64decode(input_).decode('utf-8')

        input_ = json.loads(input_)
    else:
        print("Input parameter error.")
    
    
    zip_name = input_["ZIP_NAME"]
    temp_path = input_["TEMP_PATH"]
    clean_path = input_["CLEAN_PATH"]
    init_path = input_["INIT_PATH"]
    data_info = input_["DATA_INFO"]
    log_path = input_["LOG_PATH"]
    

    parser = Parser()
    parser.data_clean(zip_name, temp_path, clean_path, init_path, data_info, log_path)