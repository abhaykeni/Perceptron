from utils.all_utils import prepare_data, save_plot
from utils.model import Perceptron
import pandas as pd
import logging
import os

log_dir = "logs"
gate = "AND Gate"
os.makedirs(log_dir,exist_ok=True)

logging.basicConfig(
    filename=os.path.join("logs","running_logs.log"),
    level=logging.INFO,
    format = '[%(asctime)s: %(levelname)s: %(module)s]: %(message)s',
    filemode ='a'
    )




def main(data, modelName, plotName, eta, epochs):
    df_AND = pd.DataFrame(data)
    logging.info(f"This is the raw dataset:\n {df_AND}")
    X, y = prepare_data(df_AND)

    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    model.save(filename=modelName, model_dir="model")
    save_plot(df_AND, model, filename=plotName)


if __name__ == "__main__":
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y" : [0,0,0,1]
    }

    ETA = 0.02
    EPOCHS = 10
    
    try:
        logging.info(f">>>>>>Starting Training For {gate}<<<<<<<<")
        main(data=AND, modelName="and.model", plotName="and.png", eta=ETA, epochs=EPOCHS)
        logging.info(f">>>>>>>>Done Training for {gate}<<<<<<<<<<\n\n") 
    except Exception as e:
        logging.exception(e)
        raise e


    