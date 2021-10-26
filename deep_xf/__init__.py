# To get modules
from . import main
from . import dpp 
from . import forecast_ml 
from . import forecast_ml_extension
from . import stats
from . import utility 
from . import denoise
from . import similarity

# To get sub-modules
from .main import *
from .dpp import *
from .forecast_ml import *
from .forecast_ml_extension import *
from .stats import *
from .utility import *
from .denoise import *
from .similarity import *

#################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.1'
print(f"""{module_type} DeepXF version:{version_number}. Example call by using:

***********************   SET MODEL/BASE CONFIGURATIONS   ************************

select_model, select_user_path, select_scaler, forecast_window = Forecast.set_model_config(select_model='rnn', select_user_path='./forecast_folder_path/', select_scaler='minmax', forecast_window=1)

----------------------------------------------------------------------------------

model_df, orig_df = Helper.get_variable(df, ts, fc)

----------------------------------------------------------------------------------

hidden_dim, layer_dim, batch_size, dropout, n_epochs, learning_rate, weight_decay = Forecast.hyperparameter_config(hidden_dim=64,layer_dim = 3, batch_size=64, dropout = 0.2, n_epochs = 30, learning_rate = 1e-3, weight_decay = 1e-6)

**********************   DEEP LEARNING BASED FORECASTING   ***********************

opt, scaler = Forecast.train(df=df_full_features, target_col='value', split_ratio=0.2, select_model=select_model, select_scaler=select_scaler, forecast_window=forecast_window, batch_size=batch_size, hidden_dim=hidden_dim, layer_dim=layer_dim,dropout=dropout, n_epochs=n_epochs, learning_rate=learning_rate, weight_decay=weight_decay)

----------------------------------------------------------------------------------

forecasted_data, ff_full_features, ff_full_features_ = Forecast.forecast(model_df, ts, fc, opt, scaler, period=25, fq='h', select_scaler=select_scaler,)

---------------------------------------------------------------------------------

Helper.explainable_forecast(df_full_features, ff_full_features_, fc, specific_prediction_sample_to_explain=145370, input_label_index_value=0, num_labels=1)


******************   DYNAMIC FACTOR MODEL BASED NOWCASTING   *********************

select_model, select_user_path, select_scaler, forecast_window = Forecast.set_model_config(select_model='em', select_user_path='./forecast_folder_path/', select_scaler='minmax', forecast_window=5)

----------------------------------------------------------------------------------

nowcast_full_data, nowcast_pred_data = EMModel.nowcast(df_full_features, ts, fc, period=5, fq='h', forecast_window=forecast_window, select_model=select_model)

----------------------------------------------------------------------------------

EMModel.explainable_nowcast(df_full_features, nowcast_pred_data, fc, specific_prediction_sample_to_explain=145370, input_label_index_value=0, num_labels=1)

""")



