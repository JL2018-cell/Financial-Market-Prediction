# Description of folders in this directory:

## bin: stores codes. Executing main.py can launch the program.
<p> Description of programs in this directory:
<ul> <li> main.py: Main program. Execute this to start financial forecating algorithm. </li>
     <li> neural_ntwk_tuning.py: Find optimal hyperparameters of Long Short-term Memory by grid-search. </li>
     <li> ARIMA.py: Forecast using Autoregressive Integrated Moving Average model. </li>
     <li> chromedriver.exe: a Chrome browser for web-scrapping. Called by web_scraper.py. </li>
     <li> Detect_Jumps.py: Call MATLAB function to calculate number of jumps observed in a vector. Called by JumpModel.py. </li>
     <li> EntityRecognition.py: Given many pieces of text and name of target entity/company, assign weight (a positive number) to each piece according to frequency of entity appears in the text. Called by further_NLP.py. </li>
     <li> Find_weight.py: Calculate optimal decaying factor of sentiment score over time such that it maximize correlation between changes in sentiment score over time and residuals of ARIMA model. Called by main.py </li>
     <li> Forecast_r_function.r: Called by ARIMA.py. </li>
     <li> further_NLP.py: Further classify sentiment score of each piece of news. Called by main.py. </li>
     <li> Jump_Model.py: Forecast using Jump Model. Called by main.py. </li>
     <li> Jump_Model_run.m: Called by Jump_Model.py. </li>
     <li> Jump_Statistics.m: Called by Detect_Jumps.py. This program caclulates parameters of jump (rates of Poisson Process and Exponential distribution) from a historical time series. Called by Detect_Jumps.py. </li>
     <li> LSTM.py: Forecast using Long Short-term Memory Model. </li>
     <li>obtain_data.py: Call quantmod_r.r to download data. Called by sql_operation.py</li>
     <li>preliminary_NLP.py: Classify different pieces of text into positive mood and negative mood. Positive mood has sentiment score = +1 and negative mood has sentiment score = -1. Called by main.py.</li>
     <li>quantmod_r.r: Called by obtain_data.py.</li>
     <li>read_JmpMdl_data.py: Called by Jump_Model.py to read the definition of parameters in Jump Model. Called by Jump_Model.py</li>
     <li>sampling_scheme.py: Remove some data from time series. This is the idea of Random Sample Consensus.</li>
     <li>show_msg.py: Display a beautiful message box.</li>
     <li>sql_operation.py: Return data from SQL database.</li>
     <li>web_scraper.py: Obtain news form e-database Factiva (Accessed by HKU library e-database website) and store news in SQL database. Called by main.py</li>
     <li>weight_optmz.r: Called by Find_weight.py.</li>
</ul>
</p>

## conf: Stores setting of program e.g. train-to-test data ratio, location of saving models, etc.
## data: Run data.sql to restore mySQL database. Username: user, password: password. 
## result: stores prediction result after the end of financial data forecaster program.
