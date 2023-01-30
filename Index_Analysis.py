##################################################################################################################################
# Libraries
##################################################################################################################################

from Strategy_Night import *
from Data_Import import *
from Performance_Metrics import *

import warnings
warnings.filterwarnings("ignore")

##################################################################################################################################
# Import Database
##################################################################################################################################
INDEX = pd.read_csv (r'/home/pisati/Desktop/INDEXES/$DAX-XET_5M.csv', index_col=False,  sep=',')

# Set the Date Range for the analysis
Date_Start = "01/03/2010"
Date_End = "11/06/2021"

# Clean Index
Cleaned_Index = Index_Cleaning(INDEX, Date_Start, Date_End)
#Cleaned_Index_D = Index_Cleaning_D(INDEX_D, Date_Start, Date_End)

# Compute intraday, overnaight and daily returns + strategy performance
overnight_threshold = 0.002     # overnight return threshold
overnight_ret_index(Cleaned_Index, overnight_threshold)

# Print summary results on the Console
index_performance(Cleaned_Index)

