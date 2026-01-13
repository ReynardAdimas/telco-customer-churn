import great_expectations as ge 
from typing import Tuple, List 

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    print("Starting data validation with Great Expectations...") 
    # Convert pandas into GE Dataset
    ge_df = ge.dataset.PandasDataset(df) 

    # Schema Validation - Essential Columns 
    print("Validating schem and required columns") 

    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_to_be_null("customerID")

    ge_df.expect_column_to_exist("gender")
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents") 

    ge_df.expect_column_to_exist("PhoneService") 
    ge_df.expect_column_to_exist("InternetService") 
    ge_df.expect_column_to_exist("Contract") 

    ge_df.expect_column_to_exist("tenure") 
    ge_df.expect_column_to_exist("MonthlyCharges") 
    ge_df.expect_column_to_exist("TotalCharges") 

    print("Validating business logic contraints..")

    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"]) 

    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )

    ge_df.expect_column_values_to_be_in_set(
        "InternetService", 
        ["DSL", "Fiber optic", "No"]
    ) 

    print("Validating numeric ranges and business contraints") 

    



