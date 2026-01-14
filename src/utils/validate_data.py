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

    ge_df.expect_column_values_to_be_between("tenure", min_value=0) 
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0) 
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0) 
    print("Validating statistical properties") 
    ge_df.excpect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200) 

    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges") 

    print("Validating data consistency") 
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A = "TotalCharges", 
        column_B = "MonthlyCharges", 
        or_equal = True, 
        mostly=0.95
    ) 

    print("Running complete validation suite")
    result = ge_df.validate() 

    failed_expectations = [] 
    for r in result["result"]: 
        if not r["success"]:
            excepectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(excepectation_type) 
    
    total_checks = len(result["result"])
    passed_checks = sum(1 for r in result["result"] if r["success"])
    failed_checks = total_checks - passed_checks 

    if result["success"]:
        print(f"Data validation passed: {passed_checks}/{total_checks} checks successful")
    else: 
        print(f"Data validation failed: {failed_checks}/{total_checks} checks failed")
        print(f"Failed expectations: {failed_expectations}") 
    
    return result["success"], failed_expectations

    



