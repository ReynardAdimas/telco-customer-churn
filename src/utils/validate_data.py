import great_expectations as ge
from great_expectations.core.expectation_suite import ExpectationSuite
import pandas as pd
from typing import Tuple, List

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    print("Starting data validation ")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Create ephemeral context
    context = ge.get_context(mode="ephemeral")

    # Add pandas datasource
    datasource = context.data_sources.add_pandas(
        name="pandas_ds"
    )

    # Add dataframe asset (NO dataframe passed here)
    data_asset = datasource.add_dataframe_asset(
        name="telco_asset"
    )

    data_asset.add_batch_definition(
        name="telco_batch_def"
    )

    # Create batch definition (NAME IS REQUIRED)
    batch_definition = data_asset.get_batch_definition(
        name="telco_batch_def"
    )

    # Inject dataframe at runtime
    batch = batch_definition.get_batch(
        batch_parameters={"dataframe": df}
    )

    # Create / get expectation suite
    suite_name = "telco_expectations"

    suite = ExpectationSuite(name=suite_name)
    context.suites.add(suite)

    # Get validator
    validator = context.get_validator(
        batch=batch,
        expectation_suite_name=suite_name
    )
    # Schema Validation
    validator.expect_column_to_exist("customerID")
    validator.expect_column_values_to_not_be_null("customerID")

    required_columns = [
        "gender", "Partner", "Dependents", "PhoneService",
        "InternetService", "Contract", "tenure",
        "MonthlyCharges", "TotalCharges"
    ]

    for col in required_columns:
        validator.expect_column_to_exist(col)

    # Business Rules
    validator.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    validator.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    validator.expect_column_values_to_be_in_set(
        "Contract", ["Month-to-month", "One year", "Two year"]
    )
    validator.expect_column_values_to_be_in_set(
        "InternetService", ["DSL", "Fiber optic", "No"]
    )
    # Numeric Rules
    validator.expect_column_values_to_be_between("tenure", 0, 120)
    validator.expect_column_values_to_be_between("MonthlyCharges", 0, 200)

    validator.expect_column_pair_values_A_to_be_greater_than_B(
        "TotalCharges",
        "MonthlyCharges",
        or_equal=True,
        mostly=0.95
    )
    # Run Validation
    result = validator.validate()

    failed = [
        r["expectation_config"]["expectation_type"]
        for r in result["results"]
        if not r["success"]
    ]

    if result["success"]:
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")
        print("Failed expectations:", list(set(failed)))

    return result["success"], failed
