import pandas as pd
import numpy as np
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points"""
    R = 6371  # Earth radius in km
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def process_transaction_data(input_file_path):
    """Process transaction data and save as processed CSV"""
    try:
        # Load dataset
        print(f"📂 Loading dataset: {input_file_path}")
        df = pd.read_csv(input_file_path)
        print(f"✅ Loaded {len(df)} transactions")

        # =============================================
        # 🔍 FEATURE ENGINEERING
        # =============================================

        # Drop unnecessary columns safely
        drop_cols = [
            "index", "merchant", "first", "last", "gender", "street", 
            "city", "zip", "city_pop", "trans_num", "unix_time"
        ]
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        df.drop(columns=existing_drop_cols, inplace=True, errors='ignore')

        # Extract hour from transaction time
        if "trans_date_trans_time" in df.columns:
            df["txn_time"] = pd.to_datetime(df["trans_date_trans_time"], errors='coerce').dt.hour
            df.drop(columns=["trans_date_trans_time"], inplace=True)

        # Handle state column - one-hot encoding
        if "state" in df.columns:
            print("🔧 Processing state categories...")
            df["state"] = df["state"].astype(str).str.strip()
            state_dummies = pd.get_dummies(df["state"], prefix="state")
            # Convert boolean to int (0/1)
            state_dummies = state_dummies.astype(int)
            df = pd.concat([df, state_dummies], axis=1)
            df.drop(columns=["state"], inplace=True)
            print(f"✅ Added {state_dummies.shape[1]} state features")

        # Handle category column
        if "category" in df.columns:
            print("🔧 Processing transaction categories...")
            df["category"] = df["category"].astype(str).str.replace(",", " -")
            category_dummies = pd.get_dummies(df["category"], prefix="TXNctg")
            # Convert boolean to int (0/1)
            category_dummies = category_dummies.astype(int)
            df = pd.concat([df, category_dummies], axis=1)
            df.drop(columns=["category"], inplace=True)
            print(f"✅ Added {category_dummies.shape[1]} transaction category features")

        # Handle job categories with improved mapping
        jobcat_path = "Uploaded_Datasets/Processed/job_categories.csv"
        if os.path.exists(jobcat_path) and "job" in df.columns:
            try:
                print("🔧 Processing job categories...")
                jobcat_df = pd.read_csv(jobcat_path)
                
                # Clean job names in both datasets
                df["job"] = df["job"].astype(str).str.strip().str.replace(",", " -")
                
                # Create job to category mapping from the wide format CSV
                job_to_category = {}
                
                # Iterate through each category column
                for category_col in jobcat_df.columns:
                    # Get all jobs in this category (drop NaN values)
                    jobs_in_category = jobcat_df[category_col].dropna()
                    
                    # Clean job names and create mapping
                    for job in jobs_in_category:
                        clean_job = str(job).strip().replace(",", " -")
                        job_to_category[clean_job] = category_col
                
                print(f"📊 Created mapping for {len(job_to_category)} jobs across {len(jobcat_df.columns)} categories")
                
                # Map jobs to categories
                df["job_category"] = df["job"].map(job_to_category)
                
                # Check for unmapped jobs
                unmapped_jobs = df[df["job_category"].isna()]["job"].unique()
                if len(unmapped_jobs) > 0:
                    print(f"⚠️ {len(unmapped_jobs)} jobs could not be mapped to categories: {unmapped_jobs[:5]}...")
                    # Fill unmapped jobs with 'Unknown' category
                    df["job_category"] = df["job_category"].fillna("Unknown")
                
                # Create dummy variables for job categories
                jobcat_dummies = pd.get_dummies(df["job_category"], prefix="JOBctg")
                # Convert boolean to int (0/1)
                jobcat_dummies = jobcat_dummies.astype(int)
                df = pd.concat([df, jobcat_dummies], axis=1)
                
                # Drop original job and job_category columns
                df.drop(columns=["job_category", "job"], inplace=True, errors='ignore')
                
                print(f"✅ Added {jobcat_dummies.shape[1]} job category features")
                
            except Exception as e:
                print(f"⚠️ Error processing job categories: {e}")
                import traceback
                traceback.print_exc()
                # If job category processing fails, drop the job column
                if "job" in df.columns:
                    df.drop(columns=["job"], inplace=True)

        # Handle date of birth
        if "dob" in df.columns:
            print("🔧 Processing date of birth...")
            df["dob_year"] = pd.to_datetime(df["dob"], errors='coerce').dt.year
            
            # Create decade flags
            decades = list(range(1920, 2010, 10))
            for start_year in decades:
                col_name = f"dob_{str(start_year)[2:]}s"
                df[col_name] = np.where(
                    (df["dob_year"] >= start_year) & (df["dob_year"] < start_year + 10), 1, 0
                )
            
            df.drop(columns=["dob_year", "dob"], inplace=True, errors='ignore')
            print("✅ Added decade features from date of birth")

        # Calculate distance between customer & merchant
        if all(col in df.columns for col in ["lat", "long", "merch_lat", "merch_long"]):
            print("🔧 Calculating merchant distances...")
            df["distance"] = haversine_distance(
                df["lat"], df["long"], df["merch_lat"], df["merch_long"]
            )
            print("✅ Added merchant distance feature")

        # User behavioral metrics
        if "cc_num" in df.columns:
            print("🔧 Calculating user behavioral metrics...")
            user_metrics = df.groupby("cc_num").agg(
                avg_txn_amt=("amt", "mean"),
                stddev_txn_amt=("amt", "std"),
                avg_txn_time=("txn_time", "mean"),
                avg_merchant_distance=("distance", "mean") if "distance" in df.columns else ("amt", "mean")
            ).round(2).fillna(0).reset_index()

            df = df.merge(user_metrics, on="cc_num", how="left")
            print("✅ Added user behavioral metrics")

        # Drop temporary columns
        final_drop_cols = ["cc_num", "lat", "long", "merch_lat", "merch_long"]
        existing_final_drop = [col for col in final_drop_cols if col in df.columns]
        df.drop(columns=existing_final_drop, inplace=True, errors='ignore')

        # Convert any remaining boolean columns to int (0/1)
        bool_columns = df.select_dtypes(include=['bool']).columns
        if len(bool_columns) > 0:
            print(f"🔧 Converting {len(bool_columns)} boolean columns to integers...")
            df[bool_columns] = df[bool_columns].astype(int)

        # Fill any remaining NaN values
        df = df.fillna(0)

        # Reorder columns if is_fraud exists
        if "is_fraud" in df.columns:
            other_cols = [c for c in df.columns if c != "is_fraud"]
            df = df[["is_fraud"] + other_cols]

        # Save processed file
        output_dir = os.path.join("Uploaded_Datasets", "Processed")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file_path = os.path.join(output_dir, f"Processed_{os.path.basename(input_file_path)}")
        df.to_csv(output_file_path, index=False)
        
        print(f"✅ Successfully processed and saved: {output_file_path}")
        print(f"📊 Final shape: {df.shape}")
        print(f"📋 Columns: {list(df.columns)}")
        
        return output_file_path

    except Exception as e:
        print(f"❌ Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Standalone testing
if __name__ == "__main__":
    # Test with sample data
    test_file = os.path.join("Uploaded_Datasets", "Raw", "financeRecords.csv")
    if os.path.exists(test_file):
        result = process_transaction_data(test_file)
        if result:
            print("✅ Data processing test completed successfully!")
    else:
        print("ℹ️ No test file found. Please run setup_environment.py first.")