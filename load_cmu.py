from datasets import load_dataset
import pandas as pd

DATASET_NAME = "textminr/cmu-book-summaries"

def load_cmu_dataset(split: str = "train") -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, split=split)
    df = ds.to_pandas()
    
    if "summary" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"summary" : "text"})
        
    missing = [c for c in ("title", "text") if c not in df.columns]
    
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found. Got: {list(df.columns)}"
        )
    return df[["title", "text"]].reset_index(drop=True)
        
        
if __name__ == "__main__":
    #just a quick check
    df = load_cmu_dataset()
    print(f"Loaded {len(df):,} rows from {DATASET_NAME}")
    print (f"Columns: {list(df.columns)}")
    print("\nFirst row:")
    print(f"  title: {df.iloc[0]['title']}")
    print(f"  text : {df.iloc[0]['text'][:500]}...")