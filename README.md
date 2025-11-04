# S3 VECTORS PIPELINE

This pipeline implements a serverless vector database using FAISS for indexing and similarity search, and Lithops for parallel processing. It efficiently manages vector data.

## STEPS TO EXECUTE PIPELINE

### Use of a custom Dataset
If you use your own dataset, you need to **specify all the necessary parameters** in the *"Initializations"* section.  
Make sure your files (**dataset**, **queries**, and **true_neighbors**) follow the correct format (as indicated in the notebook).


### 1 - Upload CSV Files
Upload the CSV files from **`files.zip`** to any **S3 bucket**.

---

### 2 - Run *Initializations*
Execute the **"Initializations"** section to:
- Import the necessary packages  
- Initialize **Serverless Vector DB**

**Important:**  
> Specify the name of the **S3 bucket resource** where the CSV files are located.

---

### 3 - Run *Vectors Indexing*
Execute the **"Vectors Indexing"** section to insert vectors into **Serverless vector DB**.

You can only insert the **entire dataset** (individual vector insertion is not supported). 

---

### 4 - Run *Querying*
Execute the **"Querying"** section to perform queries (in batch) on the indexed dataset.

---

### 5 - Run *Query Recall*
Execute the **"Query Recall"** section to calculate the **precision** of the queries.

---

### 7 - *(Optional)* Clean Environment
Execute **"Clean environment"** to delete both local and S3 resources.

---

**Recommended order:**  
`Initializations → Vectors Indexing → Querying → Query Recall`  
(Optional: `Clean environment`)