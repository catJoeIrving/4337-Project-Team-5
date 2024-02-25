# Repo for Team 5 project for COSC 4337

Method 1: Python IDE (PyCharm, VSC, etc.):

1) Open your Python IDE.
2) Download the project.py file from this repository.
3) Download the kidney_disease.csv dataset. Make sure this is in the same directory as your Python file or provide the full path to the file in the pd.read_csv() function.
4) Run the program. You can usually do this by right-clicking in the file and selecting ‘Run’, or using a keyboard shortcut (often F5 or Ctrl+Shift+F10 in many IDEs).
5) The IDE will execute the code and you can see the output in the console.

Method 2: Python Notebook:

1) Open your Python Notebook application.
2) You have two options here:
Option 1: Create a new Python notebook file and copy each section of the provided code into a separate cell in the notebook.
Option 2: Download the project.ipynb file from your GitHub repository and open it in your Python Notebook application.
3) If you’re following Option 1, you can create a new cell by clicking on the ‘+’ button in the toolbar.
4) Similar to the Python IDE, ensure the dataset kidney_disease.csv is in the same directory as your notebook or provide the full path to the file in the pd.read_csv() function.
5) To run the code, select each cell and then click the ‘Run’ button in the toolbar, or press Shift+Enter. The output of each cell will appear below the cell.
Note: Jupyter Notebook automatically saves your work. You can also manually save by clicking on the ‘Save’ button in the toolbar or by pressing Ctrl+S.




**What to Expect:**

Initial Data Cleaning: The code starts by loading the dataset and performing initial data cleaning tasks, such as dropping the 'id' column, converting certain columns to numeric values, handling missing values by filling them with mean or mode, and applying label encoding to categorical columns.

Final Check for Null Values: After the initial data cleaning, the code prints a summary of null values across all columns to confirm that there are no remaining null values in the dataset.

Winsorization: The code then applies Winsorization to the numeric columns to limit extreme values (outliers) by replacing them with the values at the 5th and 95th percentiles. This step is aimed at reducing the influence of outliers in the data.

Feature Selection using RFE: Recursive Feature Elimination (RFE) is used with a Logistic Regression model to select features that are most relevant to predicting the target variable ('classification'). The code prints the number of features selected, the boolean mask of selected features, their ranking, and a sorted list of features with their rankings.

Dataframe of Selected Features: A new dataframe is created containing only the features selected by RFE, along with the target variable. The code prints summary statistics (like count, mean, std, min, max, etc.) for this new dataframe.

Saving the Cleaned Dataset: The fully cleaned and feature-selected dataset is saved to a new CSV file named output.csv.

Visualization: The code includes sections for visualizing the distribution of numeric columns before and after Winsorization using boxplots. It adjusts the x-axis and y-axis limits to provide a clearer view of the data distribution and outlier management.

Summary Statistics Post-Winsorization: Finally, the code prints summary statistics again for the selected features after Winsorization, providing insights into how Winsorization has affected the data.
