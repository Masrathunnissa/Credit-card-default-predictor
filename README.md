Credit Card Default Prediction

A Flask-based machine learning web application for predicting credit card default risk using Python and a trained machine learning model.

Project Setup

Follow the steps below to set up and run the project on a new machine.

1. Clone the Repository

Clone the GitHub repository:

git clone <repository-url>

Navigate to the project directory:

cd <project-folder>

2. Create a Virtual Environment

Make sure Python 3.9 is installed.

Create a virtual environment:

py -3.9 -m venv venv

This creates a "venv" folder inside the project directory.

3. Activate the Virtual Environment

For Windows PowerShell:

.\venv\Scripts\Activate.ps1

For Windows Command Prompt:

venv\Scripts\activate

After successful activation, the terminal should display:

(venv)

before the current directory.

4. Upgrade pip

Run:

python -m pip install --upgrade pip

5. Install Project Dependencies

The required Python packages are listed in "requirements.txt".

Install them using:

pip install -r requirements.txt

To verify the installed packages:

pip list

6. Configure Python Interpreter in VS Code

Open the project in Visual Studio Code.

Press:

Ctrl + Shift + P

Search for:

Python: Select Interpreter

Select the Python interpreter from the virtual environment:

venv\Scripts\python.exe

Make sure VS Code is using the newly created virtual environment.

7. Run the Application

Activate the virtual environment if it is not already activated:

.\venv\Scripts\Activate.ps1

Run the application using the project's main Python file:

python app/main.py

If your project's entry-point file is different, replace "app/main.py" with the appropriate file.

8. Open the Application

After starting the Flask application, the terminal will display the local server address.

For example:

http://127.0.0.1:5000/

Open the displayed URL in a web browser.

Project Structure

A typical project structure is:

project/
│
├── app/
│   ├── main.py
│   └── ...
│
├── service/
│   └── ...
│
├── templates/
│   └── ...
│
├── requirements.txt
├── .gitignore
└── README.md

Updating Dependencies

If new Python packages are installed during development, update "requirements.txt" using:

pip freeze > requirements.txt

Then commit the updated file:

git add requirements.txt
git commit -m "Update project dependencies"
git push

Git Workflow

For future changes:

git status
git add .
git commit -m "Describe your changes"
git push or git push --set-upstream origin feature/modelfearureeng

Important Notes

- Do not commit the "venv/" folder.
- Do not commit "__pycache__/" folders or ".pyc" files.
- Do not commit ".env" files containing passwords, API keys, or other sensitive information.
- Make sure required dependencies are maintained in "requirements.txt".
- Use the Python version specified for the project when creating the virtual environment.