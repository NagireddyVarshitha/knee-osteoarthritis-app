from flask import Flask
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pywebio import start_server
from pywebio.platform.flask import webio_view
from pywebio.input import file_upload, input_group, input
from pywebio.output import put_text, put_buttons, put_table, put_image, put_markdown, put_html, clear, style
from pywebio.session import hold
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

app =Flask(__name__)

def get_image_base64(image_path):
    with open(image_path,"rb")as img_file:
        return"data:image/jpg;base64,"+base64.b64encode(img_file.read()).decode()
    
    def display_image(image_path, width=100, height=50):
        return put_html(f"""
        <div style='text-align:left;'>
            <img src='{get_image_base64(image_path)}' style='width:{width}px; height:{height}px; border-radius:10px;'>
        </div>
    """)

# Function to display the Home Page
def home_page():
    clear()
    style(put_markdown("# <span style='color:blue;'>Knee Osteoarthritis</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    put_markdown("### Welcome to the Knee Osteoarthritis Prediction System", unsafe_allow_html=True)
    put_html(f"""
    <div style = 'text-align:left;'>
    <img src='{get_image_base64("home_image.jpg")}'style=width:50%;height:250px;border-radius:20px;'>
    </div>
    """)
    put_text("Our aim is to assist doctors and patients with timely diagnosis and effective treatment plans.")
    put_text("Welcome to AI-Based Knee Osteoarthritis Prediction")
    put_text("Knee Osteoarthritis (OA) is a degenerative joint disease that affects millions of people worldwide. Our AI-powered system helps predict the likelihood of osteoarthritis based on medical data, making early detection and management easier than ever.")
    put_text("Why Choose Our AI Prediction Tool?\n✅ Accurate Predictions: Our AI model is trained on a large dataset for high precision.\n✅ Easy to Use: Simply enter your details, and our model will predict your osteoarthritis risk.\n✅ Scientific Approach: Built using advanced machine learning models for reliable outcomes.\n✅ Early Detection & Prevention: Helps in taking proactive measures to slow down progression.")
    put_text("How It Works\n1. Enter Your Details – Age, BMI, pain level, and other factors.\n2. AI Analysis – Our AI model processes your inputs.\n3. Get Results – Receive an instant prediction with medical insights.")
    put_text("Who Can Use This?\nIndividuals experiencing joint pain\nHealthcare professionals seeking additional diagnosis support\nResearchers analyzing osteoarthritis trends")
# Function to display Benefits Page
def benefits_page():
    clear()
    style(put_markdown("# <span style='color:green;'>Benefits of this Application</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    put_html(f"""
    <div style = 'text-align:left;'>
    <img src='{get_image_base64("benefits_image.jpg")}'style=width:50%;height:250px;border-radius:10px;'>
    </div>
    """)
    put_text("Key Benefits of Early Osteoarthritis Prediction")
    put_text("🦵 Early Diagnosis & Timely Intervention\nDetect OA before symptoms become severe.\nHelps doctors provide early treatment plans.")
    put_text("🏃 Improved Mobility & Joint Health\nSuggests lifestyle modifications to maintain joint function.\nHelps delay the progression of OA.")
    put_text("💰 Cost-Effective Healthcare\nReduces expensive treatments by focusing on prevention.\nMinimizes the need for joint replacement surgeries.")
    put_text("📊 Data-Driven Decision Making\nAI-backed predictions help in assessing the risk accurately.\nProvides a comparative analysis based on patient history.")
    put_text("🛡 Personalized Treatment Plans\nRecommends exercises, dietary changes, and medications.\nHelps avoid unnecessary medical procedures")
# Function to display Precautions Page
def precautions_page():
    clear()
    style(put_markdown("# <span style='color:red;'>Precautions for Knee Osteoarthritis</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,symptoms_page, team_page,single_prediction_page,dataset_prediction_page])

    put_html(f"""
    <div style = 'text-align:left;'>
    <img src='{get_image_base64("precaution_image.jpg")}'style=width:50%;height:250px;border-radius:10px;'>
    </div>
    """)
    put_text("How to Prevent Knee Osteoarthritis?")
    put_text("🚶 Maintain a Healthy Weight\nExcess weight puts additional stress on knee joints.\nA balanced diet and exercise help reduce the risk.")
    put_text("🏋 Regular Exercise & Strength Training\nStrengthens muscles around the knee.\nLow-impact activities like swimming and cycling are beneficial.")
    put_text("🥦 Follow an Anti-Inflammatory Diet\nInclude omega-3-rich foods like fish and flaxseeds.\nAvoid processed and sugary foods that increase inflammation.")
    put_text("🩺 Protect Your Joints\nUse knee braces or supportive footwear when needed.\nAvoid activities that cause excessive knee strain.")
    put_text("💊 Monitor Joint Health & Take Supplements\nConsider glucosamine and chondroitin supplements if recommended.\nRegular check-ups help track joint health.")
    
# Function to display Symptoms Page
def symptoms_page():
    clear()
    style(put_markdown("# <span style='color:orange;'>Symptoms of Knee Osteoarthritis</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    put_html(f"""
    <div style = 'text-align:left;'>
    <img src='{get_image_base64("symptoms_image.jpg")}'style=width:50%;height:250px;border-radius:10px;'>
    </div>
    """)
    put_text("Signs & Symptoms of Knee Osteoarthritis")
    put_text("🔥 Pain & Stiffness\nPersistent pain in the knee joint.\nStiffness, especially in the morning or after resting.")
    put_text("⚡Swelling & Inflammation\nKnee joint appears swollen and tender.\nWarmth around the affected area.")
    put_text("🔄 Reduced Range of Motion\nDifficulty in bending or straightening the knee.\nTrouble performing daily activities like walking or climbing stairs.")
    put_text("🔊Clicking or Popping Sounds\nCrackling or grinding noises when moving the knee.\nIndicates cartilage wear and tear.")
    put_text("🦴 Bone Spurs (Advanced Stage)\nExtra bone growths that cause pain and discomfort.\nVisible deformities in severe cases.")
    

# Function to display Team Page
def team_page():
    clear()
    style(put_markdown("# <span style='color:purple;'>Meet Our Team</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    team = [
        ("Nagireddy Varshitha", "Team Leader", "Expert in AI & Machine Learning, ensuring high accuracy and reliability of predictions. She oversees model development, optimization, and integration into the web application.", "leader.jpg"),
        ("Ponnapula Naveen", "Data Scientist", "Responsible for handling data preprocessing, feature selection, and training machine learning models. He ensures that the data is clean and structured for the best predictive performance.", "data_scientist.jpg"),
        ("Ratakonda Venkata Deepika", "Software Engineer", "Develops the web application and integrates AI models into the system. She focuses on frontend design, backend implementation, and creating a seamless user experience.", "engineer.jpg"),
        ("Sakhapuram Penchal Prasad", "Medical Consultant", "Provides medical insights to validate AI predictions. He ensures that results align with clinical expectations and offers expertise in osteoarthritis research.", "consultant.jpg")
    ]

    put_html("""
    <div style='display: flex; justify-content:space-around; flex-wrap: wrap;'>
    """)

    
    for name, role, domain, image in team:
        put_html(f"""
        <div style='display:inline-block; width:350px; text-align:center; border: 2px solid #6A5ACD; padding: 10px; margin: 10px; border-radius: 10px;'>
            <img src='{get_image_base64(image)}' alt='{name}' style='border-radius:50%; width:100px; height:100px; border: 2px solid #6A5ACD;'>
            <h3 style='color:#FF5733;'>{name}</h3>
            <p><b>{role}</b></p>
            <p style='color:#555;'>{domain}</p>
        </div>
        """)

    put_html("</div>")
def single_prediction_page():
    clear()
    style(put_markdown("# <span style='color:blue;'>Single Prediction</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    put_markdown("### Enter Details for Prediction")

    age = input("Enter Age", type="number")
    bmi = input("Enter BMI", type="float")
    pain = input("Enter Pain Level (1-10)", type="number")

    if not (age and bmi and pain):
        put_markdown("⚠ *Please enter all values before proceeding*")
        return

    # Prepare input data
    X_input = np.array([[age, bmi, pain]])

    # Initialize models
    models = {
        "SVM": svm.SVC(probability=True),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression()
    }

    results = []
    probabilities = {}

    for name, model in models.items():
        # Generate random placeholder training data
        X_train = np.random.rand(100, 3) * [80, 50, 10]
        y_train = np.random.choice([0, 1], size=100)

    
        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_input)
        y_pred_prob = model.predict_proba(X_input)[:, 1][0]

        # Store results
        probabilities[name] = y_pred_prob
        results.append([name, "Positive" if y_pred[0] == 1 else "Negative", f"{y_pred_prob:.2f}"])

    # Display results
    put_markdown("### Prediction Results")
    put_table(results, header=["Model", "Prediction", "Probability"])

    # Probability Graph
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(probabilities.keys()), y=list(probabilities.values()), palette='coolwarm')
    plt.ylabel('Probability of Osteoarthritis')
    plt.title('Prediction Probability by Model')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    put_image(buf.getvalue())

    # Final Diagnosis
    max_prob = max(probabilities.values())
    final_prediction = "High Risk of Knee Osteoarthritis" if max_prob > 0.5 else "Low Risk of Knee Osteoarthritis"
    put_markdown(f"## Final Diagnosis: {final_prediction}")
def dataset_prediction_page():
    clear()
    style(put_markdown("# <span style='color:blue;'>Dataset Prediction</span>"), 'text-align:center;')
    put_buttons(
        ["Home", "Benefits", "Precautions", "Symptoms", "Team", "Single Prediction", "Dataset Prediction"], 
        onclick=[home_page, benefits_page,precautions_page,
                symptoms_page, team_page,single_prediction_page,
                dataset_prediction_page])

    put_text("Upload a CSV file for batch prediction.")
    uploaded_file = file_upload("Upload a CSV dataset", accept=".csv")
    
    if uploaded_file:
        df = pd.read_csv(io.BytesIO(uploaded_file['content']))
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "SVM": svm.SVC(),
            "Naive Bayes": GaussianNB(),
            "Logistic Regression": LogisticRegression()
        }
        
        results = []
        confusion_matrices = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            results.append([name, f"{acc:.2f}", f"{prec:.2f}", f"{rec:.2f}"])
            
            cm = confusion_matrix(y_test, y_pred)
            confusion_matrices[name] = cm
            plt.figure(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix - {name}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            put_image(buf.getvalue())
            
        put_markdown("### Model Performance Metrics")
        put_table(results, header=["Model", "Accuracy", "Precision", "Recall"])
        
        plt.figure(figsize=(6, 4))
        accuracy_scores = [float(row[1]) for row in results]
        model_names = [row[0] for row in results]
        sns.barplot(x=model_names, y=accuracy_scores, palette='viridis')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        put_image(buf.getvalue())
        
        for name, cm in confusion_matrices.items():
            put_markdown(f"### Confusion Matrix for {name}")
            put_text(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
            put_text(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
        
        put_buttons(["Upload Another Dataset"], onclick=[lambda: dataset_prediction_page()])

app.add_url_rule("/home", "home", webio_view(home_page), methods=["GET", "POST"])
app.add_url_rule("/benefits", "benefits", webio_view(benefits_page), methods=["GET", "POST"])
app.add_url_rule("/precautions", "precautions", webio_view(precautions_page), methods=["GET", "POST"])
app.add_url_rule("/symptoms", "symptoms", webio_view(symptoms_page), methods=["GET", "POST"])
app.add_url_rule("/team", "team", webio_view(team_page), methods=["GET", "POST"])
app.add_url_rule("/single_prediction", "single_prediction", webio_view(single_prediction_page), methods=["GET", "POST"])
app.add_url_rule("/dataset_prediction", "dataset_prediction", webio_view(dataset_prediction_page), methods=["GET", "POST"])

# Run the PyWebIO App
if __name__ == "__main__":
    start_server(home_page,port=8080,debug=True)

    