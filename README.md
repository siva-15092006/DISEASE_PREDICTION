AI-Powered Medicine Recommendation System
📌 Overview

This project is an AI-driven medicine recommendation system designed to suggest suitable treatments based on patient data. It combines a local database with external API integration to handle both known and unknown medical cases.

🚀 Features

Database-Driven Recommendations → Matches patient conditions with known medicines stored in the system.

API Integration for Unknown Cases → If a condition is not found in the database, the system queries an external API to fetch relevant medicines.

AI Model for Recommendations → Uses ML algorithms to learn from historical patient-medication data for personalized suggestions.

Scalable Architecture → Built with FastAPI for easy deployment as a web service.

🛠️ Tech Stack

Programming: Python

ML/DL: Scikit-Learn, TensorFlow

Database: MySQL / SQLite

API Framework: FastAPI, Requests

Other Tools: Pandas, NumPy, Docker

📂 Project Workflow

Patient data is entered into the system (symptoms, condition, history).

The system checks the local medicine database for recommendations.

If not found, it calls an external medical API to fetch possible medicines.

AI model ranks the suggestions based on condition-medicine similarity.

Recommendations are displayed via API/Web interface.

📊 Example Output

Input: Patient with "Diabetes Type 2"
Output: Recommended medicines: Metformin, Glipizide, Sitagliptin (via DB)

Input: Rare condition not in DB
Output: Recommendations fetched via API → Suggested medicines displayed.

📜 Future Improvements

Add Drug Interaction Checker for safety.

Improve NLP-based symptom analysis.

Integrate with EHR (Electronic Health Records) for hospital use.

🤝 Contribution

Pull requests are welcome! Please open an issue first to discuss major changes.

📌 Author

Siva Kumar Raju
