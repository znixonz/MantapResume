###### Packages Used ######
import streamlit as st  # core package used in this project
import base64
import random
import time
import datetime
import os
import socket
import platform
# import geocoder # Removed for simplicity in single file, can be added back if geo-location is fully implemented
import secrets
import io
import plotly.express as px
import os  # to create visualisations at the admin session
# import plotly.graph_objects as go # Not explicitly used, can be removed if not needed for future
# from geopy.geocoders import Nominatim # Removed for simplicity

# libraries used to parse the pdf files
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
import csv
# import uuid # Not explicitly used, can be removed
import json
from openai import OpenAI
from docx import Document # For .docx file processing
from dotenv import load_dotenv

st.set_page_config(layout="wide")

###### Placeholder data (formerly from Courses.py) ######
# IMPORTANT: Populate these lists with your actual data!
ds_course = [
    ("Data Science Bootcamp", "http://example.com/ds_bootcamp"),
    ("Machine Learning A-Z", "http://example.com/ml_az"),
    ("Python for Data Science", "http://example.com/python_ds")
]
web_course = [
    ("Full-Stack Web Developer Course", "http://example.com/web_fullstack"),
    ("React - The Complete Guide", "http://example.com/react_guide"),
    ("Django for Beginners", "http://example.com/django_beg")
]
android_course = [
    ("Android App Development Masterclass", "http://example.com/android_masterclass"),
    ("Kotlin for Android Developers", "http://example.com/kotlin_android")
]
ios_course = [
    ("iOS & Swift - The Complete iOS App Development Bootcamp", "http://example.com/ios_bootcamp"),
    ("SwiftUI Masterclass", "http://example.com/swiftui_master")
]
uiux_course = [
    ("UI/UX Design Specialization", "http://example.com/uiux_spec"),
    ("Figma UI UX Design Essentials", "http://example.com/figma_essentials")
]

resume_videos = [
    "https://www.youtube.com/watch?v=BYUy1yvjHxE&ab_channel=LifeatGoogle", # Example video
    "https://www.youtube.com/watch?v=iybdUPYXPEw&ab_channel=FarahSharghi"  # Example video
]
interview_videos = [
    "https://www.youtube.com/watch?v=XOtrOSatBoY&ab_channel=LifeatGoogle", # Example video
    "https://www.youtube.com/watch?v=eIMR82oO2Dc&ab_channel=GoogleStudents"  # Example video
]

load_dotenv()

# Alibaba Cloud API endpoints and configuration
ALIBABA_API_KEY = os.getenv('API_KEY')
ALIBABA_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

###### Alibaba Cloud LLM Functions ######

def call_alibaba_llm(prompt, max_tokens=2048, system_prompt="You are a helpful assistant."):
    """
    Call Alibaba Cloud's LLM with the given prompt using OpenAI-compatible interface
    """
    try:
        client = OpenAI(api_key=ALIBABA_API_KEY, base_url=ALIBABA_BASE_URL)
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.8
        )
        return completion.choices[0].message.content
    except Exception as e:
        error_message = f"Error calling Alibaba Cloud LLM: {e}"
        st.error(error_message)
        print(error_message) # Log to console for server-side debugging
        # Consider logging the prompt as well for debugging (server-side only for security)
        print(f"Failed prompt (first 100 chars): {prompt[:100]}")
        print("For more information, see: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
        return None

def parse_llm_json_response(response_text):
    """
    Attempts to parse a JSON object from the LLM's response text.
    Handles cases where the JSON might be embedded in markdown or have surrounding text.
    """
    if not response_text:
        return None
    try:
        # Try direct parsing first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract from markdown
        if response_text.strip().startswith("```json"):
            cleaned_response = response_text.strip()[7:-3].strip() # Remove ```json and ```
        elif response_text.strip().startswith("```"):
            cleaned_response = response_text.strip()[3:-3].strip() # Remove ```
        else:
            # Fallback to finding the first '{' and last '}'
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start:
                cleaned_response = response_text[json_start:json_end]
            else: # No JSON object found
                 st.error(f"Could not find a valid JSON object in LLM response: {response_text[:200]}...")
                 return None
        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse extracted LLM JSON response: {e}. Extracted string was: {cleaned_response[:200]}...")
            return None

class ResumeParser:
    """
    Resume parsing class using Alibaba Cloud LLM model
    """
    def __init__(self, resume_file_path):
        self.resume_file_path = resume_file_path
        self.resume_text, self.actual_num_pages = self.extract_text_and_pages()

    def extract_text_and_pages(self):
        """Extract text and count pages from resume PDF or DOCX"""
        text = ""
        page_count = 0
        try:
            if self.resume_file_path.endswith('.pdf'):
                text, page_count = pdf_reader(self.resume_file_path)
            elif self.resume_file_path.endswith('.docx'):
                doc = Document(self.resume_file_path)
                full_text = [para.text for para in doc.paragraphs]
                text = '\n'.join(full_text)
                # Page count for docx is not straightforward, can estimate or use a library if crucial.
                # For simplicity, we'll report 1 or based on text length if needed.
                # Or, we can rely on the LLM if it's part of the extraction prompt.
                # For now, let's default to 1 for docx if not otherwise determined.
                page_count = 1 # Simplification for DOCX page count
            else:
                st.error("Unsupported file format. Please upload a PDF or DOCX file.")
                return "", 0
        except Exception as e:
            st.error(f"Error reading file {os.path.basename(self.resume_file_path)}: {e}")
            return "", 0
        return text, page_count

    def get_extracted_data(self):
        """Extract structured data from resume using Alibaba Cloud LLM"""
        if not self.resume_text:
            return None

        prompt = f"""
        Extract the following information from this resume. For each field, provide only the extracted information.
        If a field is not found, return "Not found" or an empty list for skills.
        Ensure your entire response is ONLY a valid JSON object.

        Resume text:
        ---
        {self.resume_text}
        ---

        Please extract and return the following in JSON format:
        1. name: Full name of the person
        2. email: Email address
        3. mobile_number: Phone number
        4. skills: List of technical and soft skills (as a JSON array of strings)
        5. education: Summary of educational background (string)
        6. experience: Summary of work experience (string)
        7. degree: Highest degree obtained (string)
        """
        # `no_of_pages` will be handled by self.actual_num_pages

        response_text = call_alibaba_llm(prompt)
        data = parse_llm_json_response(response_text)

        if data:
            expected_fields = ['name', 'email', 'mobile_number', 'skills', 'education', 'experience', 'degree']
            for field in expected_fields:
                if field not in data:
                    data[field] = "Not found" if field != 'skills' else []

            if isinstance(data.get('skills'), str): # Ensure skills is a list
                if data['skills'].lower() == "not found" or not data['skills']:
                    data['skills'] = []
                else: # Attempt to split if it's a comma-separated string by mistake from LLM
                    data['skills'] = [skill.strip() for skill in data['skills'].split(',') if skill.strip()]
            elif not isinstance(data.get('skills'), list):
                 data['skills'] = []

            data['no_of_pages'] = self.actual_num_pages if self.actual_num_pages > 0 else 1 # Use actual page count
            return data
        return None

def analyze_resume_for_jobs_and_get_feedback(resume_text, skills):
    """
    Use Alibaba Cloud LLM to analyze resume, match with job categories, and provide feedback.
    """
    prompt = f"""
    Analyze this resume text and skills to determine the most suitable job category, experience level,
    resume quality score, specific improvement reasons, and an explanation for the job category match.
    Ensure your entire response is ONLY a valid JSON object.

    Resume text:
    ---
    {resume_text}
    ---

    Skills:
    {', '.join(skills)}

    Determine which ONE of these job categories is the best match:
    - Project Manager
    - Product Manager
    - Machine Learning Engineer
    - Fullstack Engineer
    - Software Engineer
    - Data Engineer
    - Data Analyst
    - Data Science
    - FrontEnd Development
    - Backend Development
    - Android Development
    - IOS Development
    - UI-UX Development
    - Other (Specify if none of the above fit well)

    Provide the following in a JSON object:
    1. "job_category": The best matching job category from the list above.
    2. "experience_level": The candidate's experience level (e.g., Fresher, Intern, Junior, Intermediate, Experienced, Senior).
    3. "resume_score": A score from 0-100 based on overall resume quality, ATS friendliness, and clarity.
    4. "improvement_reasons": An array of 3-5 specific, actionable reasons why the resume needs improvement or how it can be made better.
    5. "match_explanation": A brief (2-3 sentences) explanation of why this job category is a good match based on the resume and skills.

    Example JSON structure:
    {{
      "job_category": "Web Development",
      "experience_level": "Intermediate",
      "resume_score": 75,
      "improvement_reasons": [
        "Quantify achievements in past roles with numbers.",
        "Tailor the resume summary to highlight web development projects.",
        "Add a portfolio link if available."
      ],
      "match_explanation": "The candidate's experience with React and Node.js, coupled with several web-based projects, strongly aligns with Web Development roles."
    }}
    """
    response_text = call_alibaba_llm(prompt)
    analysis_data = parse_llm_json_response(response_text)

    if analysis_data:
        return analysis_data

    # Default fallback response if LLM call or parsing fails
    return {
        "job_category": "NA",
        "experience_level": "Fresher",
        "resume_score": 50,
        "improvement_reasons": [
            "Could not properly analyze the resume due to an error.",
            "Please check the resume format and content.",
            "Ensure the resume contains clear information about skills and experience."
        ],
        "match_explanation": "Analysis could not be completed."
    }

def summarize_resume_llm(resume_text):
    """Generates a concise summary of the resume using LLM."""
    if not resume_text: return "Could not generate summary: No resume text provided."
    prompt = f"""
    Provide a concise summary (3-4 sentences) of the following resume.
    Highlight key experiences, skills, and the overall professional profile.
    Ensure your entire response is ONLY the summary text, no extra phrases.

    Resume text:
    ---
    {resume_text}
    ---
    Summary:
    """
    summary = call_alibaba_llm(prompt, max_tokens=350)
    return summary if summary else "Summary generation failed or returned empty."

def get_career_path_suggestions_llm(job_category, experience_level, skills_list):
    """Suggests career paths using LLM."""
    if not job_category or job_category == "NA": return {"career_paths": [{"path_title": "No specific path suggested due to insufficient initial analysis.", "focus_areas": []}]}
    prompt = f"""
    A candidate is currently identified as '{experience_level}' in '{job_category}' with skills: {', '.join(skills_list)}.
    Suggest 5-6 potential career path advancements or related technical roles they could aim for.
    For each suggestion, list 1-2 key areas or skills to focus on for that path.
    Return this as a JSON object with a list of suggestions.
    Ensure your entire response is ONLY a valid JSON object.

    Example JSON:
    {{
      "career_paths": [
        {{ "path_title": "Senior {job_category} Developer", "focus_areas": ["Project leadership", "Advanced {job_category.split()[0] if job_category else ''} frameworks"] }},
        {{ "path_title": "Cross-functional Role (e.g., DevOps for {job_category.split()[0] if job_category else ''})", "focus_areas": ["CI/CD pipelines", "Cloud infrastructure"] }}
      ]
    }}
    """
    response_text = call_alibaba_llm(prompt)
    suggestions = parse_llm_json_response(response_text)
    return suggestions if suggestions and "career_paths" in suggestions else {"career_paths": [{"path_title": "Career path suggestion generation failed.", "focus_areas": []}]}


###### Preprocessing & Helper functions ######
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def pdf_reader(file_path):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    page_count = 0
    with open(file_path, 'rb') as fh:
        for i, page in enumerate(PDFPage.get_pages(fh, caching=True, check_extractable=True)):
            try:
                page_interpreter.process_page(page)
                page_count = i + 1
            except Exception as e: # Catch errors during processing of a specific page
                st.warning(f"Skipping a page in PDF due to error: {e}")
                continue # Skip to the next page
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text, page_count

def show_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("PDF file not found. It might have been deleted or moved.")
    except Exception as e:
        st.error(f"Could not display PDF: {e}")

def recommend_skills(job_category):
    skill_recommendations = {
        "Data Science": ['Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning', 'Data Visualization (Tableau, PowerBI)', 'Statistics', 'Big Data (Spark, Hadoop)'],
        "Web Development": ['HTML', 'CSS', 'JavaScript (ES6+)', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express.js', 'Django', 'Ruby on Rails', 'PHP', 'RESTful APIs', 'GraphQL', 'Databases (SQL, NoSQL)'],
        "Android Development": ['Kotlin', 'Java', 'Android SDK', 'XML', 'Jetpack Compose', 'Firebase', 'RESTful APIs', 'Git'],
        "IOS Development": ['Swift', 'Objective-C', 'Xcode', 'UIKit', 'SwiftUI', 'Core Data', 'Firebase', 'RESTful APIs'],
        "UI-UX Development": ['Figma', 'Adobe XD', 'Sketch', 'User Research', 'Wireframing', 'Prototyping', 'Usability Testing', 'Interaction Design', 'Visual Design'],
        "Project Manager": ['Agile/Scrum', 'JIRA', 'MS Project', 'Risk Management', 'Stakeholder Communication', 'Budgeting', 'Resource Planning', 'Project Documentation', 'PMP Certification', 'Kanban', 'Confluence'],
        "Product Manager": ['Market Research', 'User Stories', 'Product Roadmapping', 'A/B Testing', 'Product Analytics', 'Competitive Analysis', 'Wireframing', 'Product Strategy', 'Business Requirements', 'Prioritization Frameworks', 'Product Lifecycle Management'],
        "Machine Learning Engineer": ['Python', 'TensorFlow', 'PyTorch', 'Scikit-learn', 'MLOps', 'Docker', 'Kubernetes', 'Cloud ML Services (AWS SageMaker, Azure ML)', 'Feature Engineering', 'Model Deployment', 'ML Algorithms', 'Data Pipelines'],
        "Fullstack Engineer": ['JavaScript/TypeScript', 'React/Angular/Vue', 'Node.js', 'Database Design', 'RESTful APIs', 'GraphQL', 'Git', 'CI/CD', 'Cloud Services', 'Docker', 'Testing Frameworks', 'System Design', 'Frontend & Backend Architecture'],
        "Software Engineer": ['Data Structures', 'Algorithms', 'System Design', 'Git', 'CI/CD', 'Testing (Unit, Integration)', 'Design Patterns', 'Programming Languages (Java/Python/C++/etc.)', 'Cloud Technologies', 'Microservices', 'DevOps'],
        "Data Engineer": ['SQL', 'Python', 'ETL Pipelines', 'Data Warehousing', 'Spark', 'Hadoop', 'Airflow', 'Cloud Data Services (Snowflake, BigQuery)', 'Data Modeling', 'Database Administration', 'Kafka', 'Data Governance'],
        "Data Analyst": ['SQL', 'Excel (Advanced)', 'Python/R', 'Data Visualization (Tableau, PowerBI)', 'Statistical Analysis', 'A/B Testing', 'Business Intelligence Tools', 'Dashboard Design', 'Data Cleaning', 'Google Analytics', 'Business Acumen'],
        "Other": ["Consider specializing further or exploring related fields based on interests."]
    }
    return skill_recommendations.get(job_category, ["No specific skill recommendations for this category yet."])

def get_course_recommendations(job_category):
    course_map = {
        "Data Science": ds_course,
        "Web Development": web_course,
        "Android Development": android_course,
        "IOS Development": ios_course,
        "UI-UX Development": uiux_course,
        "Other": [("General professional development courses", "[http://example.com/prof_dev](http://example.com/prof_dev)")]
    }
    return course_map.get(job_category, [])

def course_recommender(course_list_for_category):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course_names = []
    if not course_list_for_category:
        st.write("No specific course recommendations available for this field yet.")
        return rec_course_names

    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, min(10, len(course_list_for_category)), min(5, len(course_list_for_category)))
    random.shuffle(course_list_for_category)
    for c_name, c_link in course_list_for_category:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course_names.append(c_name)
        if c == no_of_reco:
            break
    return rec_course_names

###### Setting Page Configuration ######
# Ensure the logo path is correct or remove if no logo is available
try:
    page_icon_img = Image.open('./Logo/recommend.png')
except FileNotFoundError:
    page_icon_img = "📄" # Fallback to emoji if logo not found
    st.warning("Logo image './Logo/recommend.png' not found. Using default icon.")
try:
    st.set_page_config(
    page_title="Mantap Resume Analyzer",
    page_icon=page_icon_img,
    )
except:
    pass

###### Main function run() ######
def run():
    # Create directory for uploaded resumes if it doesn't exist
    if not os.path.exists('./Uploaded_Resumes'):
        os.makedirs('./Uploaded_Resumes')

    ###### CODE FOR CLIENT SIDE (USER) ######
    _, col, _ = st.columns([0.1, 0.8, 0.1])

    with col:
        with open( "./style.css" ) as css:
            st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)
            st.title("🚀 AI Resume Analyzer & Job Matchmaker")
            st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload Your Resume (PDF or DOCX) and Get Smart Insights!</h5>''', unsafe_allow_html=True)

            pdf_file = st.file_uploader("Choose your Resume", type=["pdf", "docx"])
            if pdf_file is not None:
                # Save uploaded file
                save_file_path = os.path.join('./Uploaded_Resumes/', secrets.token_hex(8) + "_" + pdf_file.name)
                pdf_name_original = pdf_file.name
                with open(save_file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())

                with st.spinner('Analyzing your resume... This may take a moment.'):
                    # Display PDF (if it's a PDF)
                    if save_file_path.endswith('.pdf'):
                        show_pdf(save_file_path)

                    parser = ResumeParser(save_file_path)
                    resume_data = parser.get_extracted_data()
                    resume_text_content = parser.resume_text

                if resume_data and resume_text_content:
                    st.header("**Resume Insights ✨**")
                    st.success(f"Hello {resume_data.get('name', 'User')}!")

                    # Display Basic Info
                    st.subheader("**Basic Information Extracted**")
                    basic_info_cols = st.columns(2)
                    basic_info_cols[0].text(f"Name: {resume_data.get('name', 'N/A')}")
                    basic_info_cols[0].text(f"Email: {resume_data.get('email', 'N/A')}")
                    basic_info_cols[1].text(f"Contact: {resume_data.get('mobile_number', 'N/A')}")
                    basic_info_cols[1].text(f"Degree: {str(resume_data.get('degree', 'N/A'))}")
                    st.text(f"Resume Pages: {str(resume_data.get('no_of_pages', 'N/A'))}")

                    # LLM Analysis for Job Matching, Feedback, etc.
                    with st.spinner("Getting AI-powered analysis and recommendations..."):
                        analysis_result = analyze_resume_for_jobs_and_get_feedback(resume_text_content, resume_data.get('skills', []))
                        resume_summary = summarize_resume_llm(resume_text_content)
                        career_paths_data = get_career_path_suggestions_llm(
                            analysis_result.get("job_category", "NA"),
                            analysis_result.get("experience_level", "Fresher"),
                            resume_data.get('skills', [])
                        )

                    st.subheader("**AI Analysis & Recommendations 🤖**")
                    reco_field = analysis_result.get("job_category", "NA")
                    cand_level = analysis_result.get("experience_level", "Fresher")
                    resume_score_val = analysis_result.get("resume_score", 50)
                    match_explanation_text = analysis_result.get("match_explanation", "No explanation available.")

                    # Display Experience Level
                    level_color_map = {"Fresher": "#d73b5c", "Intermediate": "#1ed760", "Experienced": "#fba171", "Junior": "#5dade2", "Senior": "#f39c12"}
                    level_color = level_color_map.get(cand_level, "#7f8c8d") # Default color
                    st.markdown(f'''<h4 style='text-align: left; color: {level_color};'>You seem to be at an {cand_level} level.</h4>''', unsafe_allow_html=True)

                    # Display Resume Score (example with progress bar)
                    st.write(f"**Overall Resume Score:** {resume_score_val}/100")
                    st.progress(int(resume_score_val))

                    # Display Resume Summary
                    st.subheader("**Resume Summary 📜**")
                    st.markdown(f"> {resume_summary}")

                    # Skills Analysis and Recommendation
                    st.subheader("**Skills Analysis & Recommendations 💡**")
                    st_tags(label='### Your Extracted Skills:', text='Review your skills below.', value=resume_data.get('skills', []), key='extracted_skills_display')

                    if reco_field != "NA" and reco_field != "Other":
                        st.success(f"**Our analysis suggests you are a good fit for roles in: {reco_field}**")
                        st.markdown(f"**Reasoning:** *{match_explanation_text}*")

                        recommended_skills_list = recommend_skills(reco_field)
                        st_tags(label='### Recommended Skills to Add/Highlight:', text='Consider these skills for your target roles.', value=recommended_skills_list, key='recommended_skills_display')
                        st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding relevant skills can significantly boost🚀 your job prospects💼!</h5>''', unsafe_allow_html=True)

                        course_list_for_reco_field = get_course_recommendations(reco_field)
                        recommended_courses = course_recommender(course_list_for_reco_field)
                    else:
                        st.warning(f"**Job Category:** {reco_field if reco_field != 'NA' else 'Could not confidently determine a specific job category.'}")
                        st.markdown(f"**Note:** *{match_explanation_text}*")
                        recommended_skills_list = ["Focus on foundational skills relevant to your general interests."]
                        st_tags(label='### General Skill Recommendations:', text='Consider these general skills.', value=recommended_skills_list, key='general_skills_display')
                        recommended_courses = ["General professional development courses."]

                    # Resume Improvement Tips
                    st.subheader("**Resume Improvement Tips 🥂**")
                    improvement_reasons_list = analysis_result.get("improvement_reasons", [])
                    if improvement_reasons_list:
                        for i, reason in enumerate(improvement_reasons_list, 1):
                            st.markdown(f"**{i}.** {reason}")
                    else:
                        st.info("No specific improvement tips generated, or resume is looking good!")

                    # Career Path Suggestions
                    st.subheader("**Potential Career Paths 🧭**")
                    if career_paths_data and career_paths_data.get("career_paths"):
                        for path in career_paths_data["career_paths"]:
                            st.markdown(f"**Path:** {path.get('path_title', 'N/A')}")
                            if path.get("focus_areas"):
                                st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;*Focus on: {', '.join(path['focus_areas'])}*")
                    else:
                        st.info("Could not generate specific career path suggestions at this time.")

                    # Bonus Videos
                    st.markdown("---")
                    st.header("**Bonus Content 🎁**")
                    vid_cols = st.columns(2)
                    if resume_videos:
                        vid_cols[0].subheader("Resume Writing Tips")
                        vid_cols[0].video(random.choice(resume_videos))  # Pass the URL directly
                    if interview_videos:
                        vid_cols[1].subheader("Interview Tips")
                        vid_cols[1].video(random.choice(interview_videos))

                    # Clean up uploaded file after processing
                    try:
                        os.remove(save_file_path)
                    except Exception as e:
                        st.warning(f"Could not delete temporary file {save_file_path}: {e}")
                else:
                    st.error('Something went wrong... Unable to analyze the resume. The file might be corrupted, password-protected, or in an unreadable format.')
                    if not resume_text_content:
                        st.error("Could not extract text from the uploaded file.")
                    if not resume_data and resume_text_content: # Text extracted but LLM failed
                        st.error("Extracted text from resume, but failed to get structured data from LLM.")

# Run the app
if __name__ == "__main__":
    run()
