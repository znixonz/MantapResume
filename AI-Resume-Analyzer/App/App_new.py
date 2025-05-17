# Developed by dnoobnerd [https://dnoobnerd.netlify.app]    Modified to use Alibaba Cloud LLM

###### Packages Used ######
import streamlit as st  # core package used in this project
import pandas as pd
import base64, random
import time, datetime
import os
import socket
import platform
import geocoder
import secrets
import io, random
import plotly.express as px  # to create visualisations at the admin session
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course, web_course, android_course, ios_course, uiux_course, resume_videos, interview_videos
import nltk
import csv
import uuid
import requests
import json
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

# Alibaba Cloud API endpoints and configuration
ALIBABA_API_KEY = os.getenv('API_KEY')
ALIBABA_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

nltk.download('stopwords')

###### Alibaba Cloud LLM Functions ######

def call_alibaba_llm(prompt, max_tokens=2048, system_prompt="You are a helpful assistant."):
    """
    Call Alibaba Cloud's LLM with the given prompt using OpenAI-compatible interface
    """
    try:
        # Initialize OpenAI client with Alibaba Cloud settings
        client = OpenAI(
            api_key=ALIBABA_API_KEY,
            base_url=ALIBABA_BASE_URL,
        )

        # Create the completion request
        completion = client.chat.completions.create(
            model="qwen-max", # Using Qwen-Max model
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.5,
            top_p=0.8
        )
        
        # Extract and return the response text
        return completion.choices[0].message.content
        
    except Exception as e:
        error_message = f"Error calling Alibaba Cloud LLM: {e}"
        st.error(error_message)
        print(error_message)
        print("For more information, see: https://www.alibabacloud.com/help/en/model-studio/developer-reference/error-code")
        return None

class ResumeParser:
    """
    Resume parsing class using Alibaba Cloud LLM model instead of spaCy
    """
    def __init__(self, resume_path):
        self.resume_path = resume_path
        self.resume_text = self.extract_text()
        
    def extract_text(self):
        """Extract text from resume PDF"""
        if self.resume_path.endswith('.pdf'):
            return pdf_reader(self.resume_path)
        elif self.resume_path.endswith('.docx'):
            # Add code to extract text from DOCX if needed
            return "DOCX extraction not implemented"
        return ""
    
    def get_extracted_data(self):
        """Extract structured data from resume using Alibaba Cloud LLM"""
        resume_text = self.extract_text()
        
        if not resume_text:
            return None
            
        # Create a prompt for the LLM to extract structured data
        prompt = f"""
        Extract the following information from this resume. For each field, provide only the extracted information without any additional text.
        If a field is not found, return "Not found".
        
        Resume text:
        {resume_text}
        
        Please extract and return the following in JSON format:
        1. name: Full name of the person
        2. email: Email address 
        3. mobile_number: Phone number
        4. skills: List of technical and soft skills (as an array)
        5. education: Educational background
        6. experience: Work experience
        7. degree: Highest degree obtained
        8. no_of_pages: Always use 1 for this field
        """
        
        # Call the LLM to extract information
        try:
            response = call_alibaba_llm(prompt)
            if response:
                # Parse the JSON response
                # The response might contain additional text, so we need to extract the JSON part
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    data = json.loads(json_str)
                    
                    # Make sure all expected fields are present
                    expected_fields = ['name', 'email', 'mobile_number', 'skills', 'education', 'experience', 'degree', 'no_of_pages']
                    for field in expected_fields:
                        if field not in data:
                            data[field] = "Not found"
                            
                    # Ensure skills is a list
                    if isinstance(data['skills'], str):
                        if data['skills'] == "Not found":
                            data['skills'] = []
                        else:
                            data['skills'] = [skill.strip() for skill in data['skills'].split(',')]
                            
                    return data
                else:
                    st.error("Failed to parse LLM response as JSON")
            else:
                st.error("No response from LLM")
        except Exception as e:
            st.error(f"Error processing resume with LLM: {e}")
            
        return None
    
def analyze_resume_for_jobs(resume_text, skills):
    """
    Use Alibaba Cloud LLM to analyze resume and match with job categories
    """
    prompt = f"""
    Analyze this resume and skills to determine which job category the person is most suitable for.
    
    Resume text:
    {resume_text}
    
    Skills:
    {', '.join(skills)}
    
    Determine which ONE of these categories is the best match:
    1. Data Science
    2. Web Development
    3. Android Development
    4. IOS Development
    5. UI-UX Development
    
    Also provide:
    1. The candidate's experience level (Fresher, Intermediate, or Experienced)
    2. A score from 0-100 based on the resume quality
    3. Three specific reasons why the resume needs improvement
    
    Return your analysis in JSON format.
    """
    
    try:
        response = call_alibaba_llm(prompt)
        if response:
            # Extract the JSON part from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
    except Exception as e:
        st.error(f"Error analyzing resume with LLM: {e}")
        
    # Default fallback response
    return {
        "job_category": "NA",
        "experience_level": "Fresher",
        "resume_score": 50,
        "improvement_reasons": [
            "Could not properly analyze the resume",
            "Please check the resume format",
            "Make sure the resume contains relevant information"
        ]
    }

def recommend_skills(job_category):
    """Return recommended skills based on job category"""
    skill_recommendations = {
        "Data Science": ['Data Visualization','Predictive Analysis','Statistical Modeling','Data Mining',
                         'Clustering & Classification','Data Analytics','Quantitative Analysis','Web Scraping',
                         'ML Algorithms','Keras','Pytorch','Probability','Scikit-learn','Tensorflow',
                         "Flask",'Streamlit'],
        
        "Web Development": ['React','Django','Node JS','React JS','php','laravel','Magento','wordpress',
                           'Javascript','Angular JS','c#','Flask','SDK'],
        
        "Android Development": ['Android','Android development','Flutter','Kotlin','XML','Java','Kivy',
                               'GIT','SDK','SQLite'],
        
        "IOS Development": ['IOS','IOS Development','Swift','Cocoa','Cocoa Touch','Xcode','Objective-C',
                           'SQLite','Plist','StoreKit',"UI-Kit",'AV Foundation','Auto-Layout'],
        
        "UI-UX Development": ['UI','User Experience','Adobe XD','Figma','Zeplin','Balsamiq','Prototyping',
                             'Wireframes','Storyframes','Adobe Photoshop','Editing','Illustrator',
                             'After Effects','Premier Pro','Indesign','Wireframe','Solid','Grasp','User Research']
    }
    
    return skill_recommendations.get(job_category, [])

def get_course_recommendations(job_category):
    """Return course recommendations based on job category"""
    course_map = {
        "Data Science": ds_course,
        "Web Development": web_course,
        "Android Development": android_course,
        "IOS Development": ios_course,
        "UI-UX Development": uiux_course
    }
    
    return course_map.get(job_category, [])

###### Preprocessing functions ######

# Generates a link allowing the data in a given panda dataframe to be downloaded in csv format
def get_csv_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


# Reads Pdf file and check_extractable
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text


# show uploaded file path to view pdf_display
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


# course recommendations which has data already loaded from Courses.py
def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 👨‍🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database Stuffs ######

# inserting miscellaneous data into user_data.csv
def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong,
                city, state, country, act_name, act_mail, act_mob, name, email,
                res_score, timestamp, no_of_pages, reco_field, cand_level,
                skills, recommended_skills, courses, pdf_name):
    file_exists = os.path.isfile('user_data.csv')
    with open('user_data.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'sec_token', 'ip_add', 'host_name', 'dev_user', 'os_name_ver',
                'latlong', 'city', 'state', 'country', 'act_name', 'act_mail',
                'act_mob', 'name', 'email', 'res_score', 'timestamp',
                'no_of_pages', 'reco_field', 'cand_level', 'skills',
                'recommended_skills', 'courses', 'pdf_name'
            ])
        writer.writerow([
            sec_token, ip_add, host_name, dev_user, os_name_ver,
            latlong, city, state, country, act_name, act_mail,
            act_mob, name, email, res_score, timestamp,
            no_of_pages, reco_field, cand_level, skills,
            recommended_skills, courses, pdf_name
        ])


# inserting feedback data into user_feedback.csv
def insertf_data(feed_name, feed_email, feed_score, comments, Timestamp):
    file_exists = os.path.isfile('user_feedback.csv')
    with open('user_feedback.csv', mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['feed_name', 'feed_email', 'feed_score', 'comments', 'Timestamp'])
        writer.writerow([feed_name, feed_email, feed_score, comments, Timestamp])


###### Setting Page Configuration (favicon, Logo, Title) ######
st.set_page_config(
   page_title="Mantap Resume Analyzer",
   page_icon='./Logo/recommend.png',
   layout="wide",
)


###### Main function run() ######
def run():
    
    # (Logo, Heading, Sidebar etc)
    # img = Image.open('./Logo/RESUM.png')
    # st.image(img)
    # st.sidebar.markdown("# Choose Something...")
    # activities = ["User", "Feedback", "About", "Admin"]
    # choice = st.sidebar.selectbox("Choose among the given options:", activities)
    # link = '<b>Built with 🤍 by <a href="https://dnoobnerd.netlify.app/" style="text-decoration: none; color: #021659;">Mantap</a> and Modified to use Alibaba Cloud LLM</b>' 
    # st.sidebar.markdown(link, unsafe_allow_html=True)
    # st.sidebar.markdown('''
    #     <!-- site visitors -->
    #     <div id="sfct2xghr8ak6lfqt3kgru233378jya38dy" hidden></div>
    #     <noscript>
    #         <a href="https://www.freecounterstat.com" title="hit counter">
    #             <img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" border="0" title="hit counter" alt="hit counter">
    #         </a>
    #     </noscript>
    #     <p>Visitors <img src="https://counter9.stat.ovh/private/freecounterstat.php?c=t2xghr8ak6lfqt3kgru233378jya38dy" title="Free Counter" Alt="web counter" width="60px" border="0" /></p>
    # ''', unsafe_allow_html=True)

    ###### CODE FOR CLIENT SIDE (USER) ######
    if 'User':
        
        # Collecting Miscellaneous Information
        act_name = "N/A"
        act_mail = "N/A"
        act_mob = "N/A"
        sec_token = secrets.token_urlsafe(12)
        host_name = "N/A"
        ip_add = "N/A"
        dev_user = "N/A"
        os_name_ver = "N/A"
        latlong = "N/A"
        city = "N/A"
        state = "N/A"
        country = "N/A"

        # Upload Resume
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload Your Resume, And Get Smart Recommendations</h5>''', unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)

            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            
            # Use our new ResumeParser class with Alibaba Cloud LLM
            parser = ResumeParser(save_image_path)
            resume_data = parser.get_extracted_data()
            resume_text = parser.resume_text
            
            if resume_data:
                st.header("**Resume Analysis 🤘**")
                st.success("Hello "+ resume_data['name'])
                st.subheader("**Your Basic info 👀**")
                try:
                    st.text('Name: '+resume_data['name'])
                    st.text('Email: ' + resume_data['email'])
                    st.text('Contact: ' + resume_data['mobile_number'])
                    st.text('Degree: '+str(resume_data['degree']))                    
                    st.text('Resume pages: '+str(resume_data['no_of_pages']))
                except:
                    pass

                # Now use the LLM to analyze the resume for job matching and experience level
                analysis_result = analyze_resume_for_jobs(resume_text, resume_data['skills'])
                
                reco_field = analysis_result.get("job_category", "NA")
                cand_level = analysis_result.get("experience_level", "Fresher")
                resume_score = analysis_result.get("resume_score", 50)
                
                # Display experience level
                if cand_level == "Fresher":
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are at Fresher level!</h4>''', unsafe_allow_html=True)
                elif cand_level == "Intermediate":
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''', unsafe_allow_html=True)
                elif cand_level == "Experienced":
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!</h4>''', unsafe_allow_html=True)

                ## Skills Analyzing and Recommendation
                st.subheader("**Skills Recommendation 💡**")
                keywords = st_tags(label='### Your Current Skills', text='See our skills recommendation below', value=resume_data['skills'], key='1')

                if reco_field != "NA":
                    recommended_skills = recommend_skills(reco_field)
                    
                    if reco_field == "Data Science":
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                    elif reco_field == "Web Development":
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                    elif reco_field == "Android Development":
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                    elif reco_field == "IOS Development":
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                    elif reco_field == "UI-UX Development":
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                    
                    st_tags(label='### Recommended skills for you.', text='Recommended skills generated from System', value=recommended_skills, key='2')
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h5>''', unsafe_allow_html=True)
                    
                    # Get course recommendations based on field
                    course_list = get_course_recommendations(reco_field)
                    rec_course = course_recommender(course_list)
                else:
                    st.warning("** Currently our tool only predicts and recommends for Data Science, Web, Android, IOS and UI/UX Development**")
                    recommended_skills = ['No Recommendations']
                    st_tags(label='### Recommended skills for you.', text='Currently No Recommendations', value=recommended_skills, key='3')
                    st.markdown('''<h5 style='text-align: left; color: #092851;'>Maybe Available in Future Updates</h5>''', unsafe_allow_html=True)
                    rec_course = "Sorry! Not Available for this Field"

                ## Resume Improvement Tips
                st.subheader("**Resume Improvement Tips 🥂**")
                
                # Display the improvement reasons from the LLM analysis
                improvement_reasons = analysis_result.get("improvement_reasons", [])
                for i, reason in enumerate(improvement_reasons, 1):
                    st.markdown(f'''<h5 style='text-align: left; color: #000000;'>{i}. {reason}</h5>''', unsafe_allow_html=True)

                # Save data to CSV
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                insert_data(str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, str(latlong), city, state, country,
                            act_name, act_mail, act_mob, resume_data['name'], resume_data['email'], str(resume_score),
                            timestamp, str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                            str(recommended_skills), str(rec_course), pdf_name)

                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                st.balloons()

            else:
                st.error('Something went wrong.. Unable to analyze the resume.')                


    ###### CODE FOR FEEDBACK SIDE ######
    # elif choice == 'Feedback':   
    #     ts = time.time()
    #     cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    #     cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    #     timestamp = str(cur_date+'_'+cur_time)

    #     with st.form("my_form"):
    #         st.write("Feedback form")            
    #         feed_name = st.text_input('Name')
    #         feed_email = st.text_input('Email')
    #         feed_score = st.slider('Rate Us From 1 - 5', 1, 5)
    #         comments = st.text_input('Comments')
    #         submitted = st.form_submit_button("Submit")
    #         if submitted:
    #             insertf_data(feed_name, feed_email, feed_score, comments, timestamp)    
    #             st.success("Thanks! Your Feedback was recorded.")
    #             st.balloons()    

    #     # Load feedback data
    #     if os.path.exists('user_feedback.csv'):
    #         df_feedback = pd.read_csv('user_feedback.csv')
    #         st.header("**User's Feedback Data**")
    #         st.dataframe(df_feedback)
    #     else:
    #         st.warning("No feedback data found.")

    #     if os.path.exists('user_feedback.csv'):
    #         plotfeed_data = pd.read_csv('user_feedback.csv')
    #         labels = plotfeed_data.feed_score.unique()
    #         values = plotfeed_data.feed_score.value_counts()
    #         fig = px.pie(values=values, names=labels, title="Chart of User Rating Score From 1 - 5", color_discrete_sequence=px.colors.sequential.Aggrnyl)
    #         st.plotly_chart(fig)


    ###### CODE FOR ABOUT PAGE ######
    # elif choice == 'About':   
    #     st.subheader("**About The Tool - AI RESUME ANALYZER with Alibaba Cloud LLM**")
    #     st.markdown('''
    #         <p align='justify'>A tool which analyzes resumes using Alibaba Cloud's Large Language Model technology to extract information and provide personalized recommendations.</p>
    #         <p align="justify"><b>How to use it: -</b><br/><br/>
    #         <b>User -</b><br/>In the Side Bar choose yourself as user and fill the required fields and upload your resume in pdf format.<br/>
    #         The resume will be analyzed by a powerful language model to extract your skills, experience, and provide recommendations.<br/><br/>
    #         <b>Feedback -</b><br/>A place where user can suggest some feedback about the tool.<br/><br/>
    #         <b>Admin -</b><br/>For login use <b>admin</b> as username and <b>admin@resume-analyzer</b> as password.<br/>
    #         It will load all the required stuffs and perform analysis.
    #         </p><br/><br/>
    #         <p align="justify">Built with 🤍 by <a href="https://dnoobnerd.netlify.app/">Mantap</a> through <a href="https://www.linkedin.com/in/mrbriit/">Dr Bright --(Data Scientist)</a><br>
    #         Modified to use Alibaba Cloud LLM API</p>
    #     ''', unsafe_allow_html=True)


    ###### CODE FOR ADMIN SIDE (ADMIN) ######
    # elif choice == 'Admin':
    #     st.success('Welcome to Admin Side')
    #     ad_user = st.text_input("Username")
    #     ad_password = st.text_input("Password", type='password')

    #     if st.button('Login'):
    #         if ad_user == 'admin' and ad_password == 'admin@resume-analyzer':

    #             if os.path.exists('user_data.csv'):
    #                 df = pd.read_csv('user_data.csv')
    #                 st.header("**User's Data**")
    #                 st.dataframe(df)
    #                 st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)
    #             else:
    #                 st.warning("No user data found.")

    #             if os.path.exists('user_feedback.csv'):
    #                 df_feedback = pd.read_csv('user_feedback.csv')
    #                 st.header("**User Feedback Data**")
    #                 st.dataframe(df_feedback)
    #             else:
    #                 st.warning("No feedback data found.")

    #             # Pie charts for analytics
    #             if os.path.exists('user_data.csv'):
    #                 plot_data = pd.read_csv('user_data.csv')
    #                 labels = plot_data.resume_score.unique()
    #                 values = plot_data.resume_score.value_counts()
    #                 fig = px.pie(df, values=values, names=labels, title='From 1 to 100 💯', color_discrete_sequence=px.colors.sequential.Agsunset)
    #                 st.plotly_chart(fig)

    #                 labels = plot_data.city.unique()
    #                 values = plot_data.city.value_counts()
    #                 fig = px.pie(df, values=values, names=labels, title='Usage Based On City 🌆', color_discrete_sequence=px.colors.sequential.Jet)
    #                 st.plotly_chart(fig)

    #                 labels = plot_data.state.unique()
    #                 values = plot_data.state.value_counts()
    #                 fig = px.pie(df, values=values, names=labels, title='Usage Based on State 🚉', color_discrete_sequence=px.colors.sequential.PuBu_r)
    #                 st.plotly_chart(fig)

    #                 labels = plot_data.country.unique()
    #                 values = plot_data.country.value_counts()
    #                 fig = px.pie(df, values=values, names=labels, title='Usage Based on Country 🌏', color_discrete_sequence=px.colors.sequential.Purpor_r)
    #                 st.plotly_chart(fig)

    #         else:
    #             st.error("Wrong ID & Password Provided")

# Run the app
if __name__ == "__main__":
    run()
